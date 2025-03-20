# Databricks notebook source
# MAGIC %md # Build Surrogate Model Feature Stores
# MAGIC
# MAGIC ### Goal
# MAGIC Transform surrogate model features (building metadata and weather) and write to feature store.
# MAGIC
# MAGIC ### Process
# MAGIC * Transform building metadata into features and subset to features of interest
# MAGIC * Apply upgrade logic to the relevant building metadata building set for each supported upgrade
# MAGIC * Pivot weather data into wide vector format with pkey `weather_file_city` and a 8760-length timeseries vector for each weather feature column
# MAGIC * Write building metadata features and weather features to feature store tables
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC Inputs are read in based on the most recent table versions according to version tagging. We don't necessarily use the the current version in pyproject.toml because the code change in this poetry version may not require modifying the upstream table.
# MAGIC - `ml.surrogate_model.building_simulation_outputs_annual_{MOST_RECENT_VERSION_NUM}`: Building simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.building_metadata__{MOST_RECENT_VERSION_NUM}`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly__{MOST_RECENT_VERSION_NUM}`: Hourly weather data indexed by (weather_file_city, hour datetime)
# MAGIC
# MAGIC ##### Outputs:
# MAGIC Outputs are written based on the current version number of this repo in `pyproject.toml`.
# MAGIC - `gs://the-cube/export/surrogate_model/model_artifacts/{CURRENT_VERSION_NUM}/mappings.json`: Feature mappings to be shared by training and inference
# MAGIC - `ml.surrogate_model.building_features_{CURRENT_VERSION_NUM}`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_features_hourly_{CURRENT_VERSION_NUM}`: Weather features indexed by (weather_file_city) with a 8760-length timeseries vector for each weather feature column
# MAGIC
# MAGIC ---
# MAGIC Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki for access if permission is denied)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Imports
import re
from functools import reduce
import pandas as pd
from pathlib import Path

import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

import src.globals as g
from src.utils import qa_utils, data_io
from src import feature_utils, versioning

# COMMAND ----------

# MAGIC %md ## Feature Transformation

# COMMAND ----------

# MAGIC %md #### Baseline
# MAGIC
# MAGIC Refer to `docs/features_upgrades.md` and [options.tsv](https://github.com/NREL/resstock/blob/run/euss/resources/options_lookup.tsv).
# MAGIC

# COMMAND ----------

# DBTITLE 1,Transform building metadata
# get most recent table version for baseline metadata -- we don't enforce the current version because the code change in this version
# may not affect the upstream table
building_metadata_table_name = versioning.get_most_recent_table_version(g.BUILDING_METADATA_TABLE)
print(building_metadata_table_name)
baseline_building_metadata_transformed = feature_utils.transform_building_features(building_metadata_table_name)

# COMMAND ----------

# MAGIC %md #### Upgrades
# MAGIC
# MAGIC Refer to `docs/features_upgrades.md`, [ResStock docs](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf), and [upgrade.yml](https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_10k.yml).

# COMMAND ----------

# write out working test case data to gcs for dohyo apply upgrade logic to check against
# NOTE: first make sure unit tests in tests/test_feature_utils.py are working

baseline_test_data_fname = "test_baseline_features_input.csv"
upgraded_test_data_fname = "test_upgraded_features.csv"
baseline_test_features = pd.read_csv(f"../tests/{baseline_test_data_fname}")
upgraded_test_features = pd.read_csv(f"../tests/{upgraded_test_data_fname}")
baseline_test_features.to_csv(str(g.GCS_CURRENT_VERSION_ARTIFACT_PATH / baseline_test_data_fname), index=False)
upgraded_test_features.to_csv(str(g.GCS_CURRENT_VERSION_ARTIFACT_PATH / upgraded_test_data_fname), index=False)

# COMMAND ----------

# DBTITLE 1,Build metadata table for all samples and upgrades
building_metadata_upgrades = feature_utils.build_upgrade_metadata_table(baseline_building_metadata_transformed)

# COMMAND ----------

# DBTITLE 1,Drop rows where upgrade was not applied
# get most recent table version for annual outputs to compare against
outputs_most_recent_version_num = versioning.get_most_recent_table_version(g.ANNUAL_OUTPUTS_TABLE, return_version_number_only=True)
building_metadata_applicable_upgrades = feature_utils.drop_non_upgraded_samples(
    building_metadata_upgrades, check_applicability_logic_against_version=outputs_most_recent_version_num
)

# COMMAND ----------

# MAGIC %md #### Summary

# COMMAND ----------

# DBTITLE 1,Make sure there are no Null Features
n_building_upgrade_samples = building_metadata_applicable_upgrades.count()
print(n_building_upgrade_samples)
non_null_df = building_metadata_applicable_upgrades.dropna()
assert (
    non_null_df.count() == n_building_upgrade_samples
), "Null values present, run qa_utils.check_for_null_values(building_metadata_upgrades)"

# COMMAND ----------

# DBTITLE 1,Print out feature counts
pkey_cols = ["weather_file_city", "upgrade_id", "building_id"]
string_columns = [
    field.name
    for field in building_metadata_upgrades.schema.fields
    if isinstance(field.dataType, StringType) and field.name not in pkey_cols
]
non_string_columns = [
    field.name
    for field in building_metadata_upgrades.schema.fields
    if not isinstance(field.dataType, StringType) and field.name not in pkey_cols
]
# count how many distinct vals there are for each categorical feature
distinct_string_counts = building_metadata_upgrades.select([F.countDistinct(c).alias(c) for c in string_columns])
# Collect the results as a dictionary
distinct_string_counts_dict = distinct_string_counts.collect()[0].asDict()
# print the total number of features:
print(f"Building Metadata Features: {len(non_string_columns + string_columns)}")
print(f"\tNumeric Features: {len(non_string_columns)}")
print(f"\tCategorical Features: {len(string_columns)}")
print(f"\tFeature Dimensionality: {len(non_string_columns) + sum(distinct_string_counts_dict.values()) }")

# COMMAND ----------

# MAGIC %md ### Weather Features

# COMMAND ----------

# DBTITLE 1,Weather feature transformation function
def transform_weather_features(table_name) -> DataFrame:
    """
    Read and transform weather timeseries table. Pivot from long format indexed by (weather_file_city, hour)
    to a table indexed by weather_file_city with a 8760 len array timeseries for each weather feature column

    Parameters:
        table_name (str): full table name (catalog.database.table) of the processed weather timeseries table
    Returns:
        DataFrame: wide(ish) format dataframe indexed by weather_file_city with timeseries array for each weather feature
    """
    weather_df = spark.read.table(table_name)
    weather_pkeys = ["weather_file_city"]

    weather_data_arrays = weather_df.groupBy(weather_pkeys).agg(
        *[F.collect_list(c).alias(c) for c in weather_df.columns if c not in weather_pkeys + ["datetime_formatted"]]
    )
    return weather_data_arrays

# COMMAND ----------

# DBTITLE 1,Transform weather features
weather_table_name = versioning.get_most_recent_table_version(g.WEATHER_DATA_TABLE)
print(weather_table_name)
weather_features = transform_weather_features(weather_table_name)

# COMMAND ----------

# DBTITLE 1,Add weather file city index
# fit the string indexer on the weather feature df
weather_file_city_indexer = feature_utils.fit_weather_city_index(df_to_fit=weather_features)
# apply indexer to weather feature df to get a weather_file_city_index column
weather_features_indexed = feature_utils.transform_weather_city_index(
    df_to_transform=weather_features, weather_file_city_indexer=weather_file_city_indexer
)
# apply indexer to building metadata feature df to get a weather_file_city_index column
building_metadata_with_weather_index = feature_utils.transform_weather_city_index(
    df_to_transform=building_metadata_applicable_upgrades, weather_file_city_indexer=weather_file_city_indexer
)

# COMMAND ----------

# MAGIC %md ## Write out training data mapping artifacts
# MAGIC
# MAGIC Write out mapping artifacts that are needed by downstream dohyo

# COMMAND ----------

# Get the mapping weather city to index as a dict -- this is needed for embedding lookup in dohyo
weather_city_to_index = {label: i for i, label in enumerate(weather_file_city_indexer.labels)}

# COMMAND ----------

# Create an indexer for climate zone -- this is needed for partitioning by climate zone in megastock
climate_zone_indexer = feature_utils.create_string_indexer(
    spark.table(building_metadata_table_name), column_name="ashrae_iecc_climate_zone_2004"
)
# Get the climate zone to index (1 indexed) as a dict
climate_zone_to_index = {label: i for i, label in enumerate(climate_zone_indexer.labels, 1)}

# COMMAND ----------

# write to artifacts
data_io.write_json(
    g.GCS_CURRENT_VERSION_ARTIFACT_PATH / "mappings.json",
    data={"climate_zone_to_index": climate_zone_to_index, "weather_city_to_index": weather_city_to_index},
    overwrite=True,
)

# COMMAND ----------

# MAGIC %md ## Create Feature Store
# MAGIC

# COMMAND ----------

# MAGIC %md ### Create/Use schema in catalog in the Unity Catalog MetaStore
# MAGIC
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.

# COMMAND ----------

# DBTITLE 1,Set up catalog and schema
# MAGIC %sql
# MAGIC -- Use existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS surrogate_model;
# MAGIC
# MAGIC USE SCHEMA surrogate_model;

# COMMAND ----------

# MAGIC %md ### Create/modify the feature stores

# COMMAND ----------

# DBTITLE 1,Create a FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = f"{g.BUILDING_FEATURE_TABLE}_{g.CURRENT_VERSION_NUM}"
print(table_name)
fe.create_table(
    name=table_name,
    primary_keys=["building_id", "upgrade_id", "weather_file_city"],
    df=building_metadata_with_weather_index,
    schema=building_metadata_with_weather_index.schema,
    description="building metadata features",
)

# COMMAND ----------

# DBTITLE 1,Write out weather data feature store
table_name = f"{g.WEATHER_FEATURE_TABLE}_{g.CURRENT_VERSION_NUM}"
print(table_name)
fe.create_table(
    name=table_name,
    primary_keys=["weather_file_city"],
    df=weather_features_indexed,
    schema=weather_features_indexed.schema,
    description="hourly weather timeseries array features",
)
