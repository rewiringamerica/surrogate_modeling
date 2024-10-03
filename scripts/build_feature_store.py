# Databricks notebook source
# MAGIC %md # Build Surrogate Model Feature Stores
# MAGIC
# MAGIC ### Goal
# MAGIC Transform surrogate model features (building metadata and weather) and write to feature store.
# MAGIC
# MAGIC ### Process
# MAGIC * Transform building metadata into features and subset to features of interest
# MAGIC * Apply upgrade logic to building metadata features
# MAGIC * Pivot weather data into wide vector format with pkey `weather_file_city` and a 8760-length timeseries vector for each weather feature column
# MAGIC * Write building metadata features and weather features to feature store tables
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC - `ml.surrogate_model.building_simulation_outputs_annual`: Building simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Hourly weather data indexed by (weather_file_city, hour datetime)
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `ml.surrogate_model.building_features`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_features_hourly`: Weather features indexed by (weather_file_city) with a 8760-length timeseries vector for each weather feature column
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
# from typing import Dict

from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from src.dmutils import sumo, data_cleaning

# COMMAND ----------

# MAGIC %md ## Feature Transformation

# COMMAND ----------

# MAGIC %md #### Baseline
# MAGIC
# MAGIC Refer to [Notion Page](https://www.notion.so/rewiringamerica/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=4#086a1d050b8c4094ad10e2275324668b) and [options.tsv](https://github.com/NREL/resstock/blob/run/euss/resources/options_lookup.tsv).
# MAGIC

# COMMAND ----------

# DBTITLE 1,Transform building metadata
baseline_building_metadata_transformed = sumo.transform_building_features('ml.surrogate_model.building_metadata')

# COMMAND ----------

# MAGIC %md #### Upgrades
# MAGIC
# MAGIC Refer to [Notion Page](https://www.notion.so/rewiringamerica/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=4#3141dfeeb07144da9fe983b2db13b6d3), [ResStock docs](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf), and [upgrade.yml](https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_10k.yml).

# COMMAND ----------

# DBTITLE 1,Build metadata table for all samples and upgrades
building_metadata_upgrades = sumo.build_upgrade_metadata_table(baseline_building_metadata_transformed)

# COMMAND ----------

# DBTITLE 1,Drop rows where upgrade was not applied
building_metadata_applicable_upgrades = sumo.drop_non_upgraded_samples(
    building_metadata_upgrades,
    check_applicability_logic=True)

# COMMAND ----------

# MAGIC %md #### Summary

# COMMAND ----------

# DBTITLE 1,Make sure there are no Null Features
data_cleaning.check_null_values(building_metadata_applicable_upgrades)

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
distinct_string_counts = building_metadata_upgrades.select(
    [F.countDistinct(c).alias(c) for c in string_columns]
)
# Collect the results as a dictionary
distinct_string_counts_dict = distinct_string_counts.collect()[0].asDict()
# print the total number of features:
print(f"Building Metadata Features: {len(non_string_columns + string_columns)}")
print(f"\tNumeric Features: {len(non_string_columns)}")
print(f"\tCategorical Features: {len(string_columns)}")
print(
    f"\tFeature Dimensionality: {len(non_string_columns) + sum(distinct_string_counts_dict.values()) }"
)

# COMMAND ----------

# MAGIC %md ### Weather Features

# COMMAND ----------

# DBTITLE 1,Weather feature transformation function
def transform_weather_features() -> DataFrame:
    """
    Read and transform weather timeseries table. Pivot from long format indexed by (weather_file_city, hour)
    to a table indexed by weather_file_city with a 8760 len array timeseries for each weather feature column

    Returns:
        DataFrame: wide(ish) format dataframe indexed by weather_file_city with timeseries array for each weather feature
    """
    weather_df = spark.read.table("ml.surrogate_model.weather_data_hourly")
    weather_pkeys = ["weather_file_city"]

    weather_data_arrays = weather_df.groupBy(weather_pkeys).agg(
        *[
            F.collect_list(c).alias(c)
            for c in weather_df.columns
            if c not in weather_pkeys + ["datetime_formatted"]
        ]
    )
    return weather_data_arrays

# COMMAND ----------

# DBTITLE 1,Transform weather features
weather_data_transformed = transform_weather_features()

# COMMAND ----------

# DBTITLE 1,Create and apply string indexer to generate weather file city index
# Create the StringIndexer
indexer = StringIndexer(
    inputCol="weather_file_city",
    outputCol="weather_file_city_index",
    stringOrderType="alphabetAsc",
)

weather_file_city_indexer = indexer.fit(weather_data_transformed)

weather_data_indexed = weather_file_city_indexer.transform(
    weather_data_transformed
).withColumn("weather_file_city_index", F.col("weather_file_city_index").cast("int"))

building_metadata_applicable_upgrades_with_weather_file_city_index = (
    weather_file_city_indexer.transform(
        building_metadata_applicable_upgrades
    ).withColumn(
        "weather_file_city_index", F.col("weather_file_city_index").cast("int")
    )
)

# COMMAND ----------

# MAGIC %md ## Create Feature Store
# MAGIC

# COMMAND ----------

# MAGIC %md ### Create/Use schema in catalog in the Unity Catalog MetaStore
# MAGIC
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.
# MAGIC To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

# DBTITLE 1,Check if you have access on ml catalog
# MAGIC %sql
# MAGIC -- if you do not see `ml` listed here, this means you do not have permissions
# MAGIC SHOW CATALOGS

# COMMAND ----------

# DBTITLE 1,Set up catalog and schema
# MAGIC %sql
# MAGIC -- Use existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS surrogate_model;
# MAGIC USE SCHEMA surrogate_model;

# COMMAND ----------

# MAGIC %md ### Create/modify the feature stores

# COMMAND ----------

# MAGIC %sql
# MAGIC -- the following code will upsert. To overwrite, uncomment this line to first drop the table
# MAGIC -- DROP TABLE ml.surrogate_model.building_features

# COMMAND ----------

# DBTITLE 1,Create a FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = "ml.surrogate_model.building_features"
df = building_metadata_applicable_upgrades_with_weather_file_city_index
if spark.catalog.tableExists(table_name):
    fe.write_table(name=table_name, df=df, mode="merge")
else:
    fe.create_table(
        name=table_name,
        primary_keys=["building_id", "upgrade_id", "weather_file_city"],
        df=df,
        schema=df.schema,
        description="building metadata features",
    )

# COMMAND ----------

# DBTITLE 1,Write out weather data feature store
table_name = "ml.surrogate_model.weather_features_hourly"
df = weather_data_indexed
if spark.catalog.tableExists(table_name):
    fe.write_table(
        name=table_name,
        df=df,
        mode="merge",
    )
else:
    fe.create_table(
        name=table_name,
        primary_keys=["weather_file_city"],
        df=df,
        schema=df.schema,
        description="hourly weather timeseries array features",
    )
