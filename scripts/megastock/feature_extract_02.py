# Databricks notebook source
dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

import sys

from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F

import src.globals as g
from src.utils import qa_utils
from src import feature_utils, versioning

# COMMAND ----------

# get number of samples to use
N_SAMPLE_TAG = dbutils.widgets.get("n_sample_tag").lower()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature store portion

# COMMAND ----------

# get most recent table version for baseline metadata -- we don't enforce the current version because the code change
# in this poetry version may not require modifying the upstream table
building_metadata_table_name = versioning.get_most_recent_table_version(f'{g.MEGASTOCK_BUILDING_METADATA_TABLE}_{N_SAMPLE_TAG}')
building_metadata_table_name

# COMMAND ----------

baseline_building_features = feature_utils.transform_building_features(building_metadata_table_name)

# COMMAND ----------

# Check for differences between the possible values of categorical features in MegaStock and in ResStock training features
comparison_dict = qa_utils.compare_dataframes_string_values(
    spark.table(versioning.get_most_recent_table_version(g.BUILDING_FEATURE_TABLE)).where(F.col('upgrade_id').isin([0.01])).drop('building_set'),
    baseline_building_features.drop('building_set'))

# NOTE: if there are differences, these should be fixed upstream in the creation of 'ml.megastock.building_metadata_*'
# This may fail on weather city for small megastocks (e.g, 10K)
assert (
    len(comparison_dict) == 0
), f"MegaStock features have different categorical values than training features \n {comparison_dict}"

# COMMAND ----------

n_building_upgrade_samples = baseline_building_features.count()
print(n_building_upgrade_samples)
non_null_df = baseline_building_features.dropna()
assert non_null_df.count() == n_building_upgrade_samples, "Null values present, run qa_utils.check_for_null_values(baseline_building_metadata_transformed)"

# COMMAND ----------

# add weather city index
baseline_building_features_with_weather_index = feature_utils.transform_weather_city_index(
    weather_file_city_indexer= feature_utils.fit_weather_city_index(spark.table(g.WEATHER_FEATURE_TABLE)),
    df_to_transform = baseline_building_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving out building features

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = f"{g.MEGASTOCK_BUILDING_FEATURE_TABLE}_{N_SAMPLE_TAG}_{g.CURRENT_VERSION_NUM}"
print(table_name)
fe.create_table(
    name=table_name,
    primary_keys=["building_id", "weather_file_city"],
    df=baseline_building_features_with_weather_index,
    schema=baseline_building_features_with_weather_index.schema,
    description="megastock building metadata features",
)
