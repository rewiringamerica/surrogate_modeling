# Databricks notebook source
dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

import sys

from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F

from src.globals import CURRENT_VERSION_NUM
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
building_metadata_table_name = versioning.get_most_recent_table_version(f'ml.megastock.building_metadata_{N_SAMPLE_TAG}')
building_metadata_table_name

# COMMAND ----------

baseline_building_metadata_transformed = feature_utils.transform_building_features(building_metadata_table_name)

# COMMAND ----------

# Check for differences between the possible values of categorical features in MegaStock and in ResStock training features
comparison_dict = qa_utils.compare_dataframes_string_values(
    spark.table(f'ml.surrogate_model.building_features_{CURRENT_VERSION_NUM}').where(F.col('upgrade_id')==0),
    baseline_building_metadata_transformed)

# NOTE: if there are differences, these should be fixed upstream in the creation of 'ml.megastock.building_metadata_*'
# This may fail on weather city for small megastocks (e.g, 10K)
assert (
    len(comparison_dict) == 0
), f"MegaStock features have different categorical values than training features \n {comparison_dict}"

# COMMAND ----------

building_metadata_upgrades = feature_utils.build_upgrade_metadata_table(baseline_building_metadata_transformed)

# COMMAND ----------

building_metadata_upgrades = feature_utils.add_weather_city_index(building_metadata_upgrades)

# COMMAND ----------

building_metadata_applicable_upgrades = feature_utils.drop_non_upgraded_samples(
    building_metadata_upgrades, check_applicability_logic=False
)

# COMMAND ----------

# building_metadata_upgrades.count()

# COMMAND ----------

n_building_upgrade_samples = building_metadata_upgrades.count()
print(n_building_upgrade_samples)
non_null_df = building_metadata_upgrades.dropna()
assert non_null_df.count() == n_building_upgrade_samples, "Null values present, run qa_utils.check_for_null_values(building_metadata_upgrades)"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving out building features

# COMMAND ----------

fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = f"ml.megastock.building_features_{N_SAMPLE_TAG}_{CURRENT_VERSION_NUM}"
print(table_name)
fe.create_table(
    name=table_name,
    primary_keys=["building_id", "upgrade_id", "weather_file_city"],
    df=building_metadata_upgrades,
    schema=building_metadata_upgrades.schema,
    description="megastock building metadata features",
)
