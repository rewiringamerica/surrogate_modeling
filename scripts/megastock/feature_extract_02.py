# Databricks notebook source
dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

import sys

from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as F

from dmutils import qa_utils
from src import feature_utils

# COMMAND ----------

N_SAMPLE_TAG = dbutils.widgets.get("n_sample_tag")

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature store portion

# COMMAND ----------

baseline_building_metadata_transformed = feature_utils.transform_building_features(f"ml.megastock.building_metadata_{N_SAMPLE_TAG}")

# COMMAND ----------

# Check for differences between the possible values of categorical features in MegaStock and in ResStock training features
comparison_dict = qa_utils.compare_dataframes_string_values(
    spark.table('ml.surrogate_model.building_features').where(F.col('upgrade_id')==0),
    baseline_building_metadata_transformed)

# NOTE: if there are differences, these should be fixed upstream in the creation of 'ml.megastock.building_metadata_*'
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

# DBTITLE 1,write out sampled building metadata with weather city index
# table_name = f"ml.megastock.building_metadata_upgrades_{N_SAMPLE_TAG}"
# building_metadata_upgrades.write.saveAsTable(table_name, mode='overwrite', overwriteSchema=True)
# spark.sql(f'OPTIMIZE {table_name}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving out building features

# COMMAND ----------


fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = f"ml.megastock.building_features_{N_SAMPLE_TAG}"
df = building_metadata_upgrades
if spark.catalog.tableExists(table_name):
    fe.write_table(name=table_name, df=df, mode="merge")
else:
    fe.create_table(
        name=table_name,
        primary_keys=["building_id", "upgrade_id", "weather_file_city"],
        df=df,
        schema=df.schema,
        description="megastock building metadata features",
    )
