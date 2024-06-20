# Databricks notebook source
# MAGIC %md ## Get Bucketed Model Metrics

# COMMAND ----------

from typing import List 

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Column

from pyspark.sql.window import Window

# COMMAND ----------

# use disk caching to accelerate data reads: https://docs.databricks.com/en/optimizations/disk-cache.html
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# predicted baseline energy consumption by bucket
predicted_bucketed_baseline_consumption = (
    spark.table("housing_profile.baseline_consumption_upfront_cost_bucketed")
    .where(~(F.col('project_category').isin(['Range', 'Dryer']) & (F.col('baseline_appliance_fuel') == 'None')))
    .select(
        "id",
        "bucket_id",
        "end_use",
        "kwh_upgrade_median",
        "appliance_option", 
        "insulation_option", 
    )
).alias("bucket_baseline")

# predicted upgrade energy consumption and savings by bucket
predicted_bucketed_upgrade_consumption = (
    spark.table("housing_profile.all_project_savings_bucketed")
    .where(~(F.col('project_category').isin(['Range', 'Dryer']) & (F.col('baseline_appliance_fuel') == 'None')))
    .select(
        "id",
        "bucket_id",
        "end_use",
        "kwh_upgrade_median",
        "kwh_delta_median",
    )
).alias("bucket_upgrade")

# baseline energy consumption by building for single family homes
building_to_baseline_bucket_with_actual_consumption = spark.sql(
    f"""
    SELECT building_id, bucket_id, upgrade_id, SUM(kwh_upgrade) AS kwh_upgrade
    FROM housing_profile.resstock_annual_baseline_consumption_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
    GROUP BY building_id, bucket_id,  upgrade_id
    """
).alias("building_baseline")

# upgrade energy consumption by building for single family homes
building_to_upgrade_bucket = spark.sql(
    f"""
    SELECT DISTINCT building_id, bucket_id, upgrade_id
    FROM housing_profile.all_resstock_annual_project_savings_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
        AND upgrade_id != 9
    """
).alias("building_upgrade")

# COMMAND ----------

# join buildings to bucket prediction (baseline)
baseline_prediction_by_building_bucket = (
    building_to_baseline_bucket_with_actual_consumption.join(
        predicted_bucketed_baseline_consumption,
        F.col("building_baseline.bucket_id") == F.col("bucket_baseline.id"),
    )
    .drop("bucket_id")
)

# join buildings to bucket predictions (upgrades)
upgrade_prediction_by_building_bucket = (
    building_to_upgrade_bucket.join(
        predicted_bucketed_upgrade_consumption,
        F.col("building_upgrade.bucket_id") == F.col("bucket_upgrade.id"),
    )
    .drop("bucket_id")
)

# combine upgrades and baseline predictions, join to projects to get upfrade id, and sum across heating and cooling
prediction_actual_by_building_upgrade = (
    baseline_prediction_by_building_bucket.unionByName(
        upgrade_prediction_by_building_bucket, allowMissingColumns=True
    )
    # assign dryer and range to diff upgrade ids to match surrogate model outputs
    .withColumn('upgrade_id', 
                F.when((F.col('upgrade_id') == 8) & (F.col('end_use') == 'dryer'), F.lit(8.1))
                .when((F.col('upgrade_id') == 8) & (F.col('end_use') == 'range'), F.lit(8.2))
                .otherwise(F.col('upgrade_id'))
    )
    .groupby("building_id", "upgrade_id")
    .agg(
        *[
            F.sum(c).alias(c)
            for c in [
                "kwh_upgrade_median",
                "kwh_delta_median",
                "kwh_upgrade"
            ]
        ]
    )
    #this will only be non-null for baseline buckets
    .withColumn("absolute_error", F.abs(F.col("kwh_upgrade") - F.col("kwh_upgrade_median")))
    .drop("kwh_upgrade")
)

# COMMAND ----------

# save the predictions to a delta table so that we can plot the building level predictions compared to other models
(
    prediction_actual_by_building_upgrade.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("ml.surrogate_model.bucketed_sf_predictions")
)
