# Databricks notebook source
# MAGIC %md ## Get Bucketed Model Metrics

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import BooleanType, FloatType
import pyspark.sql.functions as F

# COMMAND ----------

#use disk caching to accelerate data reads: https://docs.databricks.com/en/optimizations/disk-cache.html
spark.conf.set("spark.databricks.io.cache.enabled","true")

# COMMAND ----------

#predicted baseline energy consumption by bucket
predicted_bucketed_baseline_consumption = (
    spark.table("housing_profile.baseline_savings_upfront_cost_bucketed")
    .select('id', 'bucket_id', 'end_use', 'kwh_upgrade_median', 'kwh_upgrade_percentile', 'cooling_type', 'baseline_appliance_fuel')
    ).alias('pred')

# baseline energy consumption by building for single family homes in upgrade scenarios
actual_baseline_consumption_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade
    FROM housing_profile.resstock_annual_baseline_consumption_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
    GROUP BY building_id, bucket_id
    """
).alias('true')

sf_detached_buildings = spark.sql("SELECT building_id FROM building_model.resstock_metadata WHERE in_geometry_building_type_acs == 'Single-Family Detached'")

# COMMAND ----------

# Function to check if x is between the two elements of in interval
@F.udf(BooleanType())
def in_interval(x, interval):
    return x >= interval[0] and x <= interval[1]

@F.udf(FloatType())
def absolute_percentage_error(pred, true):
    if true != 0:
        return abs((true - pred) / true)
    else:
        return None
    
@F.udf(FloatType())
def absolute_error(pred, true):
    if true != 0:
        return abs(true - pred)
    else:
        return None

error_by_building_bucket=(
    actual_baseline_consumption_by_building_bucket
            .join(predicted_bucketed_baseline_consumption,F.col('true.bucket_id') == F.col('pred.id'))
            .join(sf_detached_buildings,on = 'building_id')
            .withColumn('type', 
                    F.when(F.col('cooling_type') == 'Heat Pump AC', F.lit("Heat Pump"))
                    .otherwise(F.coalesce('baseline_appliance_fuel', 'cooling_type')))
            .withColumn('absolute_percentage_error', absolute_percentage_error(F.col('kwh_upgrade_median'), F.col('kwh_upgrade')))
            .withColumn('absolute_error', absolute_error(F.col('kwh_upgrade_median'), F.col('kwh_upgrade')))
            .withColumn("in_interval", in_interval(F.col("kwh_upgrade"), F.col('kwh_upgrade_percentile')))
    )

# COMMAND ----------

def evalute_metrics(df, groupby_cols = []):
    metrics = (
        df
            .groupby(*groupby_cols)
            .agg(
                F.mean('absolute_error').alias('Mean Abs Error'),
                F.median('absolute_error').alias('Median Abs Error'),
                (F.median('absolute_percentage_error')*100).alias('Median APE'), 
                (F.mean('absolute_percentage_error')*100).alias('MAPE'), 
            )
    )
    return metrics

# COMMAND ----------

metrics_by_enduse_type = evalute_metrics(
    df = error_by_building_bucket.where(F.col('end_use').isin(['heating', 'cooling'])), 
    groupby_cols=['end_use', 'type']
    )

metrics_by_enduse  = evalute_metrics(
    df = error_by_building_bucket.where(F.col('end_use').isin(['heating', 'cooling'])), 
    groupby_cols=['end_use']
).withColumn('type', F.lit('Total'))

metrics_buckets = metrics_by_enduse_type.unionByName(metrics_by_enduse)

# COMMAND ----------

metrics_buckets.display()

# COMMAND ----------

metrics_buckets.toPandas().to_csv('gs://the-cube/export/surrogate_model_metrics/bucketed.csv', float_format="%.2f")

# COMMAND ----------


