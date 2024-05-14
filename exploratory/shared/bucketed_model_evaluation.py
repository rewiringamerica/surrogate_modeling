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
    spark.table("housing_profile.baseline_consumption_upfront_cost_bucketed")
    .select('id', 'bucket_id', 'end_use', 'appliance_option', 'insulation_option', 'kwh_upgrade_median', 'kwh_upgrade_percentile', 'cooling_type', 'baseline_appliance_fuel')
    ).alias('pred')

predicted_bucketed_upgrade_consumption = (
    spark.table("housing_profile.project_savings_upfront_cost_bucketed@v70")
    .select('id', 'bucket_id', 'end_use', 'appliance_option', 'insulation_option', 'kwh_upgrade_median', 'kwh_upgrade_percentile', 
            'kwh_delta_median', 'kwh_delta_percentile', 'cooling_type', 'baseline_appliance_fuel')
    ).alias('pred')

# baseline energy consumption by building for single family homes in upgrade scenarios
actual_baseline_consumption_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.resstock_annual_baseline_consumption_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
    GROUP BY building_id, bucket_id
    """
).alias('true')

# baseline energy consumption by building for single family homes in upgrade scenarios
actual_consumption_savings_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.resstock_annual_project_savings_by_building_geography_enduse_fuel_bucket@v3
    WHERE acs_housing_type == 'Single-Family'
        AND upgrade_id IN (1, 3, 4)
    GROUP BY building_id, bucket_id
    """
).alias('true')

sf_detached_buildings = spark.sql("SELECT building_id FROM building_model.resstock_metadata WHERE in_geometry_building_type_acs == 'Single-Family Detached'")

pep_projects = spark.sql('SELECT DISTINCT appliance_option, insulation_option, upgrade_id_pep AS upgrade_id  FROM housing_profile.pep_projects@v11')

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

error_by_building_bucket_baseline=(
    actual_baseline_consumption_by_building_bucket
            .join(predicted_bucketed_baseline_consumption,F.col('true.bucket_id') == F.col('pred.id'))
            .join(sf_detached_buildings,on = 'building_id')
            .drop('bucket_id')
    )

error_by_building_bucket_upgrade=(
    actual_consumption_savings_by_building_bucket 
            .join(predicted_bucketed_upgrade_consumption,F.col('true.bucket_id') == F.col('pred.id'))
            .join(sf_detached_buildings,on = 'building_id')
            .drop('bucket_id')
    )

error_by_building_upgrade =(
    error_by_building_bucket_baseline
        .unionByName(error_by_building_bucket_upgrade, allowMissingColumns=True)
        .join(pep_projects, on = ['appliance_option', 'insulation_option'])
        .withColumn('type', 
                    F.when(F.col('cooling_type') == 'Heat Pump AC', F.lit("Heat Pump"))
                    .otherwise(F.coalesce('baseline_appliance_fuel', 'cooling_type')))
        .withColumn('absolute_percentage_error', absolute_percentage_error(F.col('kwh_upgrade_median'), F.col('kwh_upgrade')))
        .withColumn('absolute_error', absolute_error(F.col('kwh_upgrade_median'), F.col('kwh_upgrade')))
        .withColumn('absolute_percentage_error_savings', absolute_percentage_error(F.col('kwh_delta_median'), F.col('kwh_delta')))
        .withColumn('absolute_error_savings', absolute_error(F.col('kwh_delta_median'), F.col('kwh_delta')))

)

# COMMAND ----------



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
                
                F.mean('absolute_error_savings').alias('Mean Abs Error - Savings'),
                F.median('absolute_error_savings').alias('Median Abs Error - Savings'),
                (F.median('absolute_percentage_error_savings')*100).alias('Median APE - Savings'), 
                (F.mean('absolute_percentage_error_savings')*100).alias('MAPE - Savings'), 
            )
    )
    return metrics

# COMMAND ----------

metrics_by_enduse_type_upgrade = evalute_metrics(
    df = error_by_building_upgrade.where(F.col('end_use').isin(['heating', 'cooling'])), 
    groupby_cols=['end_use', 'type', 'upgrade_id']
    )

metrics_by_enduse_upgrade  = evalute_metrics(
    df = error_by_building_upgrade.where(F.col('end_use').isin(['heating', 'cooling'])), 
    groupby_cols=['end_use', 'upgrade_id']
).withColumn('type', F.lit('Total'))


metrics_by_end_use  = evalute_metrics(
    df = error_by_building_upgrade.where(F.col('end_use').isin(['heating', 'cooling'])), 
    groupby_cols=['end_use']
).withColumn('type', F.lit('Total')).withColumn('upgrade_id', F.lit('Total'))

metrics_buckets = metrics_by_enduse_type_upgrade.unionByName(metrics_by_enduse_upgrade).unionByName(metrics_by_end_use)

# COMMAND ----------

metrics_buckets.toPandas().to_csv('gs://the-cube/export/surrogate_model_metrics/bucketed.csv', float_format="%.2f", index=None)

# COMMAND ----------


