# Databricks notebook source
# MAGIC %md ## Get Bucketed Model Metrics

# COMMAND ----------

from typing import List 

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import BooleanType, FloatType
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# use disk caching to accelerate data reads: https://docs.databricks.com/en/optimizations/disk-cache.html
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# predicted baseline energy consumption by bucket
predicted_bucketed_baseline_consumption = (
    spark.table("housing_profile.baseline_consumption_upfront_cost_bucketed").select(
        "id",
        "bucket_id",
        "end_use",
        "appliance_option",
        "insulation_option",
        "kwh_upgrade_median",
        "cooling_type",
        "baseline_appliance_fuel",
    )
).alias("pred_baseline")

predicted_bucketed_upgrade_consumption = (
    spark.table("housing_profile.all_project_savings_bucketed").select(
        "id",
        "bucket_id",
        "end_use",
        "appliance_option",
        "insulation_option",
        "kwh_upgrade_median",
        "kwh_delta_median",
        "cooling_type",
        "baseline_appliance_fuel",
    )
).alias("pred_upgrade")

# baseline energy consumption by building for single family homes
actual_baseline_consumption_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.resstock_annual_baseline_consumption_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
    GROUP BY building_id, bucket_id
    """
).alias("true_baseline")

# upgrade energy consumption by building for single family homes
actual_consumption_savings_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.all_resstock_annual_project_savings_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
        AND upgrade_id IN (0, 1, 3, 4, 6, 8, 11.05, 11.06, 13.01)
    GROUP BY building_id, bucket_id
    """
).alias("true_upgrade")

building_metadata = spark.sql(
    """
    SELECT building_id, in_heating_fuel, in_hvac_cooling_type, in_hvac_cooling_efficiency, in_hvac_heating_efficiency
    FROM building_model.resstock_metadata
    WHERE in_geometry_building_type_acs IN ('Single-Family Detached', 'Single-Family Attached')
    AND in_heating_fuel != 'Other Fuel'
    """
)

# assign dryer and range to diff upgrade ids so they don't get grouped when summing over end use
pep_projects = (
    spark.table('housing_profile.pep_projects')
    .withColumn('upgrade_id', 
                F.when((F.col('upgrade_id_pep') == 8) & (F.col('end_use_pep') == 'dryer'), F.lit(8.1))
                .when((F.col('upgrade_id_pep') == 8) & (F.col('end_use_pep') == 'range'), F.lit(8.2))
                .otherwise(F.col('upgrade_id_pep'))
    )
    .select('appliance_option', 'insulation_option', 'upgrade_id').distinct()
)

# COMMAND ----------

@F.udf(FloatType())
def absolute_percentage_error(pred, true, eps=1e-3):
    if abs(true) > eps:
        return abs((true - pred) / true) * 100
    else:
        return None

# join buildings to bucket prediction (baseline)
prediction_actual_by_building_bucket_baseline = (
    actual_baseline_consumption_by_building_bucket.join(
        predicted_bucketed_baseline_consumption,
        F.col("true_baseline.bucket_id") == F.col("pred_baseline.id"),
    )
    .join(building_metadata, on="building_id")
    .drop("bucket_id")
)

# join buildings to bucket predictions (upgrades)
prediction_actual_by_building_bucket_upgrade = (
    actual_consumption_savings_by_building_bucket.join(
        predicted_bucketed_upgrade_consumption,
        F.col("true_upgrade.bucket_id") == F.col("pred_upgrade.id"),
    )
    .join(building_metadata, on="building_id")
    .drop("bucket_id")
)


# combine upgrades and baseline predictions, join to projects to get upfrade id, and sum across heating and cooling
prediction_actual_by_building_upgrade = (
    prediction_actual_by_building_bucket_baseline.unionByName(
        prediction_actual_by_building_bucket_upgrade, allowMissingColumns=True
    )
    .withColumn(
        "cooling_type",
        F.when(F.col("in_hvac_cooling_type") == "Heat Pump AC", F.lit("Heat Pump"))
        .when(F.col("in_hvac_cooling_type") == "None", F.lit("No Cooling"))
        .when(
            (F.col("in_hvac_cooling_efficiency") == "Shared Cooling"),
            F.lit("Shared Cooling"),
        )
        .otherwise(F.col("in_hvac_cooling_type")),
    )
    .withColumn(
        "baseline_appliance_fuel",
        F.when(F.col("cooling_type") == "Heat Pump", F.lit("Heat Pump"))
        .when(F.col("in_heating_fuel") == "Electricity", F.lit("Electric Resistance"))
        .when(F.col("in_heating_fuel") == "None", F.lit("No Heating"))
        .when(
            F.col("in_hvac_heating_efficiency") == "Shared Heating",
            F.lit("Shared Heating"),
        )
        .otherwise(F.col("in_heating_fuel")),
    )
    .join(pep_projects, on=["appliance_option", "insulation_option"])
    .groupby("building_id", "upgrade_id", "baseline_appliance_fuel", "cooling_type")
    .agg(
        *[
            F.sum(c).alias(c)
            for c in [
                "kwh_upgrade_median",
                "kwh_upgrade",
                "kwh_delta_median",
                "kwh_delta",
            ]
        ]
    )
).replace('Natural Gas', 'Methane Gas')

# compute various metrics
error_by_building_upgrade = (
    prediction_actual_by_building_upgrade.withColumn(
        "absolute_percentage_error",
        absolute_percentage_error(F.col("kwh_upgrade_median"), F.col("kwh_upgrade")),
    )
    .withColumn(
        "absolute_error",
        F.round(F.abs(F.col("kwh_upgrade_median") - F.col("kwh_upgrade")))
    )
    .withColumn(
        "absolute_percentage_error_savings",
        absolute_percentage_error(F.col("kwh_delta_median"), F.col("kwh_delta")),
    )
    .withColumn(
        "absolute_error_savings",
        F.round(F.abs(F.col("kwh_delta_median") - F.col("kwh_delta")))
    )
)

# COMMAND ----------

# save the predictions to a delta table so that we can plot the building level predictions compared to other models
(
    error_by_building_upgrade.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("ml.surrogate_model.bucketed_sf_predictions")
)

# COMMAND ----------

def aggregate_metrics(df: DataFrame, groupby_cols: List[str]):
    """
    Aggregates metrics for a given DataFrame by specified grouping columns.
    This function calculates the median and mean of absolute error, absolute percentage error,
    absolute error savings, and absolute percentage error savings. The results are rounded as specified.
    Args:
        df (DataFrame): The DataFrame containing prediction savings and errors.
        groupby_cols (list or str): A list of column names to group the DataFrame by.
    Returns:
        DataFrame: A DataFrame aggregated by the specified groupby columns with the calculated metrics.
    """
    aggregation_expression = [
        F.round(f(F.col(colname)), round_precision).alias(f"{f.__name__}_{colname}")
        for f in [F.median, F.mean]
        for colname, round_precision in [
            ("absolute_error", 0),
            ("absolute_percentage_error", 1),
            ("absolute_error_savings", 0),
            ("absolute_percentage_error_savings", 1),
        ]
    ]

    return df.groupby(*groupby_cols).agg(*aggregation_expression)

# COMMAND ----------

# aggregate hvac prediction metrics by upgrade and heating type
metrics_by_heating_fuel_upgrade = aggregate_metrics(
    df=error_by_building_upgrade,
    groupby_cols=["baseline_appliance_fuel", "upgrade_id"],
).withColumnRenamed("baseline_appliance_fuel", "type")

# aggregate hvac prediction metrics by upgrade and cooling type for baseline only
# where heat pumps are already covered by the heating rows above
metrics_by_cooling_type_upgrade = aggregate_metrics(
    df=error_by_building_upgrade.where(F.col("cooling_type") != "Heat Pump").where(
        F.col("upgrade_id") == 0
    ),
    groupby_cols=["cooling_type", "upgrade_id"],
).withColumnRenamed("cooling_type", "type")

# aggregate hvac prediction metrics by upgrade
metrics_by_upgrade = aggregate_metrics(
    df=error_by_building_upgrade, groupby_cols=["upgrade_id"]
).withColumn("type", F.lit("Total"))

# combine all the various aggregated metrics
metrics_buckets = metrics_by_heating_fuel_upgrade.unionByName(
    metrics_by_cooling_type_upgrade
).unionByName(metrics_by_upgrade)

# COMMAND ----------

metrics_buckets.display()

# COMMAND ----------

# write out aggegated metrics
metrics_buckets.toPandas().to_csv(
    "gs://the-cube/export/surrogate_model_metrics/bucketed_sf.csv", index=None
)
