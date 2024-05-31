# Databricks notebook source
# MAGIC %md ## Get Bucketed Model Metrics

# COMMAND ----------

# MAGIC %pip install gcsfs==2023.5.0

# COMMAND ----------

import pandas as pd
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
        "kwh_upgrade_percentile",
        "cooling_type",
        "baseline_appliance_fuel",
    )
).alias("pred")

predicted_bucketed_upgrade_consumption = (
    spark.table("housing_profile.project_savings_upfront_cost_bucketed@v70").select(
        "id",
        "bucket_id",
        "end_use",
        "appliance_option",
        "insulation_option",
        "kwh_upgrade_median",
        "kwh_upgrade_percentile",
        "kwh_delta_median",
        "kwh_delta_percentile",
        "cooling_type",
        "baseline_appliance_fuel",
    )
).alias("pred")

# baseline energy consumption by building for single family homes in upgrade scenarios
actual_baseline_consumption_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.resstock_annual_baseline_consumption_by_building_geography_enduse_fuel_bucket
    WHERE acs_housing_type == 'Single-Family'
        AND end_use IN ('heating', 'cooling')
    GROUP BY building_id, bucket_id
    """
).alias("true")

# baseline energy consumption by building for single family homes in upgrade scenarios
actual_consumption_savings_by_building_bucket = spark.sql(
    f"""
    SELECT building_id, bucket_id, SUM(kwh_upgrade) AS kwh_upgrade, SUM(kwh_delta) AS kwh_delta
    FROM housing_profile.resstock_annual_project_savings_by_building_geography_enduse_fuel_bucket@v3
    WHERE acs_housing_type == 'Single-Family'
        AND upgrade_id IN (1, 3, 4)
    GROUP BY building_id, bucket_id
    """
).alias("true")

building_metadata = spark.sql(
    """
    SELECT building_id, in_heating_fuel, in_hvac_cooling_type, in_hvac_cooling_efficiency, in_hvac_heating_efficiency
    FROM building_model.resstock_metadata
    WHERE in_geometry_building_type_acs IN ('Single-Family Detached', 'Single-Family Attached')
    AND in_heating_fuel != 'Other Fuel'
    """
)

pep_projects = spark.sql(
    "SELECT DISTINCT appliance_option, insulation_option, upgrade_id_pep AS upgrade_id  FROM housing_profile.pep_projects@v11"
)

# COMMAND ----------

@F.udf(FloatType())
def absolute_percentage_error(pred, true, eps=1e-3):
    if abs(true) > eps:
        return abs((true - pred) / true) * 100
    else:
        return None


@F.udf(FloatType())
def absolute_error(pred, true):
    return abs(true - pred)


# join buildings to bucket prediction (baseline)
prediction_actual_by_building_bucket_baseline = (
    actual_baseline_consumption_by_building_bucket.join(
        predicted_bucketed_baseline_consumption,
        F.col("true.bucket_id") == F.col("pred.id"),
    )
    .join(building_metadata, on="building_id")
    .drop("bucket_id")
)

# join buildings to bucket predictions (upgrades)
prediction_actual_by_building_bucket_upgrade = (
    actual_consumption_savings_by_building_bucket.join(
        predicted_bucketed_upgrade_consumption,
        F.col("true.bucket_id") == F.col("pred.id"),
    )
    .join(building_metadata, on="building_id")
    .drop("bucket_id")
)


# combine upgrades and baseline predictions, join to projects to get upfrade id, and sum across heating and cooling
hvac_prediction_actual_by_building_upgrade = (
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
)

# compute various metrics
error_by_building_upgrade = (
    hvac_prediction_actual_by_building_upgrade.withColumn(
        "absolute_percentage_error",
        absolute_percentage_error(F.col("kwh_upgrade_median"), F.col("kwh_upgrade")),
    )
    .withColumn(
        "absolute_error",
        absolute_error(F.col("kwh_upgrade_median"), F.col("kwh_upgrade")),
    )
    .withColumn(
        "absolute_percentage_error_savings",
        absolute_percentage_error(F.col("kwh_delta_median"), F.col("kwh_delta")),
    )
    .withColumn(
        "absolute_error_savings",
        absolute_error(F.col("kwh_delta_median"), F.col("kwh_delta")),
    )
)

# COMMAND ----------

# save the predictions to a delta table so that we can plot the building level predictions compared to other models
(
    error_by_building_upgrade.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("ml.surrogate_model.bucketed_sf_hvac_predictions")
)

# COMMAND ----------

def aggregate_metrics(df, groupby_cols):

    # for computing various statistics of interest over all building ids within a bucket
    aggregation_expression = [
        f(F.col(c)).alias(f"{f.__name__}_{c}")
        for f in [F.median, F.mean]
        for c in [
            "absolute_error",
            "absolute_percentage_error",
            "absolute_error_savings",
            "absolute_percentage_error_savings",
        ]
    ]

    return df.groupby(*groupby_cols).agg(*aggregation_expression)

# COMMAND ----------

# aggregate hvac prediction metrics by upgrade and heating type
metrics_by_heating_fuel_upgrade = (
    aggregate_metrics(
        df=error_by_building_upgrade,
        groupby_cols=["baseline_appliance_fuel", "upgrade_id"],
    )
    .withColumnRenamed("baseline_appliance_fuel", "type")
    .withColumn("category", F.lit("heating"))
)
# aggregate hvac prediction metrics by upgrade and ccooling type
metrics_by_cooling_type_upgrade = (
    aggregate_metrics(
        df=error_by_building_upgrade.where(
            F.col("cooling_type") != "Heat Pump"
        ),  # already accounted for in heating summary above
        groupby_cols=["cooling_type", "upgrade_id"],
    )
    .withColumnRenamed("cooling_type", "type")
    .withColumn("category", F.lit("cooling"))
)
# aggregate hvac prediction metrics by upgrade
metrics_by_upgrade = (
    aggregate_metrics(df=error_by_building_upgrade, groupby_cols=["upgrade_id"])
    .withColumn("type", F.lit("Total"))
    .withColumn("category", F.lit("total"))
)
# combine all the various aggregated metrics
metrics_buckets = metrics_by_heating_fuel_upgrade.unionByName(
    metrics_by_cooling_type_upgrade
).unionByName(metrics_by_upgrade)

# COMMAND ----------



# COMMAND ----------

# write out aggegated metrics
metrics_buckets.toPandas().to_csv(
    "gs://the-cube/export/surrogate_model_metrics/bucketed_sf_hvac.csv", index=None
)
