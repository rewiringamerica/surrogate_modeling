# Databricks notebook source
# MAGIC %md # Evaluate CNN Model
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

targets = ['heating', 'cooling']
pred_df = spark.table('ml.surrogate_model.test_predictions').drop('temp_air', 'wind_speed', 'ghi', 'weekend')

# COMMAND ----------

@udf("array<double>")
def APE(prediction, actual):
    return [abs(float(x - y))/y if y != 0 else None for x, y in zip(prediction, actual) ]

# COMMAND ----------

w = Window().partitionBy('building_id').orderBy(F.asc('upgrade_id'))

def element_wise_subtract(a, b):
    return F.expr(f"transform(arrays_zip({a}, {b}), x -> abs(x.{a} - x.{b}))")


pred_df_hvac_process =  (
    pred_df
        .withColumn('hvac', F.col('heating') + F.col('cooling'))
        .withColumn("actual", F.array(targets + ['hvac']))
        .withColumn('prediction', F.array_insert("prediction", 3, F.col('prediction')[0] + F.col('prediction')[1]))
)


pred_df_savings = (
    pred_df_hvac_process
        .withColumn('baseline_heating_fuel', F.first(F.col('heating_fuel')).over(w))
        .withColumn('baseline_ac_type', F.first(F.col('ac_type')).over(w))

        .withColumn('prediction_baseline', F.first(F.col('prediction')).over(w))
        .withColumn('actual_baseline', F.first(F.col('actual')).over(w))

        .withColumn('prediction_savings', element_wise_subtract('prediction', 'prediction_baseline'))
        .withColumn('actual_savings', element_wise_subtract('actual', 'actual_baseline'))

        .withColumn('absolute_error', element_wise_subtract('prediction', 'actual'))
        .withColumn('absolute_percentage_error', APE(F.col('prediction'), F.col('actual')))
        .withColumn('absolute_error_savings', element_wise_subtract('prediction_savings', 'actual_savings'))
        .withColumn('absolute_percentage_error_savings', APE(F.col('prediction_savings'), F.col('actual_savings')))
)

# COMMAND ----------


pred_df_savings.groupby('baseline_heating_fuel', 'baseline_ac_type', 'upgrade_id').agg(
    F.mean(F.col('absolute_percentage_error')[2]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[2]).alias('median_absolute_percentage_error'),
    F.mean(F.col('absolute_error')[2]).alias('mean_absolute_error'),
    F.median(F.col('absolute_error')[2]).alias('median_absolute_error'),
    F.mean(F.col('absolute_percentage_error_savings')[2]).alias('mean_absolute_percentage_error_savings'),
    F.median(F.col('absolute_percentage_error_savings')[2]).alias('median_absolute_percentage_error_savings'),
    F.mean(F.col('absolute_error_savings')[2]).alias('mean_absolute_error_savings'),
    F.median(F.col('absolute_error_savings')[2]).alias('median_absolute_error_savings'),
    ).display()

# COMMAND ----------

pred_df_savings.groupby('heating_fuel').agg(
    F.mean(F.col('absolute_percentage_error')[0]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[0]).alias('median_absolute_percentage_error'),
    ).display()

# COMMAND ----------

# MAGIC %md ## Format for comparing to other models

# COMMAND ----------


