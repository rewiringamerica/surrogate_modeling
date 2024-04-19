# Databricks notebook source
# MAGIC %md #Model Testing
# MAGIC ### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - Node type: Single Node. Because of [this issue](https://kb.databricks.com/en_US/libraries/apache-spark-jobs-fail-with-environment-directory-not-found-error), worker nodes cannot access the directory needed to run inference on a keras trained model, meaning that the `score_batch()` function throws and OSError. Rather than dealing with the permissions errors, for now I am just using a single node cluster as a workaround.
# MAGIC - Will work on GPU cluster
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki for access)

# COMMAND ----------

# install tensorflow if not installed on cluster
%pip install tensorflow==2.15.0.post1 gcsfs
dbutils.library.restartPython()

# COMMAND ----------

import os
# fix cublann OOM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
import tensorflow as tf

tf.config.list_physical_devices("GPU")

# COMMAND ----------

# create a FeatureEngineeringClient.
fe = FeatureEngineeringClient()

mlflow.set_registry_uri('databricks-uc')

#target grouping
consumption_group_dict = {
    'heating' : [
        'electricity__heating_fans_pumps',
        'electricity__heating_hp_bkup',
        'electricity__heating',
        'fuel_oil__heating_hp_bkup',
        'fuel_oil__heating',
        'natural_gas__heating_hp_bkup',
        'natural_gas__heating',
        'propane__heating_hp_bkup',
        'propane__heating'],
    'cooling' : [
        'electricity__cooling_fans_pumps',
        'electricity__cooling']
}

targets = list(consumption_group_dict.keys())

model_name = "ml.surrogate_model.surrogate_model"

# COMMAND ----------

# Read in the "raw" data which contains the prediction target and the keys needed to join to the feature tables. 
# Right now this is kind of hacky since we need to join to the bm table to do the required train data filtering
sum_str = ', '.join([f"{'+'.join(v)} AS {k}" for k, v in consumption_group_dict.items()])

raw_data = spark.sql(f"""
                     SELECT B.building_id, B.upgrade_id, B.weather_file_city, {sum_str}
                     FROM ml.surrogate_model.annual_outputs O
                     LEFT JOIN ml.surrogate_model.building_metadata_features B 
                        ON B.upgrade_id = O.upgrade_id AND B.building_id == O.building_id
                     WHERE O.upgrade_id = 0
                        AND sqft < 8000
                        AND occupants <= 10
                     """)

train_data, test_data = raw_data.randomSplit(weights=[0.8,0.2], seed=42)

# COMMAND ----------

# MAGIC %md ## Batch scoring
# MAGIC Use `score_batch` to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key columns. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = mlflow.tracking.client.MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

model_name = "ml.surrogate_model.surrogate_model"
# batch inference on small set of held out test set
latest_model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_model_version}"

batch_pred = fe.score_batch(model_uri=model_uri, df=test_data, result_type=ArrayType(DoubleType()))
for i, target in enumerate(targets):
    batch_pred = batch_pred.withColumn(f"{target}_pred", F.col('prediction')[i])

# COMMAND ----------

@udf(returnType=FloatType())
def APE(pred:float, true:float) -> float:
    if true == 0:
        return None
    return abs(pred - true)/true

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

batch_pred_postprocess = (
    batch_pred
        .replace({'AC' : 'Central AC'}, subset = 'ac_type')
        .withColumn('heating_fuel', 
                    F.when(F.col('ac_type') == 'Heat Pump', F.lit('Heat Pump'))
                    .otherwise(F.col('heating_fuel')))
        .withColumn('hvac', F.col('heating') + F.col('cooling'))
        .melt(
            ids = ['heating_fuel', 'ac_type', 'prediction'], 
            values = ['heating', 'cooling', 'hvac'],
            valueColumnName='true', 
            variableColumnName='end_use'
        )
        .withColumn('type', 
                F.when(F.col('end_use') == 'cooling', F.col('ac_type'))
                .otherwise(F.col('heating_fuel'))
        )
        .withColumn('pred', #floor predictions at 0
                F.when(F.col('end_use') == 'heating', F.greatest(F.col('prediction')[0], F.lit(0)))
                .when(F.col('end_use') == 'cooling',F.greatest(F.col('prediction')[1], F.lit(0)))
                .otherwise(F.greatest(F.col('prediction')[1] + F.col('prediction')[0], F.lit(0)))
        )
        .withColumn('absolute_error', F.abs(F.col('pred') -  F.col('true')))
        .withColumn('absolute_percentage_error', APE(F.col('pred'), F.col('true')))
)

# COMMAND ----------

metrics_by_enduse_type = evalute_metrics(
    df = batch_pred_postprocess.where(F.col('end_use') != 'hvac'), 
    groupby_cols = ['end_use' ,'type']
)

metrics_by_enduse = evalute_metrics(
    df = batch_pred_postprocess.where(F.col('end_use') != 'hvac'), 
    groupby_cols = ['end_use']
).withColumn('type', F.lit('Total'))


df_metrics_combined = df_metrics.unionByName(df_metrics_total).toPandas()

# COMMAND ----------

df_metrics_combined.to_csv('gs://the-cube/export/surrogate_model_metrics/cnn.csv', index=False)
