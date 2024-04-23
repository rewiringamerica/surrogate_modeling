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



# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType, FloatType

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
import tensorflow as tf

import os
# fix cublann OOM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.config.list_physical_devices("GPU")

# COMMAND ----------

from model_db import Model
from datagen_db import DataGenerator, load_inference_data

# COMMAND ----------

model = Model(name="surrogate_model")

# COMMAND ----------

# #~25s to load full dataset into memory (without joining weather and bm features)
_, _, test_data = load_inference_data()

# COMMAND ----------

batch_pred = model.score_batch(test_data = test_data, targets = DataGenerator.consumption_group_dict.keys())

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


df_metrics_combined = metrics_by_enduse_type.unionByName(metrics_by_enduse).toPandas()

# COMMAND ----------

df_metrics_combined.to_csv('gs://the-cube/export/surrogate_model_metrics/cnn.csv', index=False)
