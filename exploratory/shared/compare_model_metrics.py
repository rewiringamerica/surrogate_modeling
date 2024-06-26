# Databricks notebook source
# MAGIC %md ## Compare Model Metrics

# COMMAND ----------

import pandas as pd

# COMMAND ----------

bucket_metrics = pd.read_csv('gs://the-cube/export/surrogate_model_metrics/bucketed.csv', keep_default_na=False)
cnn_metrics = pd.read_csv('gs://the-cube/export/surrogate_model_metrics/cnn/ml.surrogate_model.sf_detatched_hvac_baseline_v1.csv', keep_default_na=False)
ff_metrics = pd.read_csv('gs://the-cube/export/surrogate_model_metrics/feed_forward.csv', keep_default_na=False)

# COMMAND ----------

bucket_metrics['method'] = 'bucket'
cnn_metrics['method'] = 'cnn'
ff_metrics['method'] = 'feed forward'

# COMMAND ----------

metrics_combined = pd.concat([bucket_metrics, ff_metrics,cnn_metrics]).melt(
        id_vars = ['end_use', 'type', 'method'], 
        value_vars=['Median APE', 'MAPE', 'Median Abs Error',  'Mean Abs Error'],
        var_name='metric'
    )

metrics_combined = metrics_combined.pivot(
    index = ['end_use', 'type'],
    columns = ['metric', 'method'], 
    values = 'value')

metrics_combined 

# COMMAND ----------

metrics_combined.sort_values(['end_use', 'type']).to_csv('gs://the-cube/export/surrogate_model_metrics/comparison.csv', float_format = '%.2f')

# COMMAND ----------

metrics_combined.sort_values(['end_use', 'type'])

# COMMAND ----------


