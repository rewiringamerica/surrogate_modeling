# Databricks notebook source
# MAGIC %md
# MAGIC # Move megastock tables from databricks to bigquery
# MAGIC Copied relevant commands from in https://github.com/rewiringamerica/building_profile/blob/dev/src/attom/create_attom_big_query_table.py
# MAGIC
# MAGIC ## Inputs: delta tables on databricks
# MAGIC - `ml.megastock.building_metadata_5m`
# MAGIC - `ml.megastock.building_features_5m`
# MAGIC
# MAGIC ## Outputs: tables on BigQuery
# MAGIC - `cube-machine-learning.ds_api_datasets.megastock_metadata`
# MAGIC - `cube-machine-learning.ds_api_datasets.megastock_features`
# MAGIC

# COMMAND ----------

from google.cloud import bigquery
from google.cloud.bigquery import dbapi

# COMMAND ----------

bq_metadata_write_path = 'cube-machine-learning.ds_api_datasets.megastock_metadata'
building_metadata = spark.table('ml.megastock.building_metadata_5m')


# COMMAND ----------

(building_metadata
    .write
    .format("bigquery")
    .mode("overwrite")
    .option("table", bq_metadata_write_path)
    .option("temporaryGcsBucket", "the-cube")
    .save()
)

# COMMAND ----------

bq_features_write_path = 'cube-machine-learning.ds_api_datasets.megastock_features'
building_features = spark.table('ml.megastock.building_features_5m')

(building_features
    .write
    .format("bigquery")
    .mode("overwrite")
    .option("table", bq_features_write_path)
    .option("temporaryGcsBucket", "the-cube")
    .save()
)


# COMMAND ----------

# check that tables are there
client = bigquery.Client()

# COMMAND ----------

QUERY = f"""select count(*) from `{bq_metadata_write_path}`"""
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

for row in rows:
    print(row) #3,234,218

# COMMAND ----------

QUERY = f"""select count(*) from `{bq_features_write_path}`"""
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

for row in rows:
    print(row) #25,873,744
