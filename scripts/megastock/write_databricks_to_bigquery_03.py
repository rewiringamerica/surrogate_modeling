# Databricks notebook source
# MAGIC %md
# MAGIC # Move megastock tables from databricks to bigquery
# MAGIC * Reads in megastock metadata and feature table
# MAGIC * Adds an `__m` suffix to each column in the metadata table so we can distinguish feature and metadata column when combined in the same table. 
# MAGIC * Add an integer version of the column `ashrae_iecc_climate_zone_2004__m` to the metadata table so we can partition on it. 
# MAGIC * Subset features to only baseline (upgrade = 0) since we will recompute feature transformations locally
# MAGIC * Combine the features and metadata table into one table and write out to BQ
# MAGIC * A optimized table is then created by partitioning on an integer version of climate zone, and clustered on (heating_fuel, geometry_building_type_acs, geometry_floor_area, vintage). This is done as a separate step because the delta -> bq api does not allow for clustering. 
# MAGIC
# MAGIC Also note that computation of integer version of climate zone with partitioning and clustering can be done in one step in BQ using [this query](https://console.cloud.google.com/bigquery?inv=1&invt=AbjEtA&project=cube-machine-learning&ws=!1m17!1m4!1m3!1scube-machine-learning!2sbquxjob_2edfc0a9_19388cf6673!3sUS!1m4!16m3!1m1!1scube-machine-learning!3e12!1m6!12m5!1m3!1scube-machine-learning!2sus-central1!3sd37b8ec3-4973-4dc4-92cb-591280f6d453!2e1). 
# MAGIC
# MAGIC
# MAGIC ## Inputs: delta tables on databricks
# MAGIC Inputs are read in based on the current version number of this repo in `pyproject.toml`.
# MAGIC - `ml.megastock.building_metadata_{n_sample_tag}_{version_num}`
# MAGIC - `ml.megastock.building_features_{n_sample_tag}_{version_num}`
# MAGIC
# MAGIC ## Outputs: tables on BigQuery
# MAGIC Outputs are written based on the current version number of this repo in `pyproject.toml`.
# MAGIC - `cube-machine-learning.ds_api_datasets.megastock_combined_baseline_{n_sample_tag}_{version_num}`
# MAGIC

# COMMAND ----------

dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

from itertools import chain
from pathlib import Path

from google.cloud import bigquery
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

import src.globals as g
from src.utils.data_io import read_json
from src.versioning import get_most_recent_table_version

# COMMAND ----------

# get number of samples to use
N_SAMPLE_TAG = dbutils.widgets.get("n_sample_tag").lower()

# COMMAND ----------

# Initialize a BigQuery client
client = bigquery.Client()

# set up paths to write to
bq_project = "cube-machine-learning"
bq_dataset = "ds_api_datasets"
bq_megastock_table = f"megastock_combined_baseline_{N_SAMPLE_TAG}_{g.CURRENT_VERSION_NUM}"
bq_write_path = f"{bq_project}.{bq_dataset}.{bq_megastock_table}"

# COMMAND ----------

# read in data
building_metadata = spark.table(get_most_recent_table_version(f"{g.MEGASTOCK_BUILDING_METADATA_TABLE}_{N_SAMPLE_TAG}"))
building_features = spark.table(get_most_recent_table_version(f"{g.MEGASTOCK_BUILDING_FEATURE_TABLE}_{N_SAMPLE_TAG}"))
# read in climate zone to int mapping from artifacts in gcp
climate_zone_to_index = read_json(g.GCS_CURRENT_VERSION_ARTIFACT_PATH / "mappings.json")["climate_zone_to_index"]

# COMMAND ----------

# Define UDF
cz_mapping_expr = F.create_map([F.lit(x) for x in chain(*climate_zone_to_index.items())])

# Apply UDF to create new column
building_metadata_with_int_cz = building_metadata.withColumn(
    "climate_zone_int", cz_mapping_expr[F.col("ashrae_iecc_climate_zone_2004")]
)

# COMMAND ----------

# Add "__m" suffix to all columns in metadata table except "building_id"
building_metadata_renamed = building_metadata_with_int_cz.select(
    [F.col(c).alias(f"{c}__m") if c != "building_id" else F.col(c) for c in building_metadata_with_int_cz.columns]
)

# COMMAND ----------

# Join the tables on building_id
combined_df_baseline = building_metadata_renamed.join(building_features, on="building_id", how="inner")

# COMMAND ----------

# write out table to big query
(
    combined_df_baseline.write.format("bigquery")
    .mode("overwrite")
    .option("table", bq_write_path)
    .option("temporaryGcsBucket", "the-cube")
    .save()
)

# COMMAND ----------

# optimize the table by partitioning and clustering
query = f"""
CREATE TABLE `{bq_write_path}_optimized`
PARTITION BY RANGE_BUCKET(climate_zone_int__m, GENERATE_ARRAY(1, {len(climate_zone_to_index)+1}, 1))
CLUSTER BY  heating_fuel__m, geometry_building_type_acs__m, geometry_floor_area__m, vintage__m AS
SELECT *,
FROM `{bq_write_path}`
"""

# Run the query
query_job = client.query(query)
# Wait for the query to complete
query_job.result()

# Drop non-optimized table
query = f"""DROP TABLE `{bq_write_path}`"""
query_job = client.query(query)
query_job.result()

# Rename optimized table to original name
query = f"""
ALTER TABLE `{bq_write_path}_optimized`
RENAME TO `{bq_megastock_table}`;
"""
query_job = client.query(query)
query_job.result()

print("Partitioned table created successfully.")

# COMMAND ----------

# check that tables are there
QUERY = f"""select count(*) from `{bq_write_path}`"""
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

for row in rows:
    print(row)

# COMMAND ----------

# check that tables are there
QUERY = f"""select climate_zone_int__m from `{bq_write_path}` WHERE building_id=1"""
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish
