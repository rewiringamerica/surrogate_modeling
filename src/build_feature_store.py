# Databricks notebook source
# MAGIC %md # Feature Store for Surrogate Model 
# MAGIC ### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)

# COMMAND ----------

import pyspark.sql.functions as F

from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# MAGIC %md ## Load dataset

# COMMAND ----------

building_metadata = spark.read.table("ml.surrogate_model.building_metadata")
weather_data = spark.read.table("ml.surrogate_model.weather_data")

# COMMAND ----------

# MAGIC %md ## Feature Preprocessing

# COMMAND ----------

def transform_building_features(df):
  """
  Temp placeholder for transforming building_metadata features. 
  Will eventually copy everything over from  _get_building_metadata() in datagen.py
  """
  building_metadata_transformed = (
    df
      .withColumn('occupants', F.when(F.col('occupants') == '10+', 11).otherwise(F.col('occupants').cast("int")))
  )
  return building_metadata_transformed
    

def transform_weather_features(df):
  """
  Transform weather timeseries table
  """
  weather_pkeys = ["weather_file_city"]
  # Solution 1: Store all weather ts as a 8670 len array within each weather feature column
  weather_data_arrays = df.groupBy(weather_pkeys).agg(
      *[F.collect_list(c).alias(c) for c in df.columns if c not in weather_pkeys]
  )

  # Solution 2: Store each weather feature in a different feature store, with each timestamp as a seperate feature column
  # This appears to be incredibly slow to read in when calling fe.create_dataset() so going with solution #1 for now.
  # #transpose the weather data into format (weather city, feature name, timestamp rows (x8670))
  # weather_data_transpose = (
  #   weather_data
  #     .melt(
  #       ids = ['weather_file_city', 'datetime_formatted'],
  #       values=weather_features,
  #       variableColumnName='variable',
  #       valueColumnName='value')
  #     .groupBy('weather_file_city', 'variable')
  #     .pivot('datetime_formatted')
  #     .agg(F.first('value'))
  #     )
  return weather_data_arrays


# COMMAND ----------

weather_data_transformed = transform_weather_features(df = weather_data)

# COMMAND ----------

building_metadata_transformed = transform_building_features(building_metadata)

# COMMAND ----------

# MAGIC %md ## Create new schema in catalog in the Unity Catalog MetaStore
# MAGIC
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.
# MAGIC To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- if you do not see `ml` listed here, this means you do not have permissions
# MAGIC SHOW CATALOGS

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS surrogate_model;
# MAGIC USE SCHEMA surrogate_model;

# COMMAND ----------

# MAGIC %md ## Create the feature table
# MAGIC
# MAGIC See [API reference for GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html).

# COMMAND ----------

# create a FeatureEngineeringClient.
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS ml.surrogate_model.building_metadata_features;
# MAGIC DROP TABLE IF EXISTS ml.surrogate_model.weather_features

# COMMAND ----------

fe.create_table(
    name="ml.surrogate_model.building_metadata_features",
    primary_keys=["building_id", "upgrade_id"],
    df=building_metadata_transformed,
    schema=building_metadata_transformed.schema,
    description="building metadata features",
)


fe.create_table(
    name="ml.surrogate_model.weather_features",
    primary_keys=["weather_file_city"],
    df=weather_data_transformed,
    schema=weather_data_transformed.schema,
    description="weather timeseries array features",
)

# Alternative for solution #2
# for f in weather_features:
#     fe.create_table(
#         name=f'ml.surrogate_model.weather_{f}',
#         primary_keys=["weather_file_city"],
#         df=weather_data_transpose.where(F.col('variable') == f).drop('variable'),
#         schema=weather_data_transpose.drop('variable').schema,
#         description=f"{f} weather timeseries features"
#     )
