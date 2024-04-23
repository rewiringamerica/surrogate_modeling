# Databricks notebook source
# MAGIC %md # Model Training
# MAGIC
# MAGIC ### Goal
# MAGIC Train deep learning model to predict energy a building's HVAC energy consumption
# MAGIC
# MAGIC ### Process
# MAGIC * Transform building metadata into features and subset to features of interest
# MAGIC * Pivot weather data into wide vector format with pkey `weather_file_city` and a 8670-length timeseries vector for each weather feature column
# MAGIC * Write building metadata features and weather features to feature store tables
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs: 
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Hourly weather data indexed by (weather_file_city, hour datetime)
# MAGIC
# MAGIC ##### Outputs: 
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Weather data indexed by (weather_file_city) with a 8670-length timeseries vector for each weather feature column
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Outstanding
# MAGIC
# MAGIC #### Future Work
# MAGIC - Add upgrades to the building metadata table
# MAGIC - Extend building metadata features to cover those related to all end uses and to SF Attatched homes 
# MAGIC - More largely, updates to the feature table should merge and not overwrite, and in general transformation that are a hyperparameter of the model (i.e, that we may want to vary in different models) should be done downstream of this table. Sorting out exactly which transformations should happen in each of the `build_dataset`, `build_feature_store` and `model_training` files is still a WIP. 
# MAGIC
# MAGIC ---
# MAGIC ### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - Node type: Single Node. Because of [this issue](https://kb.databricks.com/en_US/libraries/apache-spark-jobs-fail-with-environment-directory-not-found-error), worker nodes cannot access the directory needed to run inference on a keras trained model, meaning that the `score_batch()` function throws and OSError. Rather than dealing with the permissions errors, for now I am just using a single node cluster as a workaround.
# MAGIC - Will work on GPU cluster
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)

# COMMAND ----------

# DBTITLE 1,Install tensorflow
# install tensorflow if not installed on cluster
%pip install tensorflow==2.15.0.post1
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import
# Reflect package changes without reimporting
%load_ext autoreload
%autoreload 2

import itertools
import numpy as np
import math
import os
import pandas as pd
from typing import Dict, Tuple

import pyspark.sql.functions as F


from pyspark.sql import DataFrame

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
import tensorflow as tf

import keras
from tensorflow.keras import layers, models

from model_db import Model
from datagen_db import DataGenerator, create_dataset

# fix cublann OOM
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# check that GPU is available
tf.config.list_physical_devices("GPU")

# COMMAND ----------

# MAGIC %md ## Train a model with Feature Engineering in Unity Catalog

# COMMAND ----------

model = Model(name="test")

# COMMAND ----------

str(model)

# COMMAND ----------

# #~25s to load full dataset into memory (without joining weather and bm features)
train_gen, val_gen, test_data = create_dataset(n_subset=100)

# COMMAND ----------

# print out the features to make sure they look right
train_gen[0][0]

# COMMAND ----------

keras_model = model.fit_model(train_gen = train_gen, val_gen=val_gen, epochs=2)

# COMMAND ----------

# test out model predictions
results = keras_model.predict(val_gen[0][0])
np.hstack([results[c] for c in train_gen.targets])

# COMMAND ----------

# MAGIC %md ## Batch scoring
# MAGIC Use `score_batch` to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key columns. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

pred_df = model.score_batch(test_data = test_data, targets = train_gen.targets)
