# Databricks notebook source
# MAGIC %md # Feature Store for Surrogate Model 
# MAGIC ### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - Node type: Single Node. Because of [this issue](https://kb.databricks.com/en_US/libraries/apache-spark-jobs-fail-with-environment-directory-not-found-error), worker nodes cannot access the directory needed to run inference on a keras trained model, meaning that the `score_batch()` function throws and OSError. Rather than dealing with the permissions errors, for now I am just using a single node cluster as a workaround.
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)

# COMMAND ----------

# install tensorflow if not installed on cluster
# %pip install tensorflow
# dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

import mlflow
from mlflow.tracking.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import MeanAbsoluteError

# COMMAND ----------

# features we are and targets currently using for testing -- probably will store in a class at some point
building_metadata_features = ["sqft", "occupants"]
target_cols = ["site_energy__total"]

# not including weekend/weekday for now
weather_features = [
    "temp_air",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "ghi",
    "dni",
    "diffuse_horizontal_illum",
]

# COMMAND ----------

# MAGIC %md ## Train a model with Feature Engineering in Unity Catalog

# COMMAND ----------

# create a FeatureEngineeringClient.
fe = FeatureEngineeringClient()

# COMMAND ----------

#select only a small subset of features for now
# and define the lookup keys that will be used to join with the inputs at train/inference time
building_metadata_feature_lookups = [
    FeatureLookup(
        table_name="ml.surrogate_model.building_metadata_features",
        feature_names=building_metadata_features,
        lookup_key=["building_id", "upgrade_id"], #lo
    ),
]

weather_feature_lookups = [
    FeatureLookup(
        table_name="ml.surrogate_model.weather_features",
        feature_names=weather_features,
        lookup_key=["weather_file_city"],
    ),
]

# COMMAND ----------

# Read in the "raw" data which contains the prediction target and the keys needed to join to the feature tables. 
# We might want to do whatever filtering here (e.g, subset fo SF homes)
# but would need to make modifications since we removed building features from this table
raw_data = spark.sql(f"SELECT building_id, upgrade_id, weather_file_city, {','.join(target_cols)} FROM ml.surrogate_model.annual_outputs WHERE upgrade_id == 0")
#       .where(F.col('geometry_building_type_acs') == 'Single-Family Detached')
#       .where(F.col('vacancy_status') == 'Occupied')
#       .where(F.col('sqft') < 8000)
#       .where(F.col('occupants') < 11)

# COMMAND ----------

# takes ~1m for 10K samples on single node cluster
def load_data(raw_data, feature_lookups, target_cols, n_subset=100):

    # exclude column used to just join to weather data
    exclude_columns = ["weather_file_city", "building_id", "upgrade_id"]

    # Create the training set that includes the raw input data merged with corresponding features from both feature tables
    training_set = fe.create_training_set(
        df=raw_data.limit(n_subset),
        feature_lookups=feature_lookups,
        label=target_cols,
        exclude_columns=exclude_columns,
    )

    # Load the TrainingSet into a dataframe which can be passed into keras for training a model
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop(target_cols, axis=1)
    y = training_pd[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, training_set


X_train, X_test, y_train, y_test, training_set = load_data(
    raw_data=raw_data,
    feature_lookups=building_metadata_feature_lookups + weather_feature_lookups,
    target_cols=target_cols,
    n_subset=100,
)
X_train.head()

# COMMAND ----------

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "ml.surrogate_model.surrogate_model_test"

client = MlflowClient()

try:
    client.delete_registered_model(model_name)  # Delete the model if already created
except:
    None

# COMMAND ----------

# MAGIC %md
# MAGIC The code in the next cell trains a keras model and logs the model with the Feature Engineering in UC.
# MAGIC
# MAGIC The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (`mlflow.sklearn.autolog(log_models=False)`); this is because the model is logged using `fe.log_model`.

# COMMAND ----------

#this allows us to apply pre/post processing to the inference data
class SurrogateModelingWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model):
        self.model = trained_model

    def preprocess_input(self, model_input):
        # X_train_weather = tf.convert_to_tensor(
        #     np.dstack([np.vstack(X_train[f].values) for f in weather_features])
        # )
        # X_test_weather = tf.convert_to_tensor(
        #     np.dstack([np.vstack(X_train[f].values) for f in weather_features])
        # )
        model_input = tf.convert_to_tensor(model_input[building_metadata_features])
        return model_input

    # def postprocess_result(self, results):
    #     """Return post-processed results.
    #     Creates a set of fare ranges
    #     and returns the predicted range."""

    #     return [
    #         "$0 - $9.99" if result < 10 else "$10 - $19.99" if result < 20 else " > $20"
    #         for result in results
    #     ]

    def predict(self, context, model_input):
        processed_df = self.preprocess_input(model_input.copy())
        return self.model.predict(processed_df)

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
#mlflow.tensorflow.autolog(log_models=False)
# mlflow.sklearn.autolog(log_models=False)
def train_model(X_train, X_test, y_train, y_test):

    #ignoring these for now, but should be able to feed these to the CNN
    X_train_weather = tf.convert_to_tensor(
        np.dstack([np.vstack(X_train[f].values) for f in weather_features])
    )
    X_test_weather = tf.convert_to_tensor(
        np.dstack([np.vstack(X_train[f].values) for f in weather_features])
    )

    #convert to tensors and just subet to building model features for now
    # should probably use a common function to apply same pre-processing as SurrogateModelingWrapper.preprocess_result()
    X_train = tf.convert_to_tensor(X_train[building_metadata_features])
    X_test = tf.convert_to_tensor(X_test[building_metadata_features])
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test.values)

    model = Sequential(
        [
            InputLayer(X_train.shape[1]),
            Dense(256, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    model.compile(loss="mae", optimizer="adam")

    params = {"epochs": 40, "batch_size": 256}
    history = model.fit(X_train, y_train, verbose=2, validation_split=0.2, **params)

    return model


model = train_model(X_train, X_test, y_train, y_test)

pyfunc_model = SurrogateModelingWrapper(model)

with mlflow.start_run() as run:
    fe.log_model(
        model=pyfunc_model,
        artifact_path="pyfunc_surrogate_model_prediction",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=model_name,
    )

# COMMAND ----------

# MAGIC %md ## Batch scoring
# MAGIC Use `score_batch` to apply a packaged Feature Engineering in UC model to new data for inference. The input data only needs the primary key columns. The model automatically looks up all of the other feature values from the feature tables.

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
latest_model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_model_version}"
batch_pred = fe.score_batch(model_uri=model_uri, df=raw_data)
batch_pred.display()
