# Databricks notebook source
# MAGIC %md # Model Training
# MAGIC
# MAGIC ### Goal
# MAGIC Train deep learning model to predict a homes total energy consumption
# MAGIC
# MAGIC ### Process
# MAGIC * Load in train/val/test sets containing targets and feature keys
# MAGIC * Initialize data generators on train/val sets which pulls in weather and building model features
# MAGIC * Train and log model
# MAGIC * Test that the trained model works
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC Inputs are read in based on the most recent table versions according to version tagging. We don't necessarily use the the current version in pyproject.toml because the code change in this poetry version may not require modifying the upstream table.
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Weather data indexed by (weather_file_city) with a 8760-length timeseries vector
# MAGIC - `ml.surrogate_model.building_upgrade_simulation_outputs_annual`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC
# MAGIC ##### Outputs:
# MAGIC If in test mode (DEBUG = True):
# MAGIC
# MAGIC The trained model is just logged to the unity catalog with the run id and current version number, but as of now is not registered due to issue with signature enforcement slowing down inference.
# MAGIC
# MAGIC If in production mode mode (DEBUG = False):
# MAGIC - `gs://the-cube/export/surrogate_model/model_artifacts/{CURRENT_VERSION_NUM}/model.keras`: the trained keras model
# MAGIC - `gs://the-cube/export/surrogate_model/model_artifacts/{CURRENT_VERSION_NUM}/features_targets_upgrades.json`: some parameters of trained model
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Outstanding
# MAGIC
# MAGIC #### Future Work
# MAGIC * Figure out how to register the model with a signature without slowing down inference
# MAGIC * Handle retracing issues
# MAGIC * Install model dependencies using the logged requirements.txt file
# MAGIC * Get checkpointing to work
# MAGIC
# MAGIC ---
# MAGIC #### Cluster/ User Requirements
# MAGIC - Can be run on CPU or GPU, with 2x speedup on GPU
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)
# MAGIC

# COMMAND ----------

# we need a newer version of MLFlow in order to use a custom loss
%pip install mlflow==2.13.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set debug mode
# this controls the training parameters, with test mode on a much smaller training set for fewer epochs
dbutils.widgets.dropdown("mode", "test", ["test", "production"])
DEBUG = dbutils.widgets.get("mode") == "test"
print(DEBUG)

# COMMAND ----------

# DBTITLE 1,Allow GPU growth
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# COMMAND ----------

# DBTITLE 1,Import
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC from pathlib import Path
# MAGIC import pyspark.sql.functions as F
# MAGIC import tensorflow as tf
# MAGIC import shutil
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.types import DoubleType
# MAGIC from tensorflow import keras
# MAGIC from typing import Tuple, Dict
# MAGIC
# MAGIC from src.utils.data_io import write_json
# MAGIC from src.datagen import DataGenerator, load_data
# MAGIC from src.surrogate_model import SurrogateModel
# MAGIC
# MAGIC # list available GPUs
# MAGIC tf.config.list_physical_devices("GPU")

# COMMAND ----------

# DBTITLE 1,Set experiment location
# location to store the experiment runs if in production mode:
# specifying this allows for models trained in notebook or job to be written to same place
EXPERIMENT_LOCATION = "/Shared/surrogate_model/"

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

# DBTITLE 1,Load data
# load data using most recent versions of tables, and the data params in current version config
train_data, val_data, test_data = load_data(n_train=1000 if DEBUG else None)

# COMMAND ----------

# DBTITLE 1,Initialize train/val data generators
train_gen = DataGenerator(train_data=train_data)
val_gen = DataGenerator(train_data=val_data)

# COMMAND ----------

# DBTITLE 1,Inspect data gen output for one batch
if DEBUG:
    print("FEATURES:")
    print(train_gen[0][0])
    print("\n OUTPUTS:")
    print(train_gen[0][1])

# COMMAND ----------

# MAGIC %md ## Train model

# COMMAND ----------

# DBTITLE 1,Define wrapper class for processing at inference time
class SurrogateModelingWrapper(mlflow.pyfunc.PythonModel):
    """
    A wrapper class that applies the pre/post processing to the data at inference time,
    where the pre-processing must transform the inputs to match the format used during training.
    This is then packaged up as part of the model, and will automatically be applied when
    running inference with the packaged mlflow model.

    Attributes:
        - model: The trained mlflow keras model
        - building_features (list of str) : List of building features that the model was trained on
        - weather_features (list of str) : List of weather features that the model was trained on
        - targets (list of str) : List of consumption group targets
    """

    def __init__(self, trained_model, building_features, weather_features, targets):
        """
        Parameters:
        - trained_model: The trained mlflow keras model
        See class attributes for details on other params.
        """
        self.model = trained_model
        self.building_features = building_features
        self.weather_features = weather_features
        self.targets = targets

    def preprocess_input(self, model_input: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocesses the model input of P features over N samples

        Parameters:
        - model_input (pd.Dataframe): The input features for the model of shape [N, P].

        Returns:
        - The preprocessed feature data in format {feature_name(str) : np.array [N,]}
        """
        return self.convert_feature_dataframe_to_dict(model_input)

    def postprocess_result(self, results: Dict[str, np.ndarray], feature_df: pd.DataFrame) -> np.ndarray:
        """
        Postprocesses the model results for N samples over M targets by clipping at 0
        and setting targets to 0 if the home does not have an applaince using that fuel.

        Parameters:
        - results (dict of {str: np.ndarray}): The outputs of the model in format {target_name (str) : np.ndarray [N,]}
        - feature_df (pd.DataFrame): The features for the samples of shape [N, *]. Only the features flagging which fuels are present are used here.

        Returns:
        - np.ndarray of shape [N, M]

        """
        for fuel in self.targets:
            if fuel == "electricity":
                results[fuel] = results[fuel].flatten()
            else:
                # null out fuel target if fuel is not present in any appliance in the home
                results[fuel] = np.where(
                    ~feature_df[f"has_{fuel}_appliance"],
                    np.nan,
                    results[fuel].flatten(),
                )
        # stack into N x M array and clip at 0
        return np.clip(np.vstack(list(results.values())).T, a_min=0, a_max=None)

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the model for N samples over M targets.

        Parameters:
        - context (Any): Ignored here. It's a placeholder for additional data or utility methods.
        - model_input (pd.Dataframe): The input features for the model of shape [N, P]

        Returns:
        - np.ndarray of shape [N, M]
        """
        processed_df = self.preprocess_input(model_input)
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df, model_input)

    def convert_feature_dataframe_to_dict(self, feature_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Converts the feature data from a pandas dataframe to a dictionary.

        Parameters:
        - feature_df: : The input features for the model of shape [N, P] where feature columns
                        for weather features contain len 8760 arrays.

        Returns:
        - The preprocessed feature data in format {feature_name (str) : np.array of shape [N]
        """
        return {col: np.array(feature_df[col]) for col in self.building_features + ["weather_file_city_index"]}

# COMMAND ----------

# DBTITLE 1,Initialize model
if DEBUG:
    sm = SurrogateModel(name="test")
else:  # named based on current version
    sm = SurrogateModel()

# COMMAND ----------

# DBTITLE 1,Train model
# Train keras model and log the model with the Feature Engineering in UC. Note that right we are skipping registering the model in the UC-- this requires storing the signature, which for unclear reasons, is slowing down inference more than 10x.

# Init FeatureEngineering client
fe = FeatureEngineeringClient()

# Set the activation function and numeric data type for the model's layers
layer_params = {
    "dtype": train_gen.dtype,
    "kernel_initializer": "he_normal",
}
# skip logging signatures for now...
# signature_df = train_gen.training_set.load_df().select(train_gen.building_features + train_gen.targets + train_gen.weather_features).limit(1).toPandas()
# signature=mlflow.models.infer_signature(model_input = signature_df[train_gen.building_features + train_gen.weather_features], model_output = signature_df[train_gen.targets])

mlflow.tensorflow.autolog(log_every_epoch=True, log_models=False, log_datasets=False, checkpoint=False)

# if production, log to shared experiment space, otherwise just log at notebook level by default
if not DEBUG:
    mlflow.set_experiment(EXPERIMENT_LOCATION)

# Starts an MLflow experiment to track training parameters and results.
with mlflow.start_run() as run:
    # Get the unique ID of the current run in case we aren't registering it
    run_id = mlflow.active_run().info.run_id
    # Set the tag based on the version num
    mlflow.set_tag(key="version", value=sm.name)

    # Create the keras model
    keras_model = sm.create_model(train_gen=train_gen, layer_params=layer_params)

    # Fit the model
    history = keras_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2 if DEBUG else 200,
        batch_size=train_gen.batch_size,
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)],
    )

    # wrap in custom class that defines pre and post processing steps to be applied when called at inference time
    pyfunc_model = SurrogateModelingWrapper(
        trained_model=keras_model,
        building_features=train_gen.building_features,
        weather_features=train_gen.weather_features,
        targets=train_gen.targets,
    )

    mlflow.pyfunc.log_model(
        python_model=pyfunc_model,
        artifact_path=sm.artifact_path,
        code_paths=["../src/surrogate_model.py"],
        # signature=signature
    )
    # skip registering model for now..
    # mlflow.register_model(f"runs:/{run_id}/{sm.artifact_path}", str(sm))

if not DEBUG:
    # serialize the keras model and save to GCP
    sm.save_keras_model(run_id=run_id)

# COMMAND ----------

# MAGIC %md ## Evaluate Model

# COMMAND ----------

# MAGIC %md #### Test Mode

# COMMAND ----------

# DBTITLE 1,Inspect predictions for one batch
# print out model predictions just to make sure everything worked
results = keras_model.predict(val_gen[0][0])
print(np.hstack([results[c] for c in train_gen.targets]))

# COMMAND ----------

# DBTITLE 1,Inspect predictions using logged model
# evaluate the unregistered model we just logged and make sure everything runs
print(run_id)
# mlflow.pyfunc.get_model_dependencies(model_uri=sm.get_model_uri(run_id=run_id))
# Load the model using its registered name and version/stage from the MLflow model registry
model_loaded = mlflow.pyfunc.load_model(model_uri=sm.get_model_uri(run_id=run_id))
test_gen = DataGenerator(train_data=test_data.limit(10))
# load input data table as a Spark DataFrame
input_data = test_gen.training_set.load_df().toPandas()
# run prediction and output a N x M matrix of predictions where N is the number of rows in the input data table and M is the number of target columns
print(model_loaded.predict(input_data))

# COMMAND ----------

# DBTITLE 1,Pass Run ID to next notebook if running in job
if not DEBUG:
    dbutils.jobs.taskValues.set(key="run_id", value=run_id)
