# Databricks notebook source
# MAGIC %md # Model Training
# MAGIC
# MAGIC ### Goal
# MAGIC Train deep learning model to predict a building's HVAC energy consumption
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
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Weather data indexed by (weather_file_city) with a 8670-length timeseries vector
# MAGIC - `ml.surrogate_model.building_upgrade_simulation_outputs_annual`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC
# MAGIC ##### Outputs: 
# MAGIC None. The model is logged to the unity catalog with the run id, but as of now is not registered due to issue with signature enforcement slowing down inference. 
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Outstanding
# MAGIC * Figure out how to register the model with a signature without slowing down inference
# MAGIC * Handle retracing issues
# MAGIC * Install model dependencies using the logged requirements.txt file
# MAGIC
# MAGIC #### Future Work
# MAGIC
# MAGIC ---
# MAGIC #### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 14.3 ML (or >= Databricks Runtime 14.3 +  `%pip install databricks-feature-engineering`)
# MAGIC - Node type: Single Node. Because of [this issue](https://kb.databricks.com/en_US/libraries/apache-spark-jobs-fail-with-environment-directory-not-found-error), worker nodes cannot access the directory needed to run inference on a keras trained model, meaning that the `score_batch()` function throws and OSError. 
# MAGIC - Can be run on CPU or GPU, with 2x speedup on GPU
# MAGIC - Cluster-level packages: `gcsfs==2023.5.0`, `mlflow==2.13.0` (newer than default, which is required to pass a `code_paths` in logging)
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Set debug mode
# this controls the training parameters, with test mode on a much smaller training set for fewer epochs
dbutils.widgets.dropdown("mode", "test", ["test", "production"])

if dbutils.widgets.get("mode") == "test":
    DEBUG = True
else:
    DEBUG = False
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
# MAGIC import pyspark.sql.functions as F
# MAGIC import tensorflow as tf
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.types import DoubleType
# MAGIC from tensorflow import keras
# MAGIC from typing import Tuple, Dict
# MAGIC
# MAGIC from datagen import DataGenerator, load_data
# MAGIC from surrogate_model import SurrogateModel
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
train_data, val_data, test_data = load_data(n_train=1000 if DEBUG else None)

# COMMAND ----------

# DBTITLE 1,Initialize train/val data generators
train_gen = DataGenerator(train_data=train_data, batch_size=256)
val_gen = DataGenerator(train_data=val_data, batch_size=256)

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

    def load_context(self, context):
        pass

    def preprocess_input(self, model_input: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocesses the model input of P features over N samples

        Parameters:
        - model_input (pd.Dataframe): The input features for the model of shape [N, P].

        Returns:
        - The preprocessed feature data in format {feature_name(str) : np.array [N,]}
        """
        return self.convert_feature_dataframe_to_dict(model_input)

    def postprocess_result(self, results: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Postprocesses the model results for N samples over M targets.

        Parameters:
        - results (dict of {str: np.ndarray}): The outputs of the model in format {target_name (str) : np.ndarray [N,]}

        Returns:
        - The model predictions floored at 0: np.ndarray of shape [N, M]

        """
        return np.clip(
            np.hstack([results[c] for c in self.targets]), a_min=0, a_max=None
        )

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the model for N samples over M targets.

        Parameters:
        - context (Any): Ignored here. It's a placeholder for additional data or utility methods.
        - model_input (pd.Dataframe): The input features for the model of shape [N, P]

        Returns:
        - The model predictions floored at 0: np.ndarray of shape [N, M]
        """
        processed_df = self.preprocess_input(model_input)
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df)

    def convert_feature_dataframe_to_dict(
        self, feature_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Converts the feature data from a pandas dataframe to a dictionary.

        Parameters:
        - feature_df: : The input features for the model of shape [N, P] where feature columns
                        for weather features contain len 8760 arrays.

        Returns:
        - The preprocessed feature data in format {feature_name (str) :
                np.array of shape [N] for building model features and shape [N,8760] for weather features}
        """
        X_train_bm = {col: np.array(feature_df[col]) for col in self.building_features}
        X_train_weather = {
            col: np.array(np.vstack(feature_df[col].values))
            for col in self.weather_features
        }
        return {**X_train_bm, **X_train_weather}

# COMMAND ----------

# DBTITLE 1,Initialize model
sm = SurrogateModel(name="test" if DEBUG else "sf_hvac_by_fuel")

# COMMAND ----------

# DBTITLE 1,Fit model
# Train keras model and log the model with the Feature Engineering in UC. Note that right we are skipping registering the model in the UC-- this requires storing the signature, which for unclear reasons, is slowing down inference more than 10x.

# Init FeatureEngineering client
fe = FeatureEngineeringClient()

# Set the activation function and numeric data type for the model's layers
layer_params = {
    "activation": "leaky_relu",
    "dtype": np.float32,
    "kernel_initializer": "he_normal",
}

# signature_df = train_gen.training_set.load_df().select(train_gen.building_features + train_gen.targets + train_gen.weather_features).limit(1).toPandas()
# signature=mlflow.models.infer_signature(model_input = signature_df[train_gen.building_features + train_gen.weather_features], model_output = signature_df[train_gen.targets])

# turn on tf logging but without model checkpointing since this slows down training 2x
mlflow.tensorflow.autolog(
    log_every_epoch=True, log_models=False, log_datasets=False, checkpoint=False
)

# if production, log to shared experiment space, otherwise just log at notebook level by default
if not DEBUG:
    mlflow.set_experiment(EXPERIMENT_LOCATION)

# Starts an MLflow experiment to track training parameters and results.
with mlflow.start_run() as run:

    # Get the unique ID of the current run in case we aren't registering it
    run_id = mlflow.active_run().info.run_id

    # Create the keras model
    keras_model = sm.create_model(train_gen=train_gen, layer_params=layer_params)

    # Fit the model
    history = keras_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2 if DEBUG else 100,
        batch_size=train_gen.batch_size,
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
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
        code_paths=["surrogate_model.py"],
        #signature=signature
    )
    #mlflow.register_model(f"runs:/{run_id}/{sm.artifact_path}", str(sm))

# COMMAND ----------

# MAGIC %md ## Evaluate Model

# COMMAND ----------

# MAGIC %md #### Test Mode

# COMMAND ----------

# DBTITLE 1,Inspect predictions for one batch
# print out model predictions just to make sure everything worked
if DEBUG:
    results = keras_model.predict(val_gen[0][0])
    print(np.hstack([results[c] for c in train_gen.targets]))

# COMMAND ----------

# DBTITLE 1,Inspect predictions using logged model
# evaluate the unregistered model we just logged and make sure everything runs
if DEBUG:
    print(run_id)
    #mlflow.pyfunc.get_model_dependencies(model_uri=sm.get_model_uri(run_id=run_id))
    # Load the model using its registered name and version/stage from the MLflow model registry
    model_loaded = mlflow.pyfunc.load_model(model_uri=sm.get_model_uri(run_id=run_id))
    test_gen = DataGenerator(train_data=test_data)
    # load input data table as a Spark DataFrame
    input_data = test_gen.training_set.load_df().toPandas()
    #run prediction and output a N x M matrix of predictions where N is the number of rows in the input data table and M is the number of target columns
    print(model_loaded.predict(input_data))
