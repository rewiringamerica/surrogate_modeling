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
# MAGIC - Figure out issues with training on GPU: When using the existing default tf version on a GPU cluster, the code errors out at eval time with various inscritible errors. By downgrading to required tensorflow version in `requirements.txt`, it then shows no GPUs avaialble, and during training it seems to show 0% GPU utilization, which makes me assume that it is not actually using the GPU. However, it seems to train faster on a GPU cluster than on a CPU cluster with even more memory. Further, when the downgraded tf version is installed at the cluster level, it also doesn't work. 
# MAGIC - Troubleshoot `env_manager` issues with loading env that model was trained in
# MAGIC
# MAGIC #### Future Work
# MAGIC - Once upgrades to the building metadata table, remove subset to upgrade_id = 0
# MAGIC - Support more granular temporal output reslution using dynamic aggregation of hourly outputs table
# MAGIC - Maybe figure out how to define the `load_context()` method of the `SurrogateModelingWrapper` class in such a way that we can define it in a different file (currently spark pickling issues prevent this)
# MAGIC
# MAGIC ---
# MAGIC #### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 14.3 ML (or >= Databricks Runtime 14.3 +  `%pip install databricks-feature-engineering`)
# MAGIC - Node type: Single Node. Because of [this issue](https://kb.databricks.com/en_US/libraries/apache-spark-jobs-fail-with-environment-directory-not-found-error), worker nodes cannot access the directory needed to run inference on a keras trained model, meaning that the `score_batch()` function throws and OSError. 
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)

# COMMAND ----------

# install required packages: note that tensorflow must be installed at the notebook-level
%pip install gcsfs==2023.5.0 tensorflow==2.15.0.post1

# COMMAND ----------

# this controls the training parameters, with test mode on a much smaller training set for fewer epochs
dbutils.widgets.dropdown("Mode", "Test", ["Test", "Production"])

if dbutils.widgets.get("Mode") == "Test":
    DEBUG = True
else:
    DEBUG = False
print(DEBUG)

# COMMAND ----------

# DBTITLE 1,Import
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import os
# MAGIC
# MAGIC os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# MAGIC
# MAGIC from typing import Tuple, Dict
# MAGIC
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import pyspark.sql.functions as F
# MAGIC import tensorflow as tf
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.types import DoubleType
# MAGIC from tensorflow import keras
# MAGIC
# MAGIC from datagen import DataGenerator
# MAGIC from model import Model
# MAGIC
# MAGIC # list available GPUs
# MAGIC tf.config.list_physical_devices("GPU")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

# DBTITLE 1,Data loading function
def load_data(
    consumption_group_dict=DataGenerator.consumption_group_dict,
    building_feature_table_name=DataGenerator.building_feature_table_name,
    n_subset=None,
    p_val=0.2,
    p_test=0.1,
    seed=42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load the data for model training prediction containing the targets and the keys needed to join to feature tables

    Parameters:
        consumption_group_dict (dict): Dictionary mapping consumption categories (e.g., 'heating') to columns.
            Default is DataGenerator.consumption_group_dict.
        building_feature_table_name (str): Name of the building feature table.
            Default is DataGenerator.building_feature_table_name
        n_subset (int): Number of subset records to select. Default is None (select all records).
        p_val (float): Proportion of data to use for validation. Default is 0.2.
        p_test (float): Proportion of data to use for testing. Default is 0.1.
        seed (int): Seed for random sampling. Default is 42.

    Returns:
        train data (DataFrame)
        val_data (DataFrame)
        test_data (DataFrame)
    """
    # Read outputs table and sum over consumption columns within each consumption group
    # join to the bm table to get required keys to join on and filter the building models based on charactaristics
    sum_str = ", ".join(
        [f"{'+'.join(v)} AS {k}" for k, v in consumption_group_dict.items()]
    )
    inference_data = spark.sql(
        f"""
        SELECT B.building_id, B.upgrade_id, B.weather_file_city, {sum_str}
        FROM ml.surrogate_model.building_upgrade_simulation_outputs_annual O
        LEFT JOIN {building_feature_table_name} B 
            ON B.upgrade_id = O.upgrade_id AND B.building_id == O.building_id
        WHERE sqft < 8000
        """
    )

    # get list of unique building ids, which will be the basis for the dataset split
    unique_building_ids = inference_data.where(F.col("upgrade_id") == 0).select(
        "building_id"
    )

    # Subset the data if n_subset is specified
    if n_subset is not None:
        n_total = unique_building_ids.count()
        if n_subset > n_total:
            print(
                "'n_subset' is more than the total number of records, returning all records..."
            )
        else:
            unique_building_ids = unique_building_ids.sample(
                fraction=1.0, seed=seed
            ).limit(n_subset)

    # Split the building_ids into train, validation, and test sets (may not exactly match passed proportions)
    p_train = 1 - p_val - p_test
    train_ids, val_ids, test_ids = unique_building_ids.randomSplit(
        weights=[p_train, p_val, p_test], seed=seed
    )

    # select train, val and test set based on building ids
    train_df = train_ids.join(inference_data, on="building_id")
    val_df = val_ids.join(inference_data, on="building_id")
    test_df = test_ids.join(inference_data, on="building_id")

    return train_df, val_df, test_df

# COMMAND ----------

# DBTITLE 1,Load data
train_data, val_data, test_data = load_data(n_subset=100 if DEBUG else None)

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
        processed_df = self.preprocess_input(model_input.copy())
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
model = Model(name="test" if DEBUG else "sf_hvac")

# COMMAND ----------

# DBTITLE 1,Fit model
# Train keras model and log the model with the Feature Engineering in UC.

# Set the activation function and numeric data type for the model's layers
layer_params = {"activation": "leaky_relu", "dtype": np.float32}

# Disable MLflow autologging and instead log the model using Feature Engineering in UC using `fe.log_model
mlflow.tensorflow.autolog(log_models=False)
mlflow.sklearn.autolog(log_models=False)

# Starts an MLflow experiment to track training parameters and results.
with mlflow.start_run() as run:

    # Get the unique ID of the current run in case we aren't registering it
    run_id = mlflow.active_run().info.run_id

    # Create the keras model
    keras_model = model.create_model(train_gen=train_gen, layer_params=layer_params)

    # Fit the model
    history = keras_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2 if DEBUG else 100,
        batch_size=train_gen.batch_size,
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )

    # wrap in custom class that defines pre and post processing steps to be applied when called at inference time
    pyfunc_model = SurrogateModelingWrapper(
        trained_model=keras_model,
        building_features=train_gen.building_features,
        weather_features=train_gen.weather_features,
        targets=train_gen.targets,
    )

    # If in test mode, don't register the model, just pull it based on run_id in evaluation testing
    model.fe.log_model(
        model=pyfunc_model,
        artifact_path=model.artifact_path,
        flavor=mlflow.pyfunc,  # since using custom pyfunc wrapper
        training_set=train_gen.training_set,
        registered_model_name= None if DEBUG else str(model),  # registered the model name if in DEBUG mode
    )

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
    pred_df = model.score_batch(test_data=test_data, run_id=run_id)
    pred_df.display()

# COMMAND ----------

# MAGIC %md #### Production Mode

# COMMAND ----------

# DBTITLE 1,Run inference on test set
if not DEBUG:
    # for right now have to limit the test set since driver seems to be running out of mem
    target_test_size = 75000
    target_n_building_frac = target_test_size / test_data.count()
    test_building_id_subset = (
        test_data.select("building_id").distinct().sample(target_n_building_frac)
    )
    test_data_sub = test_data.join(test_building_id_subset, on="building_id")
    print(test_data_sub.count())
    # score using  latest registered model
    mlflow.pyfunc.get_model_dependencies(model.get_model_uri())
    pred_df = model.score_batch(test_data=test_data_sub)

# COMMAND ----------

# DBTITLE 1,Write out predictions for evaluation
# save the predictions to a delta table-- aggregating in eval without first writing to delta seems to often kill the driver
if not DEBUG:
    (
        pred_df.select(
            "building_id",
            "upgrade_id",
            "prediction",
            *train_gen.targets,
        ).write.saveAsTable(
            f"{str(model)}_predictions",
            format="delta",
            mode="overwrite",
            overwriteSchema=True,
            userMetadata=model.get_latest_model_version(),
        )
    )
