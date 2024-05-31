# Databricks notebook source
# MAGIC %md # Model Training
# MAGIC
# MAGIC ### Goal
# MAGIC Train deep learning model to predict energy a building's HVAC energy consumption
# MAGIC
# MAGIC ### Process
# MAGIC * Load in train/val/test sets containing targets and feature keys
# MAGIC * Initialize data generators on train/val sets which pulls in weather and building model features
# MAGIC * Train model
# MAGIC * Evaluate model and write out metrics
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs: 
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Weather data indexed by (weather_file_city) with a 8670-length timeseries vector
# MAGIC - `ml.surrogate_model.building_upgrade_simulation_outputs_annual`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC
# MAGIC ##### Outputs: 
# MAGIC - `gs://the-cube/export/surrogate_model_metrics/cnn/{model_name}_v{model_version_num}.csv'`: Aggregated evaluation metrics
# MAGIC
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

if dbutils.widgets.get('Mode') == 'Test':
    DEBUG = True
else:
    DEBUG = False
print(DEBUG)

# COMMAND ----------

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# COMMAND ----------

# DBTITLE 1,Import
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from typing import Tuple, Dict
# MAGIC
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import pyspark.sql.functions as F
# MAGIC import tensorflow as tf
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.types import FloatType
# MAGIC from tensorflow import keras
# MAGIC
# MAGIC from datagen import DataGenerator
# MAGIC from surrogate_model import SurrogateModel
# MAGIC
# MAGIC # list available GPUs
# MAGIC tf.config.list_physical_devices("GPU")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

# DBTITLE 1,Data loading function
def load_data(
    consumption_group_dict= DataGenerator.consumption_group_dict,
    building_feature_table_name= DataGenerator.building_feature_table_name,
    n_subset=None,
    p_val=0.15,
    p_test=0.15,
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
        p_val (float): Proportion of data to use for validation. Default is 0.15.
        p_test (float): Proportion of data to use for testing. Default is 0.15.
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
        WHERE O.upgrade_id = 0
            AND sqft < 8000
            AND occupants <= 10
        """
    )

    # Subset the data if n_subset is specified
    if n_subset is not None:
        n_total = inference_data.count()
        if n_subset > n_total:
            print("'n_subset' is more than the total number of records, returning all records...")
        else:
            inference_data = inference_data.sample(
                fraction=n_subset / n_total, seed=seed
            )

    p_train = 1 - p_val - p_test

    # Split the data into train, validation, and test sets
    return inference_data.randomSplit(weights=[p_train, p_val, p_test], seed=seed)

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
        - train_gen (DataGenerator): The training data generator
        """
        self.model = trained_model
        # self.building_features = train_gen.building_features
        # self.weather_features = train_gen.weather_features
        # self.targets = train_gen.targets
        self.building_features = building_features
        self.weather_features = weather_features
        self.targets = targets

    def preprocess_input(self, model_input:pd.DataFrame)->Dict[str,np.ndarray]:
        """
        Preprocesses the model input of P features over N samples 

        Parameters:
        - model_input (pd.Dataframe): The input features for the model of shape [N, P].

        Returns:
        - The preprocessed feature data in format {feature_name(str) : np.array [N,]}
        """
        return self.convert_feature_dataframe_to_dict(model_input)

    def postprocess_result(self, results:Dict[str,np.ndarray]) -> np.ndarray:
        """
        Postprocesses the model results for N samples over M targets. 

        Parameters:
        - results (dict of {str: np.ndarray}): The outputs of the model in format {target_name (str) : np.ndarray [N,]}

        Returns:
        - The model predictions floored at 0: np.ndarray of shape [N, M]
                                   
        """
        return np.clip(np.hstack([results[c] for c in self.targets]), a_min=0, a_max=None)

    def predict(self, context, model_input:pd.DataFrame) -> np.ndarray:
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

    def convert_feature_dataframe_to_dict(self, feature_df:pd.DataFrame)->Dict[str,np.ndarray]:
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
            col: np.array(np.vstack(feature_df[col].values)) for col in self.weather_features
        }
        return {**X_train_bm, **X_train_weather}

# COMMAND ----------

# DBTITLE 1,Initialize model
model = SurrogateModel(name='test' if DEBUG else 'sf_detatched_hvac_baseline')

# COMMAND ----------

# DBTITLE 1,Fit model
# Train keras model and log the model with the Feature Engineering in UC. 

# Set the activation function and numeric data type for the model's layers
layer_params = {
    "activation": "leaky_relu", 
    "dtype": np.float32 
}

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
        epochs = 2 if DEBUG else 100,  
        batch_size = train_gen.batch_size,
        verbose=2, 
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8)]
    )

    # wrap in custom class that defines pre and post processing steps to be applied when called at inference time
    pyfunc_model = SurrogateModelingWrapper(keras_model, train_gen.building_features, train_gen.weather_features, train_gen.targets) 

    # If in test mode, don't register the model, just pull it based on run_id in evaluation testing
    model.fe.log_model(
        model=pyfunc_model,
        artifact_path=model.artifact_path,
        flavor=mlflow.pyfunc,  # since using custom pyfunc wrapper 
        training_set=train_gen.training_set,
        registered_model_name= None if DEBUG else str(model)  # registered the model name if in DEBUG mode
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
    np.hstack([results[c] for c in train_gen.targets])

# COMMAND ----------

# DBTITLE 1,Inspect predictions using logged model
if DEBUG: # evaluate the unregistered model we just logged and make sure everything runs
    print(run_id)
    pred_df = model.score_batch(test_data = test_data, run_id = run_id, targets = train_gen.targets)
    pred_df.display()

# COMMAND ----------

# MAGIC %md #### Production Mode

# COMMAND ----------

# DBTITLE 1,Evaluation functions
@udf(returnType=FloatType())
def APE(pred:float, true:float) -> float:
    if true == 0:
        return None
    return abs(pred - true)/true

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

# DBTITLE 1,Run inference on test set
if not DEBUG:
    # score using  latest registered model
    mlflow.pyfunc.get_model_dependencies(model.get_model_uri())
    pred_df = model.score_batch(test_data = test_data, targets = train_gen.targets) 

# COMMAND ----------

# DBTITLE 1,Create aggregated prediction metric table
if not DEBUG:
    pred_df_long = (
        pred_df
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
            .withColumn('pred',
                    F.when(F.col('end_use') == 'heating', F.col('prediction')[0])
                    .when(F.col('end_use') == 'cooling',F.col('prediction')[1])
                    .otherwise(F.col('prediction')[1] + F.col('prediction')[0])
            )
            .withColumn('absolute_error', F.abs(F.col('pred') -  F.col('true')))
            .withColumn('absolute_percentage_error', APE(F.col('pred'), F.col('true')))
    )

    metrics_by_enduse_type = evalute_metrics(
        df = pred_df_long.where(F.col('end_use') != 'hvac'), 
        groupby_cols = ['end_use' ,'type']
    )

    metrics_by_enduse = evalute_metrics(
        df = pred_df_long, 
        groupby_cols = ['end_use']
    ).withColumn('type', F.lit('Total'))

    df_metrics_combined = metrics_by_enduse_type.unionByName(metrics_by_enduse).toPandas()

    df_metrics_combined.display()

    df_metrics_combined.to_csv(f'gs://the-cube/export/surrogate_model_metrics/cnn/{str(model)}_v{model.get_latest_model_version()}.csv', index=False)
