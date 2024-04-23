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
# MAGIC - Figure out issues with training on GPU
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
# MAGIC - Libraries: `gcsfs==2023.5.0`. Ignore all requirements in `requriements.txt`. 

# COMMAND ----------

# DBTITLE 1,Widget Mode Debug Tool
dbutils.widgets.dropdown("Mode", "Test", ["Test", "Production"])

if dbutils.widgets.get('Mode') == 'Test':
    DEBUG = True
else:
    DEBUG = False
print(DEBUG)

# COMMAND ----------

# DBTITLE 1,Import
# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
%load_ext autoreload
%autoreload 2

from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
from tensorflow import keras

from datagen_db import DataGenerator
from model_db import Model

# check that GPU is available
# tf.config.list_physical_devices("GPU")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

def load_inference_data(
    consumption_group_dict= DataGenerator.consumption_group_dict,
    building_feature_table_name= DataGenerator.building_feature_table_name,
    n_subset=None,
    p_val=0.15,
    p_test=0.15,
    seed=42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:

    # Read in the "raw" data which contains the prediction target and the keys needed to join to the feature tables.
    # Right now this is kind of hacky since we need to join to the bm table to do the required train data filtering
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

    if n_subset is not None:
        n_total = inference_data.count()
        if n_subset > n_total:
            print(
                "'n_subset' is more than the total number of records, returning all records..."
            )
        else:
            inference_data = inference_data.sample(
                fraction=n_subset / n_total, seed=seed
            )

    p_train = 1 - p_val - p_test
    return inference_data.randomSplit(weights=[p_train, p_val, p_test], seed=seed)

# COMMAND ----------

# DBTITLE 1,Load data
# consumption_group_dict={'cooling': ['electricity__cooling_fans_pumps', 'electricity__cooling']} 
train_data, val_data, test_data = load_inference_data(n_subset=100 if DEBUG else None)

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
# this allows us to apply pre/post processing to the inference data
class SurrogateModelingWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model, train_gen):
        self.model = trained_model
        self.building_features = train_gen.building_features
        self.weather_features = train_gen.weather_features
        self.targets = train_gen.targets

    def preprocess_input(self, model_input):
        return self.convert_training_data_to_dict(model_input)

    def postprocess_result(self, results):
        return np.hstack([results[c] for c in self.targets])

    def predict(self, context, model_input):
        processed_df = self.preprocess_input(model_input.copy())
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df)
    
    def convert_training_data_to_dict(self, building_feature_df):
        X_train_bm = {col: np.array(building_feature_df[col]) for col in self.building_features}
        X_train_weather = {
            col: np.array(np.vstack(building_feature_df[col].values)) for col in self.weather_features
        }
        return {**X_train_bm, **X_train_weather}

# COMMAND ----------

# DBTITLE 1,Initialize model
model = Model(name='test' if DEBUG else 'sf_detatched_hvac_baseline')

# COMMAND ----------

# DBTITLE 1,Fit model
# Train keras model and logs the model with the Feature Engineering in UC. 

# The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (`mlflow.sklearn.autolog(log_models=False)`); this is because the model is logged using `fe.log_model`.

layer_params = {
    "activation": "leaky_relu",
    "dtype": np.float32
}

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.tensorflow.autolog(log_models=False)
mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run() as run:
    run_id = mlflow.active_run().info.run_id

    keras_model = model.create_model(train_gen=train_gen, layer_params=layer_params)

    history = keras_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs = 2,
        batch_size = train_gen.batch_size, 
        verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)]
    )

    pyfunc_model = SurrogateModelingWrapper(keras_model, train_gen)

    # If in test mode, don't register the model, just pull it based on run_id in evaluation testing
    model.fe.log_model(
        model=pyfunc_model,
        artifact_path=model.artifact_path,
        flavor=mlflow.pyfunc,
        training_set=train_gen.training_set,
        registered_model_name= None if DEBUG else str(model),
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
            .withColumn('pred', #floor predictions at 0
                    F.when(F.col('end_use') == 'heating', F.greatest(F.col('prediction')[0], F.lit(0)))
                    .when(F.col('end_use') == 'cooling',F.greatest(F.col('prediction')[1], F.lit(0)))
                    .otherwise(F.greatest(F.col('prediction')[1] + F.col('prediction')[0], F.lit(0)))
            )
            .withColumn('absolute_error', F.abs(F.col('pred') -  F.col('true')))
            .withColumn('absolute_percentage_error', APE(F.col('pred'), F.col('true')))
    )

    metrics_by_enduse_type = evalute_metrics(
        df = pred_df_long.where(F.col('end_use') != 'hvac'), 
        groupby_cols = ['end_use' ,'type']
    )

    metrics_by_enduse = evalute_metrics(
        df = pred_df_long.where(F.col('end_use') != 'hvac'), 
        groupby_cols = ['end_use']
    ).withColumn('type', F.lit('Total'))

    df_metrics_combined = metrics_by_enduse_type.unionByName(metrics_by_enduse).toPandas()

    df_metrics_combined.display()

    df_metrics_combined.to_csv(f'gs://the-cube/export/surrogate_model_metrics/cnn/{str(model)}_v{model.get_latest_model_version()}.csv', index=False)
