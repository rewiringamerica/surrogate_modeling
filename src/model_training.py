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

import itertools
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow

import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras.backend as K
from tensorflow.keras import layers, models

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from tensorflow.keras.losses import MeanAbsoluteError

# COMMAND ----------

# features we are and targets currently using for testing -- probably will store in a class at some point
building_metadata_features = ["sqft", "occupants"]
targets = ["site_energy__total"]

# not including weekend/weekday for now
weather_features = [
    "temp_air",
    # "relative_humidity",
    "wind_speed",
    # "wind_direction",
    "ghi",
    # "dni",
    # "diffuse_horizontal_illum",
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
        lookup_key=["building_id", "upgrade_id"],
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
raw_data = spark.sql(f"SELECT building_id, upgrade_id, weather_file_city, {','.join(targets)} FROM ml.surrogate_model.annual_outputs WHERE upgrade_id == 0")
train_data, test_data = raw_data.randomSplit(weights=[0.8,0.2], seed=100)
#       .where(F.col('geometry_building_type_acs') == 'Single-Family Detached')
#       .where(F.col('vacancy_status') == 'Occupied')
#       .where(F.col('sqft') < 8000)
#       .where(F.col('occupants') < 11)

# COMMAND ----------

def convert_training_data_to_dict(train_pd):
    X_train_bm = {col: np.array(train_pd[col]) for col in building_metadata_features}
    X_train_weather = {col: np.array(np.vstack(train_pd[col].values)) for col in weather_features}
    return {**X_train_bm, **X_train_weather}


# # takes ~1m for 10K samples on single node cluster
def load_data(raw_data, feature_lookups, targets, n_subset=100, val_split = .2, seed = 42):
    # exclude column used to just join to weather data
    exclude_columns = ["weather_file_city", "building_id", "upgrade_id"]
    #exclude_columns = ["weather_file_city"]

    # Create the training set that includes the raw input data merged with corresponding features from both feature tables
    training_set = fe.create_training_set(
        df=train_data.limit(n_subset),
        feature_lookups=feature_lookups,
        label=targets,
        exclude_columns=exclude_columns,
    )

    # Load the TrainingSet into a dataframe which can be passed into keras for training a model
    train_df, val_df = training_set.load_df().randomSplit(weights=[1-val_split,val_split], seed=seed)
    #training_dict = train_df.select(building_metadata_features).toPandas().to_dict(orient='list')

    # Convert DataFrame columns to NumPy arrays and create the dictionary
    train_pd = train_df.toPandas()
    X_train = convert_training_data_to_dict(train_pd)
    y_train = {col: np.array(train_pd[col]) for col in targets}

    val_pd = val_df.toPandas()
    X_val = convert_training_data_to_dict(val_pd)
    y_val = {col: np.array(val_pd[col]) for col in targets}

    return X_train, X_val, y_train, y_val, training_set

X_train, X_val, y_train, y_val, training_set = load_data(
    raw_data=raw_data,
    feature_lookups=building_metadata_feature_lookups + weather_feature_lookups,
    targets=targets,
    n_subset=100,
)

# COMMAND ----------

# Configure MLflow client to access models in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = "ml.surrogate_model.surrogate_model"

client = mlflow.tracking.client.MlflowClient()

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

    # def load_context(self, context):

    def preprocess_input(self, model_input):
        model_input_dict = convert_training_data_to_dict(model_input)
        return model_input_dict

    def postprocess_result(self, results):
        return np.concatenate([results[c] for c in targets])

    def predict(self, context, model_input):
        processed_df = self.preprocess_input(model_input.copy())
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df)

# COMMAND ----------

def create_model(layer_params=None):
    # Building model
    bmo_inputs_dict = {
        building_feature: layers.Input(
            name=building_feature, shape=(1,),
            #dtype=train_gen.feature_dtype(building_feature)
        )
        for building_feature in building_metadata_features
    }

    bmo_inputs = []
    for feature, layer in bmo_inputs_dict.items():
        # handle categorical, ordinal, etc. features.
        # Here it is detected by dtype; perhaps explicit feature list and
        # handlers would be better
        # if train_gen.feature_dtype(feature) == tf.string:
        #     encoder = layers.StringLookup(
        #         name=feature+'_encoder', output_mode='one_hot',
        #         dtype=layer_params['dtype']
        #     )
        #     encoder.adapt(train_gen.feature_vocab(feature))
        #     layer = encoder(layer)
        bmo_inputs.append(layer)

    bm = layers.Concatenate(name='concat_layer', dtype=layer_params['dtype'])(bmo_inputs)
    bm = layers.Dense(32, name='second_dense', **layer_params)(bm)
    bm = layers.Dense(8, name='third_dense', **layer_params)(bm)

    bmo = models.Model(inputs=bmo_inputs_dict, outputs=bm, name='building_features_model')

    # Weather data model
    weather_inputs_dict = {
        weather_feature: layers.Input(
            name=weather_feature, shape=(None, 1,), dtype=layer_params['dtype'])
        for weather_feature in weather_features
    }
    weather_inputs = list(weather_inputs_dict.values())

    wm = layers.Concatenate(
        axis=-1, name='weather_concat_layer', dtype=layer_params['dtype']
    )(weather_inputs)
    wm = layers.Conv1D(
        filters=16,
        kernel_size=8,
        padding='same',
        data_format='channels_last',
        name='first_1dconv',
        **layer_params
    )(wm)
    wm = layers.Conv1D(
        filters=8,
        kernel_size=8,
        padding='same',
        data_format='channels_last',
        name='last_1dconv',
        **layer_params
    )(wm)

    # sum the time dimension
    wm = layers.Lambda(
        lambda x: tf.keras.backend.sum(x, axis=1), dtype=layer_params['dtype'])(wm)

    wmo = models.Model(
        inputs=weather_inputs_dict, outputs=wm, name='weather_features_model')

    # Combined model and separate towers for output groups
    cm = layers.Concatenate(name='combine_features')([bmo.output, wmo.output])
    cm = layers.Dense(16, **layer_params)(cm)
    cm = layers.Dense(16, **layer_params)(cm)
    # cm is a chokepoint representing embedding of a building + climate it is in

    # building a separate tower for each output group
    final_outputs = {}
    for consumption_group in targets:
        io = layers.Dense(8, name=consumption_group+'_entry', **layer_params)(cm)
        # ... feel free to add more layers
        io = layers.Dense(8, name=consumption_group+'_mid', **layer_params)(io)
        # no activation on the output
        io = layers.Dense(1, name=consumption_group, **layer_params)(io)
        final_outputs[consumption_group] = io

    final_model = models.Model(
        inputs={**bmo.input, **wmo.input}, outputs=final_outputs)

    final_model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer='adam'
    )
    return final_model

# COMMAND ----------

# Disable MLflow autologging and instead log the model using Feature Engineering in UC
mlflow.tensorflow.autolog(log_models=False)
mlflow.sklearn.autolog(log_models=False)
# def train_model():

with mlflow.start_run() as run:

    training_params = {
        "epochs": 40,
        "batch_size": 256}
    
    layer_params = {
        'activation': 'leaky_relu',
        'dtype': np.float32,
    }

    model = create_model(layer_params)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        verbose=2,
        **training_params)

    pyfunc_model = SurrogateModelingWrapper(model)


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
    mlflow_client = mlflow.tracking.client.MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

## For simplicity, this example uses inference_data_df as input data for prediction
latest_model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_model_version}"
batch_pred = fe.score_batch(model_uri=model_uri, df=test_data.limit(10), result_type=ArrayType(DoubleType()))
batch_pred.display()

# COMMAND ----------


