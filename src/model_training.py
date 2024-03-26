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
import math
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# COMMAND ----------

#target grouping
consumption_group_dict = {
    'heating' : [
        'electricity__heating_fans_pumps',
        'electricity__heating_hp_bkup',
        'electricity__heating',
        'fuel_oil__heating_hp_bkup',
        'fuel_oil__heating',
        'natural_gas__heating_hp_bkup',
        'natural_gas__heating',
        'propane__heating_hp_bkup',
        'propane__heating'],
    'cooling' : [
        'electricity__cooling_fans_pumps',
        'electricity__cooling']
}

targets = list(consumption_group_dict.keys())

# COMMAND ----------

# features we are and targets currently using for testing -- probably will store in a class at some point
building_metadata_features = ["sqft", "occupants"]

weather_features = [
    "temp_air",
    # "relative_humidity",
    "wind_speed",
    # "wind_direction",
    "ghi",
    # "dni",
    # "diffuse_horizontal_illum",
    "weekend"
]

# COMMAND ----------

# #3.5e-07 of mem per building for 2 bm features, 1 target (annual), and 3 weather features (8670)
# n_buildings = 550000
# n_upgrades = 10
# mem_per_building_gb = 3.5e-7
# mem_per_building_gb * n_buildings * n_upgrades
# #df_pd.memory_usage(index=True, deep = True).sum() / 1E9

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

sum_str = ', '.join([f"{'+'.join(v)} AS {k}" for k, v in consumption_group_dict.items()])

raw_data = spark.sql(f"SELECT building_id, upgrade_id, weather_file_city, {sum_str} FROM ml.surrogate_model.annual_outputs WHERE upgrade_id == 0")

train_data, test_data = raw_data.randomSplit(weights=[0.8,0.2], seed=42)
#       .where(F.col('geometry_building_type_acs') == 'Single-Family Detached')
#       .where(F.col('vacancy_status') == 'Occupied')
#       .where(F.col('sqft') < 8000)
#       .where(F.col('occupants') < 11)

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

def convert_training_data_to_dict(train_pd, building_metadata_features, weather_features):
    X_train_bm = {col: np.array(train_pd[col]) for col in building_metadata_features}
    X_train_weather = {col: np.array(np.vstack(train_pd[col].values)) for col in weather_features}
    return {**X_train_bm, **X_train_weather}

#this allows us to apply pre/post processing to the inference data
class SurrogateModelingWrapper(mlflow.pyfunc.PythonModel):
   
    def __init__(self, trained_model):
        self.model = trained_model

    # def load_context(self, context):

    def preprocess_input(self, model_input):
        model_input_dict = convert_training_data_to_dict(
            model_input,
            building_metadata_features=building_metadata_features,
            weather_features=weather_features)
        return model_input_dict

    def postprocess_result(self, results):
        return np.hstack([results[c] for c in targets])

    def predict(self, context, model_input):
        processed_df = self.preprocess_input(model_input.copy())
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df)

# COMMAND ----------

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    fe = FeatureEngineeringClient()

    def __init__(self, train_data, building_metadata_features, weather_features, targets, batch_size=64):
        self.batch_size = batch_size
        self.targets = targets
        self.building_metadata_features = building_metadata_features
        self.weather_features = weather_features
        self.training_set = self.init_training_set(train_data = train_data)
        self.train_pd = self.init_training_features(train_data = train_data)
        self.weather_pd = self.init_weather_features()
        #self.training_set_with_index = self.training_set.rdd.zipWithIndex()
        self.on_epoch_end()

    def init_training_set(self, train_data):
        # exclude column used to just join to weather data
        # Create the training set that includes the raw input data merged 
        # with corresponding features from both feature tables
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=building_metadata_feature_lookups + weather_feature_lookups,
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id", "weather_file_city"],
        )
        return training_set
    
    def init_training_features(self, train_data):
        # Create the training set that includes the raw input data merged with corresponding features
        # from building model features only, and load into memory
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=building_metadata_feature_lookups,
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id"],
        )
        return training_set.load_df().toPandas()

    def init_weather_features(self):
        weather_features_table = fe.read_table(name = "ml.surrogate_model.weather_features")
        return weather_features_table.select(*self.weather_features, 'weather_file_city').toPandas()

    def __len__(self):
        # number of batches; last batch might be smaller
        return math.ceil(len(self.train_pd) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_pd = self.train_pd.head(self.batch_size).merge(self.weather_pd, on = 'weather_file_city', how = 'left')

        # Convert DataFrame columns to NumPy arrays and create the dictionary
        X = convert_training_data_to_dict(
            train_pd = batch_pd,
            building_metadata_features = self.building_metadata_features,
            weather_features = self.weather_features)
        y = {col: np.array(batch_pd[col]) for col in self.targets}

        return X, y
    
    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     # Generate indexes of the batch
    #     batch_pd = self.init_training_set(train_data = self.train_data.limit(self.batch_size))

    #     X = convert_training_data_to_dict(
    #         train_pd = batch_pd,
    #         building_metadata_features = self.building_metadata_features,
    #         weather_features = self.weather_features)
        
    #     y= {col: np.array(batch_pd[col]) for col in self.targets}

    #     return X, y

    # def __getitem__(self, index):
    #     'Generate one batch of data'
    #     # Generate indexes of the batch
    #     min_idx = index*self.batch_size
    #     max_idx = (index+1)*self.batch_size
    #     batch_df = self.training_set_with_index.filter(lambda element: min_idx <= element[1] < max_idx).map(lambda element: element[0]).toDF()

    #     # Convert DataFrame columns to NumPy arrays and create the dictionary
    #     batch_pd = batch_df.toPandas()
    #     X = convert_training_data_to_dict(
    #         train_pd = batch_pd,
    #         building_metadata_features = self.building_metadata_features,
    #         weather_features = self.weather_features)
    #     y= {col: np.array(batch_pd[col]) for col in self.targets}

    #     return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.training_set = self.training_set.orderBy(F.rand())
        # self.training_set_with_index = self.training_set.rdd.zipWithIndex()
        self.train_pd = self.train_pd.sample(frac=1.0)
        pass

# COMMAND ----------

# #~25s to load full dataset into memory (without joining weather and bm features)
# train_gen = DataGenerator(
#     train_data = train_data,
#     building_metadata_features=building_metadata_features,
#     weather_features=weather_features,
#     targets=targets)

# #~.1s to load a batch
# train_gen[0]

# COMMAND ----------

#Takes ~30s to initialize both generators
N = 10000 #using same as current model training for benchmarking

_train_data, _val_data = train_data.limit(N).randomSplit(weights=[0.8,0.2], seed=42)

train_gen = DataGenerator(
    train_data = _train_data,
    building_metadata_features=building_metadata_features,
    weather_features=weather_features,
    targets=targets)

val_gen = DataGenerator(
    train_data = _val_data,
    building_metadata_features=building_metadata_features,
    weather_features=weather_features,
    targets=targets)

# COMMAND ----------

# other than removing catagorical feature processing
# and dependencies on the datagen class, this is identical to model in model.py
# TODO: completely align models so they can read from same file
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

#~40s/epoch
with mlflow.start_run() as run:

    training_params = {
        "epochs": 10,
        "batch_size": 64}
    
    layer_params = {
        'activation': 'leaky_relu',
        'dtype': np.float32,
    }

    model = create_model(layer_params)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        verbose=2,
        **training_params)

    pyfunc_model = SurrogateModelingWrapper(model)

    fe.log_model(
        model=pyfunc_model,
        artifact_path="pyfunc_surrogate_model_prediction",
        flavor=mlflow.pyfunc,
        training_set=train_gen.training_set,
        registered_model_name=model_name,
    )

# COMMAND ----------

#test out prediction
results = model.predict(val_gen[0][0])
np.hstack([results[c] for c in targets])

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

# batch inference on small set of held out test set
latest_model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_model_version}"

batch_pred = fe.score_batch(model_uri=model_uri, df=test_data.limit(10), result_type=ArrayType(DoubleType()))
for i, target in enumerate(targets):
    batch_pred = batch_pred.withColumn(f"{target}_pred", F.col('prediction')[i])
batch_pred.display()
