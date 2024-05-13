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

# import math
# from typing import Any, Dict, List, Tuple

# import mlflow
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
# from databricks.ml_features.training_set import TrainingSet
# from databricks.ml_features.entities.feature_lookup import FeatureLookup
# from pyspark.sql import DataFrame


# class DataGenerator(tf.keras.utils.Sequence):
#     """
#     A data generator for generating training data to feed to a keras model. Since the weather features are large and duplicative across many buildings,
#     The weather features are only joined to the rest of the training data (building features and targets) at generation time for the given batch.

#     Let N and M be the number of samples and targets in the training set respectively.
#     Let P_b and P_w be the number of building and weather features respectively, where P = P_b + P_w is the total number of features.

#     Attributes:
#     - building_features (List[str]): names of the building features to use in training. Defaults to class attribute.
#     - weather_features (List[str]): names of the weather features to use in training. Defaults to class attribute.
#     - upgrade_ids (List[str]): ids of upgrades to include in training set. Defaults to class attribute.
#     - consumption_group_dict (Dict[str,str]): consumption group dictionary of format {target_name : list of Resstock output columns}.
#                                             Defaults to class attribute.
#     - building_feature_table_name (str), building feature table name. Defaults to class attribute.
#     - weather_feature_table_name (str): weather feature table name. Defaults to class attribute.
#     - batch_size (int): Defaults to 64.
#     - dtype (numpy.dtype): the data type to be used for numeric features. Defaults to np.float32.
#     - targets (List[str]): targets to predict, which are the keys of self.consumption_group_dict.
#     - training_set (TrainingSet): Databricks TrainingSet object contaning targets, building feautres and weather features.
#     - training_df (pd.DataFrame): Dataframe of building features and targets of shape [N, P_b + M]. Does not include weather features.
#     - weather_features_df (pd.DataFrame): Dataframe of building features of shape [N, P_w] where each column contains a 8760-length vector.
#     - building_feature_vocab_dict (dict): Dict of format {feature_name : {"dtype": feature_dtype, "vocab": np.array
#                                         of all possible features if string feature else empty}}.
#     - fe (databricks.feature_engineering.client.FeatureEngineeringClient: client for interacting with the
#                                                                             Databricks Feature Engineering in Unity Catalog

#     """

#     # init FeatureEngineering client
#     fe = FeatureEngineeringClient()

#     # init all of the class attribute defaults
#     building_feature_table_name = "ml.surrogate_model.building_features"
#     weather_feature_table_name = "ml.surrogate_model.weather_features_hourly"

#     building_features = [
#         "heating_fuel",
#         "heating_appliance_type",
#         "heating_efficiency",
#         "heating_setpoint",
#         "heating_setpoint_offset_magnitude",
#         "ac_type",
#         "has_ac",
#         "cooled_space_proportion",
#         "cooling_efficiency_eer",
#         "cooling_setpoint",
#         "cooling_setpoint_offset_magnitude",
#         "has_ducts",
#         "ducts_insulation",
#         "ducts_leakage",
#         "infiltration_ach50",
#         "wall_material",
#         "insulation_wall",
#         "insulation_slab",
#         "insulation_rim_joist",
#         "insulation_floor",
#         "insulation_ceiling_roof",
#         "bedrooms",
#         "stories",
#         "foundation_type",
#         "attic_type",
#         "climate_zone_temp",
#         "climate_zone_moisture",
#         "sqft",
#         "vintage",
#         "occupants",
#         "orientation",
#         "window_area",
#     ]

#     weather_features = [
#         "temp_air",
#         # "relative_humidity",
#         "wind_speed",
#         # "wind_direction",
#         "ghi",
#         # "dni",
#         # "diffuse_horizontal_illum",
#         "weekend",
#     ]

#     # just hvac for now
#     consumption_group_dict = {
#         "heating": [
#             "electricity__heating_fans_pumps",
#             "electricity__heating_hp_bkup",
#             "electricity__heating",
#             "fuel_oil__heating_hp_bkup",
#             "fuel_oil__heating",
#             "natural_gas__heating_hp_bkup",
#             "natural_gas__heating",
#             "propane__heating_hp_bkup",
#             "propane__heating",
#         ],
#         "cooling": ["electricity__cooling_fans_pumps", "electricity__cooling"],
#     }

#     # baseline and HVAC upgrades
#     upgrade_ids = ["0", "1", "3", "4"]

#     def __init__(
#         self,
#         train_data: DataFrame,
#         building_features: List[str] = None,
#         weather_features: List[str] = None,
#         upgrade_ids: List[str] = None,
#         consumption_group_dict: Dict[str, str] = None,
#         building_feature_table_name: str = None,
#         weather_feature_table_name: str = None,
#         batch_size: int = 64,
#         dtype: np.dtype = np.float32,
#     ):
#         """
#         Initializes the DataGenerator object.

#         Parameters:
#         - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.
#         See class docstring for all other parameters.
#         """

#         self.upgrades = upgrade_ids or self.upgrade_ids
#         self.building_features = building_features or self.building_features
#         self.weather_features = weather_features or self.weather_features

#         self.building_feature_table_name = (
#             building_feature_table_name or self.building_feature_table_name
#         )
#         self.weather_feature_table_name = (
#             weather_feature_table_name or self.weather_feature_table_name
#         )

#         self.consumption_group_dict = (
#             consumption_group_dict or self.consumption_group_dict
#         )
#         self.targets = list(self.consumption_group_dict.keys())

#         self.batch_size = batch_size
#         self.dtype = dtype

#         self.training_set = self.init_training_set(train_data=train_data)
#         self.training_df = self.init_building_features_and_targets(
#             train_data=train_data
#         )
#         self.weather_features_df = self.init_weather_features()
#         self.building_feature_vocab_dict = self.init_building_feature_vocab_dict()

#         self.on_epoch_end()

#     def get_building_feature_lookups(self) -> FeatureLookup:
#         """
#         Returns the FeatureLookup objects for building features.

#         Returns:
#         - list: List of FeatureLookup objects for building features.
#         """
#         return [
#             FeatureLookup(
#                 table_name=self.building_feature_table_name,
#                 feature_names=self.building_features,
#                 lookup_key=["building_id", "upgrade_id"],
#             ),
#         ]

#     def get_weather_feature_lookups(self) -> FeatureLookup:
#         """
#         Returns the FeatureLookup objects for weather features.

#         Returns:
#         - list: List of FeatureLookup objects for weather features.
#         """
#         return [
#             FeatureLookup(
#                 table_name=self.weather_feature_table_name,
#                 feature_names=self.weather_features,
#                 lookup_key=["weather_file_city"],
#             ),
#         ]

#     def init_training_set(self, train_data: DataFrame) -> TrainingSet:
#         """
#         Initializes the Databricks TrainingSet object contaning targets, building feautres and weather features.

#         Parameters:
#             - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.

#         Returns:
#         - TrainingSet
#         """
#         # Join to feature tables and drop join keys since these aren't features we wanna train on
#         training_set = self.fe.create_training_set(
#             df=train_data,
#             feature_lookups=self.get_building_feature_lookups()
#             + self.get_weather_feature_lookups(),
#             label=self.targets,
#             exclude_columns=["building_id", "upgrade_id", "weather_file_city"],
#         )
#         return training_set

#     def init_building_features_and_targets(self, train_data: DataFrame) -> pd.DataFrame:
#         """
#         Loads dataframe containing building features and targets into memory.
#         Note that weather features are not joined until generation time when __get_item__() is called.

#         Parameters:
#          - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.

#         Returns:
#         - pd.DataFrame: dataframe containing building features and targets.
#         """
#         # Join to building feature tables and drop join keys since these aren't features we wanna train on
#         training_set = self.fe.create_training_set(
#             df=train_data,
#             feature_lookups=self.get_building_feature_lookups(),
#             label=self.targets,
#             exclude_columns=["building_id", "upgrade_id"],
#         )
#         return training_set.load_df().toPandas()

#     def init_weather_features(self) -> pd.DataFrame:
#         """
#         Loads dataframe weather features into memory

#         Returns:
#         - pd.DataFrame: The weather features dataframe.
#         """
#         weather_features_table = self.fe.read_table(
#             name=self.weather_feature_table_name
#         )

#         return weather_features_table.select(
#             "weather_file_city", *self.weather_features
#         ).toPandas()

#     def feature_dtype(self, feature_name: str) -> Any:
#         """
#         Returns the dtype of the feature.

#         Parameters:
#         - feature_name (str): the name of the feature.

#         Returns:
#         - The dtype of the feature, which is tf.string if catagorical
#         """
#         is_string_feature = self.training_df[feature_name].dtype == "O"
#         return tf.string if is_string_feature else self.dtype

#     def feature_vocab(self, feature_name: str) -> np.ndarray:
#         """
#         Returns the vocabulary of the feature: unique list of possible features
#         (only used for categorical).

#         Parameters:
#             - feature_name: str, the name of the feature.

#         Returns:
#         - np.ndarray: The unique list of possible categorical features
#         """
#         return self.training_df[feature_name].unique()

#     def init_building_feature_vocab_dict(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Initializes the building feature vocabulary dictionary.

#         Returns:
#             Dict of format {feature_name : {"dtype": feature_dtype, "vocab": np.array of all possible features if string feature else empty}}.
#         """
#         bm_dict = {}
#         for feature in self.building_features:
#             feature_vocab = []
#             feature_dtype = self.feature_dtype(feature)
#             if feature_dtype == tf.string:
#                 feature_vocab = self.feature_vocab(feature)
#             bm_dict[feature] = {"dtype": feature_dtype, "vocab": feature_vocab}
#         return bm_dict

#     def convert_dataframe_to_dict(
#         self, feature_df: pd.DataFrame
#     ) -> Dict[str, np.ndarray]:
#         """
#         Converts the training features from a pandas dataframe to a dictionary.

#         Parameters:
#         - feature_df: pd.DataFrame, the input features for the model of shape [N, P + 1] where feature columns
#                         for weather features contain len 8760 arrays. Note the one extra column "in_weather_city"
#                         which was used in join and will get dropped here.

#         Returns:
#             Dict[str,np.ndarray]: The preprocessed feature data in format {feature_name (str):
#                     np.array of shape [len(feature_df)] for building model features
#                     and shape [len(feature_df), 8760] for weather features}
#         """
#         X_train_bm = {col: np.array(feature_df[col]) for col in self.building_features}
#         X_train_weather = {
#             col: np.array(np.vstack(feature_df[col].values))
#             for col in self.weather_features
#         }
#         return {**X_train_bm, **X_train_weather}

#     def __len__(self) -> int:
#         """
#         Returns the number of batches.

#         Returns:
#         - int: The number of batches.
#         """
#         return math.ceil(len(self.training_df) / self.batch_size)

#     def __getitem__(
#         self, index: int
#     ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
#         """
#         Generates one batch of data.

#         Parameters:
#         - index: int, the batch index.

#         Returns:
#         - X (dict): features for batch in format {feature_name (str):
#               np.array of shape [batch_size] for building model features and shape [batch_size, 8760] for weather features}
#         - y (dict) : targets for the batch in format {target_name (str): np.array of shape [batch_size]}
#         """
#         # subset rows of targets and building features to batch
#         batch_df = self.training_df.iloc[
#             self.batch_size * index : self.batch_size * (index + 1)
#         ]
#         # join batch targets and building features to weather features
#         batch_df = batch_df.merge(
#             self.weather_features_df, on="weather_file_city", how="left"
#         )
#         # convert from df to dict
#         X = self.convert_dataframe_to_dict(feature_df=batch_df)
#         y = {col: np.array(batch_df[col]) for col in self.targets}
#         return X, y

#     def on_epoch_end(self):
#         """
#         Shuffles training set after each epoch.
#         """
#         self.training_df = self.training_df.sample(frac=1.0)

# COMMAND ----------

import numpy as np
from typing import Any, Dict, List, Tuple

import mlflow
import pyspark.sql.functions as F
import tensorflow as tf
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, DoubleType
from tensorflow import keras
from tensorflow.keras import layers, models

from datagen import DataGenerator

class Model:
    """
    A Deep Learning model for surrogate modeling energy consumption prediction.

    Attributes:
    - name (str): the name of the model. 
    - batch_size (int): the batch size for training. Defaults to 64.
    - dtype (np.dtype), the data type for the numeric features in the model. Defaults to np.float32.
    - artifact_path (str): name under which mlflow model artifact is saved. Defaults to 'model'. 
    - fe (databricks.feature_engineering.client.FeatureEngineeringClient): client for interacting with the
                                                                            Databricks Feature Engineering in Unity Catalog
    - catalog (str): name of catalog where data and models are stored. Set to "ml". 
    - schema (str): name of schema where data and models are stored. Set to "surrogate_model".

    """

    # Configure MLflow client to access models in Unity Catalog
    mlflow.set_registry_uri('databricks-uc')
    
    #Init FeatureEngineering client
    fe = FeatureEngineeringClient()

    catalog = "ml"
    schema = "surrogate_model"

    def __init__(self, name:str, batch_size:int=64, dtype:np.dtype=np.float32, artifact_path = 'model'):
        """
        See class attributes for details on params. 
        """
        self.name = name
        self.batch_size = batch_size
        self.dtype = dtype
        self.artifact_path = artifact_path

    def __str__(self):
        return f"{self.catalog}.{self.schema}.{self.name}"
    
    def create_model(self, train_gen:DataGenerator, layer_params:Dict[str, Any]=None):
        """
        Create a keras model based on the given data generator and layer parameters.

        Parameters:
        - train_gen (DataGenerator):, the data generator object for training.
        - layer_params (Dict[str, Any]): the layer parameters for the model. 

        Returns:
        - tensorflow.keras.src.engine.functional.Functional: the created keras model

        """
        # Building metadata model
        bmo_inputs_dict = {
            building_feature: layers.Input(
                name=building_feature,
                shape=(1,),
                dtype=train_gen.building_feature_vocab_dict[building_feature]["dtype"],
            )
            for building_feature in train_gen.building_features
        }

        bmo_inputs = []
        for feature, layer in bmo_inputs_dict.items():
            # encode categorical features.
            if train_gen.building_feature_vocab_dict[feature]["dtype"] == tf.string:
                encoder = layers.StringLookup(
                    name=feature + "_encoder",
                    output_mode="one_hot",
                    dtype=layer_params["dtype"],
                )
                encoder.adapt(train_gen.building_feature_vocab_dict[feature]["vocab"])
                layer = encoder(layer)
            bmo_inputs.append(layer)

        bm = layers.Concatenate(name="concat_layer", dtype=layer_params["dtype"])(
            bmo_inputs
        )
        bm = layers.Dense(128, name="first_dense", **layer_params)(bm)
        bm = layers.Dense(64, name="second_dense", **layer_params)(bm)
        bm = layers.Dense(32, name="third_dense", **layer_params)(bm)
        bm = layers.Dense(8, name="fourth_dense", **layer_params)(bm)

        bmo = models.Model(
            inputs=bmo_inputs_dict, outputs=bm, name="building_features_model"
        )

        # Weather data model
        weather_inputs_dict = {
            weather_feature: layers.Input(
                name=weather_feature,
                shape=(
                    None,
                    1,
                ),
                dtype=layer_params["dtype"],
            )
            for weather_feature in train_gen.weather_features
        }
        weather_inputs = list(weather_inputs_dict.values())

        wm = layers.Concatenate(
            axis=-1, name="weather_concat_layer", dtype=layer_params["dtype"]
        )(weather_inputs)
        wm = layers.Conv1D(
            filters=16,
            kernel_size=8,
            padding="same",
            data_format="channels_last",
            name="first_1dconv",
            **layer_params,
        )(wm)
        wm = layers.Conv1D(
            filters=8,
            kernel_size=8,
            padding="same",
            data_format="channels_last",
            name="last_1dconv",
            **layer_params,
        )(wm)

        # sum the time dimension
        wm = layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1), dtype=layer_params["dtype"]
        )(wm)

        wmo = models.Model(
            inputs=weather_inputs_dict, outputs=wm, name="weather_features_model"
        )

        # Combined model and separate towers for output groups
        cm = layers.Concatenate(name="combine_features")([bmo.output, wmo.output])
        cm = layers.Dense(16, **layer_params)(cm)
        cm = layers.Dense(16, **layer_params)(cm)

        # building a separate tower for each output group
        final_outputs = {}
        for consumption_group in train_gen.targets:
            io = layers.Dense(8, name=consumption_group + "_entry", **layer_params)(cm)
            io = layers.Dense(8, name=consumption_group + "_mid", **layer_params)(io)
            io = layers.Dense(1, name=consumption_group, **layer_params)(io)
            final_outputs[consumption_group] = io

        final_model = models.Model(
            inputs={**bmo.input, **wmo.input}, outputs=final_outputs
        )

        final_model.compile(
            loss=keras.losses.MeanAbsoluteError(),
            optimizer="adam",
            metrics=[self.mape],
        )
        return final_model
    
    def get_latest_model_version(self) -> int:
        """
        Returns the latest version of the registered model.

        Returns:
        - int, the latest version of the registered model

        """
        latest_version = 0
        mlflow_client = mlflow.tracking.client.MlflowClient()
        for mv in mlflow_client.search_model_versions(f"name='{str(self)}'"):
            version_int = int(mv.version)
            if version_int > latest_version:
                latest_version = version_int
        if latest_version == 0:
            return None
        return latest_version
    
    def get_latest_registered_model_uri(self, verbose:bool = True) -> str:
        """
        Returns the URI for the latest version of the registered model.

        Raises:
        - ValueError: If no version of the model has been registered yet

        Returns:
        - str: the URI for the latest version of the registered model

        """
        latest_version = self.get_latest_model_version()
        if not latest_version:
            raise ValueError(f"No version of the model {str(self)} has been registered yet")
        if verbose:
            print(f"Returning URI for latest model version: {latest_version}")

        return f"models:/{str(self)}/{latest_version}"
    
    def get_model_uri(self, run_id:str = None, version:int = None, verbose:bool = True):
        """
        Returns the URI for model based on:
            * the run id if specified (usually used for an unregistered model)
            * the model version if specified
            * the latest registered model otherwise 
        
        Raises:
        - ValueError: If no run_id is not passed and no version of the model has been registered yet

        Parameters:
        - run_id (str): the ID of the run. Defaults to None. 
        - version (int): the version of the model. Ignored if run_id is passed. Defaults to None. 

        Returns:
        - str, the URI for the specified model version or the latest registered model

        """
        if run_id is None:
            return self.get_latest_registered_model_uri(verbose=verbose)
        else:
             return f'runs:/{run_id}/{self.artifact_path}'
         
    
    def score_batch(self, test_data:DataFrame, run_id:str = None, version:int = None, targets:List[str] = None) -> DataFrame:
        """
        Runs inference on the test data using the specified model, using:
            * the run id if specified (usually used for an unregistered model)
            * the model version if specified
            * the latest registered model otherwise 
        Returns the input dataframe with a column containing predicted values as an array (one for each target),
        as well as a column split out for each target if the target names `targets` are passed. 

        Parameters:
        - test_data (DataFrame): the test data to run inference on containing the keys to join to feature tables on.
        - run_id (str): the ID of the run. Defaults to None. 
        - version (int): the version of the model. Ignored if run_id is passed. Defaults to None. 
        - targets (List[str]): the list of target columns.

        Returns:
        - DataFrame: test data with predictions

        """
        batch_pred = self.fe.score_batch(
            model_uri=self.get_model_uri(run_id = run_id, version= version, verbose = True),
            df=test_data,
            result_type=ArrayType(DoubleType())
        )
        if targets: 
            for i, target in enumerate(targets):
                batch_pred = batch_pred.withColumn(f"{target}_pred", F.col("prediction")[i])
        return batch_pred
    
    
    @staticmethod
    def mape(y_true, y_pred):
        """
        Computes the Mean Absolute Percentage Error between the true and predicted values, 
        ignoring elements where the true value is 0. 

        Parameters:
        - y_true (array): the true values
        - y_pred (array): the predicted values

        Returns:
        - float: the Mean Absolute Percentage Error

        """
        diff = tf.keras.backend.abs((y_true - y_pred) / y_true)
        return 100.0 * tf.keras.backend.mean(diff[y_true != 0], axis=-1)

# COMMAND ----------

# this controls the training parameters, with test mode on a much smaller training set for fewer epochs
dbutils.widgets.dropdown("Mode", "Test", ["Test", "Production"])

if dbutils.widgets.get('Mode') == 'Test':
    DEBUG = True
else:
    DEBUG = False
print(DEBUG)

# COMMAND ----------

# DBTITLE 1,Import
# %load_ext autoreload
# %autoreload 2

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from typing import Tuple, Dict

import mlflow
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import tensorflow as tf
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from tensorflow import keras

#from datagen import DataGenerator
# from model import Model

# list available GPUs
tf.config.list_physical_devices("GPU")

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

# DBTITLE 1,Data loading function
def load_data(
    consumption_group_dict= DataGenerator.consumption_group_dict,
    building_feature_table_name= DataGenerator.building_feature_table_name,
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
            AND occupants <= 10
        """
    )

    #get list of unique building ids, which will be the basis for the dataset split
    unique_building_ids = inference_data.where(F.col('upgrade_id')== 0).select("building_id")

    # Subset the data if n_subset is specified
    if n_subset is not None:
        n_total = unique_building_ids.count()
        if n_subset > n_total:
            print("'n_subset' is more than the total number of records, returning all records...")
        else:
            unique_building_ids = unique_building_ids.sample(fraction=1.0, seed=seed).limit(n_subset)

    # Split the building_ids into train, validation, and test sets (may not exactly match passed proportions)
    p_train = 1 - p_val - p_test
    train_ids, val_ids, test_ids  = unique_building_ids.randomSplit(weights=[p_train, p_val, p_test], seed=seed)

    #select train, val and test set based on building ids
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

    def __init__(self, trained_model, train_gen:DataGenerator):
        """
        Parameters:
        - trained_model: The trained mlflow keras model
        - train_gen (DataGenerator): The training data generator
        """
        self.model = trained_model
        self.building_features = train_gen.building_features
        self.weather_features = train_gen.weather_features
        self.targets = train_gen.targets

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
model = Model(name='test' if DEBUG else 'sf_detatched_hvac')

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
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
    )

    # wrap in custom class that defines pre and post processing steps to be applied when called at inference time
    pyfunc_model = SurrogateModelingWrapper(keras_model, train_gen) 

    # If in test mode, don't register the model, just pull it based on run_id in evaluation testing
    model.fe.log_model(
        model=pyfunc_model,
        artifact_path=model.artifact_path,
        flavor=mlflow.pyfunc,  # since using custom pyfunc wrapper 
        training_set=train_gen.training_set,
        registered_model_name= str(model)  # registered the model name if in DEBUG mode
        #registered_model_name= None if DEBUG else str(model)  # registered the model name if in DEBUG mode
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
if DEBUG: # evaluate the unregistered model we just logged and make sure everything runs
    print(run_id)
    pred_df = model.score_batch(test_data = test_data, run_id = run_id)
    pred_df.display()

# COMMAND ----------

pred_df.drop(*train_gen.weather_features).write.mode('overwrite').saveAsTable('ml.surrogate_model.test_predictions')

# COMMAND ----------

# MAGIC %md #### Production Mode

# COMMAND ----------

# DBTITLE 1,Evaluation functions
@udf(returnType=DoubleType())
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
    pred_df = model.score_batch(test_data = test_data.sample(fraction=.5), targets = train_gen.targets) 

# COMMAND ----------

pred_df.drop(*train_gen.weather_features).write.mode('overwrite').saveAsTable('ml.surrogate_model.test_predictions')

# COMMAND ----------

@udf("array<double>")
def APE(prediction, actual):
    return [abs(float(x - y))/y if y != 0 else None for x, y in zip(prediction, actual) ]

# COMMAND ----------

pred_df_long = (
    pred_df
        .drop(*train_gen.weather_features)
        .withColumn('hvac', F.col('heating') + F.col('cooling'))
        .withColumn("actual", F.array(train_gen.targets + ['hvac']))
        .withColumn('prediction', F.array_insert("prediction", 3, F.col('prediction')[0] + F.col('prediction')[1]))
        # .withColumn('type', 
        #         F.when(F.col('end_use') == 'cooling', F.col('ac_type'))
        #         .otherwise(F.col('heating_fuel'))
        # )
)

# COMMAND ----------



# COMMAND ----------

pred_df_long.write.mode('overwrite').saveAsTable('ml.surrogate_model.sf_detatched_hvac_predictions')

# COMMAND ----------

pred_df_in = spark.table('ml.surrogate_model.sf_detatched_hvac_predictions')

# COMMAND ----------

# .withColumn('absolute_error', F.expr("transform(arrays_zip(prediction, actual), x -> abs(x.prediction - x.actual))"))
# .withColumn('absolute_percentage_error', APE(F.col('prediction'), F.col('actual')))

# COMMAND ----------

from pyspark.sql.window import Window

w = Window().partitionBy('building_id').orderBy(F.asc('upgrade_id'))
(pred_df_long
    .withColumn('prediction_baseline', F.first(F.col('prediction')).over(w))
    .withColumn('prediction_savings', F.expr("transform(arrays_zip(prediction, prediction_baseline), x -> abs(x.prediction - x.prediction_baseline))"))
).display()

# COMMAND ----------

pred_df_in.groupby('heating_fuel').agg(
    F.mean(F.col('absolute_percentage_error')[0]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[0]).alias('median_absolute_percentage_error'),
    ).display()

pred_df_in.groupby('ac_type').agg(
    F.mean(F.col('absolute_percentage_error')[1]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[1]).alias('median_absolute_percentage_error'),
    ).display()

# COMMAND ----------

pred_df_in.groupby('heating_fuel').agg(
    F.mean(F.col('absolute_percentage_error')[2]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[2]).alias('median_absolute_percentage_error'),
    ).display()

pred_df_in.groupby('ac_type').agg(
    F.mean(F.col('absolute_percentage_error')[2]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[2]).alias('median_absolute_percentage_error'),
    ).display()

# COMMAND ----------

pred_df_in.groupby('ac_type', 'heating_fuel').agg(
    F.mean(F.col('absolute_percentage_error')[2]).alias('mean_absolute_percentage_error'),
    F.median(F.col('absolute_percentage_error')[2]).alias('median_absolute_percentage_error'),
    ).display()

# COMMAND ----------

pred_df_long.display()

# COMMAND ----------

509/4467

# COMMAND ----------

# pred_df_long = (
#     pred_df
#     .withColumn('hvac', F.col('heating') + F.col('cooling'))
#             .melt(
#                 ids = ['heating_fuel', 'ac_type', 'prediction', 'upgrade_id'], 
#                 values = ['heating', 'cooling', 'hvac'],
#                 valueColumnName='true', 
#                 variableColumnName='end_use'
#             )
#         .withColumn('pred',
#             F.when(F.col('end_use') == 'heating', F.col('prediction')[0])
#             .when(F.col('end_use') == 'cooling',F.col('prediction')[1])
#             .otherwise(F.col('prediction')[1] + F.col('prediction')[0])
#         )
#         .withColumn('absolute_error', F.abs(F.col('pred') -  F.col('true')))
#         .withColumn('absolute_percentage_error', APE(F.col('pred'), F.col('true')))
# )

# evalute_metrics(
#     df = pred_df_long, 
#     groupby_cols = ['end_use']
# ).display()

# COMMAND ----------

# DBTITLE 1,Create aggregated prediction metric table
if not DEBUG:
    pred_df_long = (
        pred_df
            # .replace({'AC' : 'Central AC'}, subset = 'ac_type')
            # .withColumn('heating_fuel', 
            #             F.when(F.col('ac_type') == 'Heat Pump', F.lit('Heat Pump'))
            #             .otherwise(F.col('heating_fuel')))
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

    # df_metrics_combined.to_csv(f'gs://the-cube/export/surrogate_model_metrics/cnn/{str(model)}_v{model.get_latest_model_version()}.csv', index=False)

# COMMAND ----------

df_metrics_combined

# COMMAND ----------

pred_df_long = (
    pred_df
        .replace({'AC' : 'Central AC'}, subset = 'ac_type')
        .withColumn('heating_fuel', 
                    F.when(F.col('ac_type') == 'Heat Pump', F.lit('Heat Pump'))
                    .otherwise(F.col('heating_fuel')))
        .withColumn('hvac', F.col('heating') + F.col('cooling'))
        # .melt(
        #     ids = ['heating_fuel', 'ac_type', 'prediction'], 
        #     values = ['heating', 'cooling', 'hvac'],
        #     valueColumnName='true', 
        #     variableColumnName='end_use'
        # )
        # .withColumn('type', 
        #         F.when(F.col('end_use') == 'cooling', F.col('ac_type'))
        #         .otherwise(F.col('heating_fuel'))
        # )
        # .withColumn('pred',
        #         F.when(F.col('end_use') == 'heating', F.col('prediction')[0])
        #         .when(F.col('end_use') == 'cooling',F.col('prediction')[1])
        #         .otherwise(F.col('prediction')[1] + F.col('prediction')[0])
        # )
        # .withColumn('absolute_error', F.abs(F.col('pred') -  F.col('true')))
        # .withColumn('absolute_percentage_error', APE(F.col('pred'), F.col('true')))
)

# metrics_by_enduse_type = evalute_metrics(
#     df = pred_df_long.where(F.col('end_use') != 'hvac'), 
#     groupby_cols = ['end_use' ,'type']
# )

# metrics_by_enduse = evalute_metrics(
#     df = pred_df_long, 
#     groupby_cols = ['end_use']
# ).withColumn('type', F.lit('Total'))

# df_metrics_combined = metrics_by_enduse_type.unionByName(metrics_by_enduse).toPandas()

# df_metrics_combined.display()

# df_metrics_combined.to_csv(f'gs://the-cube/export/surrogate_model_metrics/cnn/{str(model)}_v{model.get_latest_model_version()}.csv', index=False)
