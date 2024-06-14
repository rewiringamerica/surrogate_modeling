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

from src.databricks.datagen import DataGenerator

class SurrogateModel:
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
    
    def create_model(self,train_gen:DataGenerator, layer_params:Dict[str, Any]=None):
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
        bm = layers.BatchNormalization(name = 'init_batchnorm')(bm)
        bm = layers.Dense(128, name="first_dense", **layer_params)(bm)
        bm = layers.BatchNormalization(name = 'first_batchnorm')(bm)
        bm = layers.LeakyReLU(name = 'first_leakyrelu')(bm)
        bm = layers.Dense(64, name="second_dense", **layer_params)(bm)
        bm = layers.BatchNormalization(name = 'second_batchnorm')(bm)
        bm = layers.LeakyReLU(name = 'second_leakyrelu')(bm)
        bm = layers.Dense(32, name="third_dense", **layer_params)(bm)
        bm = layers.BatchNormalization(name = 'third_batchnorm')(bm)
        bm = layers.LeakyReLU(name = 'third_leakyrelu')(bm)
        bm = layers.Dense(16, name="fourth_dense", **layer_params)(bm)
        bm = layers.BatchNormalization(name = 'fourth_batchnorm')(bm)
        bm = layers.LeakyReLU(name = 'fourth_leakyrelu')(bm)
        bm = layers.Dense(8, name="fifth_dense", **layer_params)(bm)
        bm = layers.BatchNormalization(name = 'fifth_batchnorm')(bm)
        bm = layers.LeakyReLU(name = 'fifth_leakyrelu')(bm)

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
        wm = layers.BatchNormalization(name = 'init_conv_batchnorm')(wm)
        wm = layers.Conv1D(
            filters=16,
            kernel_size=8,
            padding="same",
            data_format="channels_last",
            name="first_1dconv",
            **layer_params,
        )(wm)
        wm = layers.BatchNormalization(name = 'first_conv_batchnorm')(wm)
        wm = layers.LeakyReLU(name = 'first_conv_leakyrelu')(wm)
        wm = layers.Conv1D(
            filters=8,
            kernel_size=8,
            padding="same",
            data_format="channels_last",
            name="last_1dconv",
            **layer_params,
        )(wm)
        wm = layers.BatchNormalization(name = 'second_conv_batchnorm')(wm)
        wm = layers.LeakyReLU(name = 'second_conv_leakyrelu')(wm)

        # sum the time dimension
        wm = layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1),
            dtype=layer_params["dtype"],
            #output_shape = (8,) -- needed for tf v2.16.1
        )(wm)

        wmo = models.Model(
            inputs=weather_inputs_dict, outputs=wm, name="weather_features_model"
        )

        # Combined model and separate towers for output groups
        cm = layers.Concatenate(name="combine")([bmo.output, wmo.output])
        cm = layers.Dense(16,name="combine_first_dense",  **layer_params)(cm)
        #cm = layers.BatchNormalization(name = 'first_combine_batchnorm')(cm)
        cm = layers.LeakyReLU(name = 'first_combine_leakyrelu')(cm)
        cm = layers.Dense(16, name="combine_second_dense", **layer_params)(cm)
        #cm = layers.BatchNormalization(name = 'second_combine_batchnorm')(cm)
        cm = layers.LeakyReLU(name = 'second_combine_leakyrelu')(cm)

        # building a separate tower for each output group
        final_layer_params = layer_params.copy()
        final_layer_params['activation'] = 'leaky_relu'
        final_outputs = {}
        for consumption_group in train_gen.targets:
            io = layers.Dense(1, name=consumption_group + "_entry", **layer_params)(cm)
            #io = layers.BatchNormalization(name=consumption_group + "_entry_batchnorm")(io)
            io = layers.LeakyReLU(name=consumption_group + "_entry_leakyrelu")(io)
            io = layers.Dense(1, name=consumption_group + "_mid", **layer_params)(io)
            #io = layers.BatchNormalization(name=consumption_group + "_mid_batchnorm")(io)
            io = layers.LeakyReLU(name=consumption_group + "_mid_leakyrelu")(io)
            io = layers.Dense(1, name=consumption_group, **final_layer_params)(io)
            final_outputs[consumption_group] = io

        final_model = models.Model(
            inputs={**bmo.input, **wmo.input}, outputs=final_outputs
        )

        final_model.compile(
            loss=masked_mae,
            optimizer="adam",
            metrics=[mape],
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
        Returns the input dataframe with a column containing predicted values as an array (one for each target)

        Parameters:
        - test_data (DataFrame): the test data to run inference on containing the keys to join to feature tables on.
        - run_id (str): the ID of the run. Defaults to None. 
        - version (int): the version of the model. Ignored if run_id is passed. Defaults to None. 

        Returns:
        - DataFrame: test data with predictions

        """
        batch_pred = self.fe.score_batch(
            model_uri=self.get_model_uri(run_id = run_id, version= version, verbose = True),
            df=test_data,
            result_type=ArrayType(DoubleType())
        )
        return batch_pred
        
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
    
@keras.saving.register_keras_serializable(package="my_package", name="masked_mae")
def masked_mae(y_true, y_pred):
    # # Create a mask where targets are not zero
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    # # Apply the mask to remove zero-target influence
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    # Calculate the mean abs error
    return tf.reduce_mean(tf.math.abs(y_true_masked - y_pred_masked))
