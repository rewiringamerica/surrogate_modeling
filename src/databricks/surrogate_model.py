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
    mlflow.set_registry_uri("databricks-uc")

    # Init FeatureEngineering client
    fe = FeatureEngineeringClient()

    catalog = "ml"
    schema = "surrogate_model"

    def __init__(
        self,
        name: str,
        batch_size: int = 64,
        dtype: np.dtype = np.float32,
        artifact_path="model",
    ):
        """
        See class attributes for details on params.
        """
        self.name = name
        self.batch_size = batch_size
        self.dtype = dtype
        self.artifact_path = artifact_path

    def __str__(self):
        return f"{self.catalog}.{self.schema}.{self.name}"

    def create_model(
        self, train_gen: DataGenerator, layer_params: Dict[str, Any] = None
    ):
        """
        Create a keras model based on the given data generator and layer parameters.

        Parameters:
        - train_gen (DataGenerator):, the data generator object for training.
        - layer_params (Dict[str, Any]): the layer parameters for the model.

        Returns:
        - tensorflow.keras.src.engine.functional.Functional: the created keras model

        """
        #Dense-BatchNorm-LeakyReLU block
        def dense_batchnorm_leakyrelu(x:tf.keras.layers, n_units:int, name:str, **layer_params):
            x = layers.Dense(n_units, name=f"{name}_dense", **layer_params)(x)
            x = layers.BatchNormalization(name=f"{name}_batchnorm")(x)
            x = layers.LeakyReLU(name=f"{name}_leakyrelu")(x)
            return x
        
        #Conv-BatchNorm-LeakyReLU block
        def conv_batchnorm_relu(x:tf.keras.layers, filters:int, kernel_size:int, name:str, **layer_params):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                data_format="channels_last",
                name=f"{name}_1dconv",
                **layer_params,
            )(x)
            x = layers.BatchNormalization(name=f"{name}_conv_batchnorm")(x)
            x = layers.LeakyReLU(name=f"{name}_conv_leakyrelu")(x)
            return x
        
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

        bm = layers.BatchNormalization(name="init_batchnorm")(bm)
        bm = dense_batchnorm_leakyrelu(bm, n_units = 128, name = "first")
        bm = dense_batchnorm_leakyrelu(bm, n_units = 64, name = "second")
        bm = dense_batchnorm_leakyrelu(bm, n_units = 32, name = "third")
        bm = dense_batchnorm_leakyrelu(bm, n_units = 16, name = "fourth")

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
        wm = layers.BatchNormalization(name="init_conv_batchnorm")(wm)
        wm = conv_batchnorm_relu(wm, filters=16, kernel_size=8, name = "first")
        wm = conv_batchnorm_relu(wm, filters=8, kernel_size=8, name = "second")

        # sum the time dimension
        wm = layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1),
            dtype=layer_params["dtype"],
            # output_shape = (8,) -- needed for tf v2.16.1
        )(wm)

        wmo = models.Model(
            inputs=weather_inputs_dict, outputs=wm, name="weather_features_model"
        )

        # Combined model and separate towers for output groups
        cm = layers.Concatenate(name="combine")([bmo.output, wmo.output])
        cm = layers.Dense(24, name="combine_first_dense", activation="leaky_relu", **layer_params)(cm)
        cm = layers.Dense(24, name="combine_second_dense", activation="leaky_relu", **layer_params)(cm)
        cm = layers.Dense(16, name="third_second_dense", activation="leaky_relu", **layer_params)(cm)

        # building a separate tower for each output group
        final_outputs = {}
        for consumption_group in train_gen.targets:
            io = layers.Dense(4, name=consumption_group + "_entry", activation="leaky_relu", **layer_params)(cm)
            io = layers.Dense(2, name=consumption_group + "_mid", activation="leaky_relu", **layer_params)(io)
            io = layers.Dense(1, name=consumption_group, activation="leaky_relu")(io)
            final_outputs[consumption_group] = io

        final_model = models.Model(
            inputs={**bmo.input, **wmo.input}, outputs=final_outputs
        )

        final_model.compile(
            loss=masked_mae,
            optimizer="adam",
            # metrics=[mape],
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

    def get_latest_registered_model_uri(self, verbose: bool = True) -> str:
        """
        Returns the URI for the latest version of the registered model.

        Raises:
        - ValueError: If no version of the model has been registered yet

        Returns:
        - str: the URI for the latest version of the registered model

        """
        latest_version = self.get_latest_model_version()
        if not latest_version:
            raise ValueError(
                f"No version of the model {str(self)} has been registered yet"
            )
        if verbose:
            print(f"Returning URI for latest model version: {latest_version}")

        return f"models:/{str(self)}/{latest_version}"

    def get_model_uri(
        self, run_id: str = None, version: int = None, verbose: bool = True
    ):
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
            return f"runs:/{run_id}/{self.artifact_path}"

    def score_batch(
        self,
        test_data: DataFrame,
        run_id: str = None,
        version: int = None,
        targets: List[str] = None,
    ) -> DataFrame:
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
            model_uri=self.get_model_uri(run_id=run_id, version=version, verbose=True),
            df=test_data,
            result_type=ArrayType(DoubleType()),
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
def masked_mae(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted values, ignoring those where y_true=0.

    This custom loss function is designed for scenarios where zero values in the true values are considered to be irrelevant and should not contribute to the loss calculation. It applies a mask to both the true and predicted values to exclude these zero entries before computing the MAE. The decorator allows this function to be serialized and logged alongside the keras model. 

    Args:
    - y_true (tf.Tensor): The true values.
    - y_pred (tf.Tensor): The predicted values.

    Returns:
    - tf.Tensor: The mean absolute error computed over non-zero true values. This is just a single scalar stored in a tensor. 
    """
    # Create a mask where targets are not zero
    mask = tf.not_equal(y_true, 0)

    # Apply the mask to remove zero-target influence
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Check if the masked tensor is empty
    if tf.size(y_true_masked) == 0:
        # Return zero as the loss if no elements to process
        return tf.constant(0.0)
    else:
        # Calculate the mean absolute error on the masked data
        return tf.reduce_mean(tf.abs(y_true_masked - y_pred_masked))