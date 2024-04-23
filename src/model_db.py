import numpy as np

import mlflow
import pyspark.sql.functions as F
import tensorflow as tf
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.types import ArrayType, DoubleType
from tensorflow import keras
from tensorflow.keras import layers, models

class Model:
    # Configure MLflow client to access models in Unity Catalog
    mlflow.set_registry_uri('databricks-uc')
    
    fe = FeatureEngineeringClient()

    catalog = "ml"
    schema = "surrogate_model"

    def __init__(self, name, batch_size=64, dtype=np.float32):

        self.name = name

        # self.time_granularity = time_granularity
        self.batch_size = batch_size
        self.dtype = dtype
        self.artifact_path = 'model'

    def __str__(self):
        return f"{self.catalog}.{self.schema}.{self.name}"
    

    def create_model(self, train_gen, layer_params=None):
        # Building model
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
            # handle categorical, ordinal, etc. features.
            # Here it is detected by dtype; perhaps explicit feature list and
            # handlers would be better
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
        bm = layers.Dense(32, name="second_dense", **layer_params)(bm)
        bm = layers.Dense(8, name="third_dense", **layer_params)(bm)

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
        cm = layers.Dense(128, **layer_params)(cm)
        cm = layers.Dense(64, **layer_params)(cm)
        cm = layers.Dense(32, **layer_params)(cm)
        cm = layers.Dense(16, **layer_params)(cm)
        cm = layers.Dense(16, **layer_params)(cm)
        # cm is a chokepoint representing embedding of a building + climate it is in

        # force output to be non-negative
        # layer_params_final = layer_params.copy()
        # layer_params_final['activation'] = 'relu'

        # building a separate tower for each output group
        final_outputs = {}
        for consumption_group in train_gen.targets:
            io = layers.Dense(8, name=consumption_group + "_entry", **layer_params)(cm)
            # ... feel free to add more layers
            io = layers.Dense(8, name=consumption_group + "_mid", **layer_params)(io)
            # no activation on the output
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

    def fit_model(self, train_gen, val_gen, epochs = 100, layer_params = {"activation": "leaky_relu", "dtype": np.float32}):
        # The code in the next cell trains a keras model and logs the model with the Feature Engineering in UC. 

        # The code starts an MLflow experiment to track training parameters and results. Note that model autologging is disabled (`mlflow.sklearn.autolog(log_models=False)`); this is because the model is logged using `fe.log_model`.

        # Disable MLflow autologging and instead log the model using Feature Engineering in UC
        mlflow.tensorflow.autolog(log_models=False)
        mlflow.sklearn.autolog(log_models=False)

        with mlflow.start_run() as run:

            keras_model = self.create_model(train_gen=train_gen, layer_params=layer_params)

            history = keras_model.fit(
                train_gen,
                validation_data=val_gen,
                epochs = epochs,
                batch_size = train_gen.batch_size, 
                verbose=2,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)]
            )

            pyfunc_model = SurrogateModelingWrapper(keras_model, train_gen)

            self.fe.log_model(
                model=pyfunc_model,
                artifact_path="model_packaged",
                flavor=mlflow.pyfunc,
                training_set=train_gen.training_set,
                registered_model_name=str(self),
            )
        return keras_model
    
    def get_latest_model_version(self):
        latest_version = 0
        mlflow_client = mlflow.tracking.client.MlflowClient()
        for mv in mlflow_client.search_model_versions(f"name='{str(self)}'"):
            version_int = int(mv.version)
            if version_int > latest_version:
                latest_version = version_int
        if latest_version == 0:
            return None
        return latest_version
    
    def get_latest_registered_model_uri(self, verbose = True):
        latest_version = self.get_latest_model_version()
        if not latest_version:
            raise ValueError(f"No version of the model {str(self)} has been registered yet")
        if verbose:
            print(f"Returning URI for latest model version: {latest_version}")

        return f"models:/{str(self)}/{latest_version}"
    
    def get_model_uri(self, run_id = None, version = None, verbose = True):
        if run_id is None:
            return self.get_latest_registered_model_uri(verbose=verbose)
        else:
             return f'runs:/{run_id}/{self.artifact_path}'
         
    
    def score_batch(self, test_data, run_id = None, version = None, targets = None):
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
        """Version of Mean Absolute Percentage Error that ignores samples where y_true = 0"""
        diff = tf.keras.backend.abs((y_true - y_pred) / y_true)
        return 100.0 * tf.keras.backend.mean(diff[y_true != 0], axis=-1)
