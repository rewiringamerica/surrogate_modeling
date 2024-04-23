import numpy as np
import math
from typing import Dict, Tuple

from pyspark.sql import DataFrame

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from tensorflow import keras
import mlflow
import tensorflow as tf

from src import spark,dbutils

class DataGenerator(keras.utils.Sequence):
    fe = FeatureEngineeringClient()

    building_feature_table_name = "ml.surrogate_model.building_features"
    weather_feature_table_name = "ml.surrogate_model.weather_features_hourly"

    # features
    building_features = [
        "heating_fuel",
        "heating_appliance_type",
        "heating_efficiency",
        "heating_setpoint",
        "heating_setpoint_offset_magnitude",
        "ac_type",
        "has_ac",
        "cooled_space_proportion",
        "cooling_efficiency_eer",
        "cooling_setpoint",
        "cooling_setpoint_offset_magnitude",
        "has_ducts",
        "ducts_insulation",
        "ducts_leakage",
        "infiltration_ach50",
        "wall_material",
        "insulation_wall",
        "insulation_slab",
        "insulation_rim_joist",
        "insulation_floor",
        "insulation_ceiling_roof",
        "bedrooms",
        "stories",
        "foundation_type",
        "attic_type",
        "climate_zone_temp",
        "climate_zone_moisture",
        "sqft",
        "vintage",
        "occupants",
        "orientation",
        "window_area",
    ]

    weather_features = [
        "temp_air",
        # "relative_humidity",
        "wind_speed",
        # "wind_direction",
        "ghi",
        # "dni",
        # "diffuse_horizontal_illum",
        "weekend",
    ]

    # targets
    consumption_group_dict = {
        "heating": [
            "electricity__heating_fans_pumps",
            "electricity__heating_hp_bkup",
            "electricity__heating",
            "fuel_oil__heating_hp_bkup",
            "fuel_oil__heating",
            "natural_gas__heating_hp_bkup",
            "natural_gas__heating",
            "propane__heating_hp_bkup",
            "propane__heating",
        ],
        "cooling": ["electricity__cooling_fans_pumps", "electricity__cooling"],
    }

    upgrade_ids = ["0"]

    def __init__(
        self,
        train_data,
        building_features=None,
        weather_features=None,
        upgrade_ids=None,
        consumption_group_dict=None,
        building_feature_table_name=None,
        weather_feature_table_name=None,
        batch_size=64,
        dtype=np.float32,
    ):
        self.building_features = building_features or self.building_features
        self.weather_features = weather_features or self.weather_features
        self.building_feature_table_name = (
            building_feature_table_name or self.building_feature_table_name
        )
        self.weather_feature_table_name = weather_feature_table_name or self.weather_feature_table_name

        self.upgrades = upgrade_ids or self.upgrade_ids

        self.consumption_group_dict = (
            consumption_group_dict or self.consumption_group_dict
        )
        self.targets = list(self.consumption_group_dict.keys())

        self.batch_size = batch_size
        self.dtype = dtype

        self.training_set = self.init_training_set(train_data=train_data)
        self.building_feature_df = self.init_building_features(train_data=train_data)
        self.weather_features_df = self.init_weather_features()

        self.building_feature_vocab_dict = self.init_building_feature_vocab_dict()
        self.on_epoch_end()

    def get_building_feature_lookups(self):
        return [
            FeatureLookup(
                table_name=self.building_feature_table_name,
                feature_names=self.building_features,
                lookup_key=["building_id", "upgrade_id"],
            ),
        ]

    def get_weather_feature_lookups(self):
        # and define the lookup keys that will be used to join with the inputs at train/inference time
        return [
            FeatureLookup(
                table_name=self.weather_feature_table_name,
                feature_names=self.weather_features,
                lookup_key=["weather_file_city"],
            ),
        ]

    def init_training_set(self, train_data):

        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=self.get_building_feature_lookups()
            + self.get_weather_feature_lookups(),
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id", "weather_file_city"],
        )
        return training_set

    def init_building_features(self, train_data):
        # Create the training set that includes the raw input data merged with corresponding features
        # from building model features only, and load into memory
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=self.get_building_feature_lookups(),
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id"],
        )
        return training_set.load_df().toPandas()

    def init_weather_features(self):
        weather_features_table = self.fe.read_table(name=self.weather_feature_table_name)

        return weather_features_table.select(
            "weather_file_city", *self.weather_features
        ).toPandas()

    def feature_dtype(self, feature_name):
        is_string_feature = self.building_feature_df[feature_name].dtype == "O"
        return tf.string if is_string_feature else self.dtype

    def feature_vocab(self, feature_name):
        """Get all possible values for a feature

        This method is used to create encoders for string (categorical/ordinal)
        features
        """
        return self.building_feature_df[feature_name].unique()

    def init_building_feature_vocab_dict(self):
        bm_dict = {}
        for feature in self.building_features:
            feature_vocab = []
            feature_dtype = self.feature_dtype(feature)
            if feature_dtype == tf.string:
                feature_vocab = self.feature_vocab(feature)
            bm_dict[feature] = {"dtype": feature_dtype, "vocab": feature_vocab}
        return bm_dict

    def __len__(self):
        # number of batches; last batch might be smaller
        return math.ceil(len(self.building_feature_df) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        batch_df = self.building_feature_df.iloc[
            self.batch_size * index : self.batch_size * (index + 1)
        ]
        batch_df = batch_df.merge(
            self.weather_features_df, on="weather_file_city", how="left"
        )

        # Convert DataFrame columns to NumPy arrays and create the dictionary
        X = convert_training_data_to_dict(
            building_feature_df=batch_df,
            building_features=self.building_features,
            weather_features=self.weather_features,
        )
        y = {col: np.array(batch_df[col]) for col in self.targets}
        return X, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.building_feature_df = self.building_feature_df.sample(frac=1.0)


# this allows us to apply pre/post processing to the inference data
class SurrogateModelingWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model, train_gen):
        self.model = trained_model
        self.building_features = train_gen.building_features
        self.weather_features = train_gen.weather_features
        self.targets = train_gen.targets

    def preprocess_input(self, model_input):
        model_input_dict = convert_training_data_to_dict(
            model_input,
            building_features=self.building_features,
            weather_features=self.weather_features,
        )
        return model_input_dict

    def postprocess_result(self, results):
        return np.hstack([results[c] for c in self.targets])

    def predict(self, context, model_input):
        processed_df = self.preprocess_input(model_input.copy())
        predictions_df = self.model.predict(processed_df)
        return self.postprocess_result(predictions_df)


def convert_training_data_to_dict(building_feature_df, building_features, weather_features):
    X_train_bm = {col: np.array(building_feature_df[col]) for col in building_features}
    X_train_weather = {
        col: np.array(np.vstack(building_feature_df[col].values)) for col in weather_features
    }
    return {**X_train_bm, **X_train_weather}


def load_inference_data(
    consumption_group_dict= DataGenerator.consumption_group_dict,
    building_feature_table_name= DataGenerator.building_feature_table_name,
    outputs_table_name="ml.surrogate_model.building_upgrade_simulation_outputs_annual",
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
                        FROM {outputs_table_name} O
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


def create_dataset(
    datagen_params: Dict = {},
    n_subset: int = None,
    p_val: float = 0.15,
    p_test: float = 0.15,
    seed: int = 42,
) -> Tuple[DataGenerator, DataGenerator, DataFrame]:
    
    train_data, val_data, test_data = load_inference_data(
        n_subset=n_subset,
        p_val=p_val,
        p_test=p_test,
        seed=seed,
        **datagen_params
    )

    train_gen = DataGenerator(train_data=train_data, **datagen_params)
    val_gen = DataGenerator(train_data=val_data, **datagen_params)

    return train_gen, val_gen, test_data

