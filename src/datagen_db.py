import math
from typing import Dict, Tuple

import mlflow
import numpy as np
import tensorflow as tf
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


class DataGenerator(tf.keras.utils.Sequence):
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
        X = self.convert_training_data_to_dict(building_feature_df=batch_df)
        y = {col: np.array(batch_df[col]) for col in self.targets}
        return X, y
    
    def convert_training_data_to_dict(self, building_feature_df):
        X_train_bm = {col: np.array(building_feature_df[col]) for col in self.building_features}
        X_train_weather = {
            col: np.array(np.vstack(building_feature_df[col].values)) for col in self.weather_features
        }
        return {**X_train_bm, **X_train_weather}

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.building_feature_df = self.building_feature_df.sample(frac=1.0)

