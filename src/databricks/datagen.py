import math
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.ml_features.training_set import TrainingSet
from databricks.ml_features.entities.feature_lookup import FeatureLookup
from pyspark.sql import DataFrame


class DataGenerator(tf.keras.utils.Sequence):
    """
    A data generator for generating training data to feed to a keras model. Since the weather features are large and duplicative across many buildings,
    The weather features are only joined to the rest of the training data (building features and targets) at generation time for the given batch.

    Let N and M be the number of samples and targets in the training set respectively.
    Let P_b and P_w be the number of building and weather features respectively, where P = P_b + P_w is the total number of features.

    Attributes:
    - building_features (List[str]): names of the building features to use in training. Defaults to class attribute.
    - weather_features (List[str]): names of the weather features to use in training. Defaults to class attribute.
    - upgrade_ids (List[str]): ids of upgrades to include in training set. Defaults to class attribute.
    - consumption_group_dict (Dict[str,str]): consumption group dictionary of format {target_name : list of Resstock output columns}.
                                            Defaults to class attribute.
    - building_feature_table_name (str), building feature table name. Defaults to class attribute.
    - weather_feature_table_name (str): weather feature table name. Defaults to class attribute.
    - batch_size (int): Defaults to 64.
    - dtype (numpy.dtype): the data type to be used for numeric features. Defaults to np.float32.
    - targets (List[str]): targets to predict, which are the keys of self.consumption_group_dict.
    - training_set (TrainingSet): Databricks TrainingSet object contaning targets, building feautres and weather features.
    - training_df (pd.DataFrame): Dataframe of building features and targets of shape [N, P_b + M]. Does not include weather features.
    - weather_features_df (pd.DataFrame): Dataframe of building features of shape [N, P_w] where each column contains a 8760-length vector.
    - building_feature_vocab_dict (dict): Dict of format {feature_name : {"dtype": feature_dtype, "vocab": np.array
                                        of all possible features if string feature else empty}}.
    - fe (databricks.feature_engineering.client.FeatureEngineeringClient: client for interacting with the
                                                                            Databricks Feature Engineering in Unity Catalog

    """

    # init FeatureEngineering client
    fe = FeatureEngineeringClient()

    # init all of the class attribute defaults
    building_feature_table_name = "ml.surrogate_model.building_features"
    weather_feature_table_name = "ml.surrogate_model.weather_features_hourly"

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

    # just hvac for now
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

    # baseline and HVAC upgrades
    upgrade_ids = ["0", "1", "3", "4"]

    def __init__(
        self,
        train_data: DataFrame,
        building_features: List[str] = None,
        weather_features: List[str] = None,
        upgrade_ids: List[str] = None,
        consumption_group_dict: Dict[str, str] = None,
        building_feature_table_name: str = None,
        weather_feature_table_name: str = None,
        batch_size: int = 64,
        dtype: np.dtype = np.float32,
    ):
        """
        Initializes the DataGenerator object.

        Parameters:
        - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.
        See class docstring for all other parameters.
        """

        self.upgrades = upgrade_ids or self.upgrade_ids
        self.building_features = building_features or self.building_features
        self.weather_features = weather_features or self.weather_features

        self.building_feature_table_name = (
            building_feature_table_name or self.building_feature_table_name
        )
        self.weather_feature_table_name = (
            weather_feature_table_name or self.weather_feature_table_name
        )

        self.consumption_group_dict = (
            consumption_group_dict or self.consumption_group_dict
        )
        self.targets = list(self.consumption_group_dict.keys())

        self.batch_size = batch_size
        self.dtype = dtype

        self.training_set = self.init_training_set(train_data=train_data)
        self.training_df = self.init_building_features_and_targets(
            train_data=train_data
        )
        self.weather_features_df = self.init_weather_features()
        self.building_feature_vocab_dict = self.init_building_feature_vocab_dict()

        self.on_epoch_end()

    def get_building_feature_lookups(self) -> FeatureLookup:
        """
        Returns the FeatureLookup objects for building features.

        Returns:
        - list: List of FeatureLookup objects for building features.
        """
        return [
            FeatureLookup(
                table_name=self.building_feature_table_name,
                feature_names=self.building_features,
                lookup_key=["building_id", "upgrade_id"],
            ),
        ]

    def get_weather_feature_lookups(self) -> FeatureLookup:
        """
        Returns the FeatureLookup objects for weather features.

        Returns:
        - list: List of FeatureLookup objects for weather features.
        """
        return [
            FeatureLookup(
                table_name=self.weather_feature_table_name,
                feature_names=self.weather_features,
                lookup_key=["weather_file_city"],
            ),
        ]

    def init_training_set(self, train_data: DataFrame) -> TrainingSet:
        """
        Initializes the Databricks TrainingSet object contaning targets, building feautres and weather features.

        Parameters:
            - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.

        Returns:
        - TrainingSet
        """
        # Join to feature tables and drop join keys since these aren't features we wanna train on
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=self.get_building_feature_lookups()
            + self.get_weather_feature_lookups(),
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id", "weather_file_city"],
        )
        return training_set

    def init_building_features_and_targets(self, train_data: DataFrame) -> pd.DataFrame:
        """
        Loads dataframe containing building features and targets into memory.
        Note that weather features are not joined until generation time when __get_item__() is called.

        Parameters:
         - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.

        Returns:
        - pd.DataFrame: dataframe containing building features and targets.
        """
        # Join to building feature tables and drop join keys since these aren't features we wanna train on
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=self.get_building_feature_lookups(),
            label=self.targets,
            exclude_columns=["building_id", "upgrade_id"],
        )
        return training_set.load_df().toPandas()

    def init_weather_features(self) -> pd.DataFrame:
        """
        Loads dataframe weather features into memory

        Returns:
        - pd.DataFrame: The weather features dataframe.
        """
        weather_features_table = self.fe.read_table(
            name=self.weather_feature_table_name
        )

        return weather_features_table.select(
            "weather_file_city", *self.weather_features
        ).toPandas()

    def feature_dtype(self, feature_name: str) -> Any:
        """
        Returns the dtype of the feature.

        Parameters:
        - feature_name (str): the name of the feature.

        Returns:
        - The dtype of the feature, which is tf.string if catagorical
        """
        is_string_feature = self.training_df[feature_name].dtype == "O"
        return tf.string if is_string_feature else self.dtype

    def feature_vocab(self, feature_name: str) -> np.ndarray:
        """
        Returns the vocabulary of the feature: unique list of possible features
        (only used for categorical).

        Parameters:
            - feature_name: str, the name of the feature.

        Returns:
        - np.ndarray: The unique list of possible categorical features
        """
        return self.training_df[feature_name].unique()

    def init_building_feature_vocab_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Initializes the building feature vocabulary dictionary.

        Returns:
            Dict of format {feature_name : {"dtype": feature_dtype, "vocab": np.array of all possible features if string feature else empty}}.
        """
        bm_dict = {}
        for feature in self.building_features:
            feature_vocab = []
            feature_dtype = self.feature_dtype(feature)
            if feature_dtype == tf.string:
                feature_vocab = self.feature_vocab(feature)
            bm_dict[feature] = {"dtype": feature_dtype, "vocab": feature_vocab}
        return bm_dict

    def convert_dataframe_to_dict(
        self, feature_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Converts the training features from a pandas dataframe to a dictionary.

        Parameters:
        - feature_df: pd.DataFrame, the input features for the model of shape [N, P + 1] where feature columns
                        for weather features contain len 8760 arrays. Note the one extra column "in_weather_city"
                        which was used in join and will get dropped here.

        Returns:
            Dict[str,np.ndarray]: The preprocessed feature data in format {feature_name (str):
                    np.array of shape [len(feature_df)] for building model features
                    and shape [len(feature_df), 8760] for weather features}
        """
        X_train_bm = {col: np.array(feature_df[col]) for col in self.building_features}
        X_train_weather = {
            col: np.array(np.vstack(feature_df[col].values))
            for col in self.weather_features
        }
        return {**X_train_bm, **X_train_weather}

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
        - int: The number of batches.
        """
        return math.ceil(len(self.training_df) / self.batch_size)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generates one batch of data.

        Parameters:
        - index: int, the batch index.

        Returns:
        - X (dict): features for batch in format {feature_name (str):
              np.array of shape [batch_size] for building model features and shape [batch_size, 8760] for weather features}
        - y (dict) : targets for the batch in format {target_name (str): np.array of shape [batch_size]}
        """
        # subset rows of targets and building features to batch
        batch_df = self.training_df.iloc[
            self.batch_size * index : self.batch_size * (index + 1)
        ]
        # join batch targets and building features to weather features
        batch_df = batch_df.merge(
            self.weather_features_df, on="weather_file_city", how="left"
        )
        # convert from df to dict
        X = self.convert_dataframe_to_dict(feature_df=batch_df)
        y = {col: np.array(batch_df[col]) for col in self.targets}
        return X, y

    def on_epoch_end(self):
        """
        Shuffles training set after each epoch.
        """
        self.training_df = self.training_df.sample(frac=1.0)