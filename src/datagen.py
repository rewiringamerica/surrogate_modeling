import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import tensorflow as tf
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.ml_features.training_set import TrainingSet
from databricks.sdk.runtime import spark

from pyspark.sql import DataFrame

import src.globals as g
from src.utils.data_io import read_json
from src.versioning import get_most_recent_table_version


class DataGenerator(tf.keras.utils.Sequence):
    """
    A data generator for generating training data to feed to a keras model. Since the weather features are large and duplicative across many buildings,
    The weather features are only joined to the rest of the training data (building features and targets) at generation time for the given batch.

    Let N and M be the number of samples and targets in the training set respectively.
    Let P_b and P_w be the number of building and weather features respectively, where P = P_b + P_w is the total number of features.

    Attributes
    ----------
    - building_features (List[str]): names of the building features to use in training. Defaults to DataGenerator.data_params['building_features'].
    - weather_features (List[str]): names of the weather features to use in training. Defaults to DataGenerator.data_params['weather_features'].
    - targets (List[str]): names of targets to predict in training.  Defaults to list(DataGenerator.data_params['consumption_group_dict'].keys()).
    - batch_size (int): Defaults to 64.
    - dtype (numpy.dtype): the data type to be used for numeric features. Defaults to np.float32.
    - training_set (TrainingSet): Databricks TrainingSet object contaning targets, building feautres and weather features.
    - training_df (pd.DataFrame): Dataframe of building features and targets of shape [N, P_b + M]. Does not include weather features.
    - weather_features_df (pd.DataFrame): Dataframe of building features of shape [N, P_w] where each column contains a 8760-length vector.
    - weather_features_matrix (numpy.ndarray): A 3D matrix of shape (number of weather file cities, number of weather features, and number of hours in a year) representing weather data for various cities over the course of a year.
    - building_feature_vocab_dict (dict): Dict of format {feature_name : {"dtype": feature_dtype, "vocab": np.array
                                        of all possible features if string feature else empty}}.
    - fe (databricks.feature_engineering.client.FeatureEngineeringClient: client for interacting with the
                                                                            Databricks Feature Engineering in Unity Catalog

    """

    # init FeatureEngineering client
    fe = FeatureEngineeringClient()

    # This could be adjusted manually if wanted to use different version numbers
    # TODO: This is a little janky, and running spark commands hereis not ideadl
    # but we'll move over to a more robust process later and this works for now
    building_feature_table_name = get_most_recent_table_version(g.BUILDING_FEATURE_TABLE)
    weather_feature_table_name = get_most_recent_table_version(g.WEATHER_FEATURE_TABLE)
    target_table_name = get_most_recent_table_version(g.ANNUAL_OUTPUTS_TABLE)
    # load the default features, targets, and upgrades to use for this training run based on params stored in current version's config in GCS
    data_params = read_json(g.GCS_CURRENT_VERSION_ARTIFACT_PATH / "features_targets_upgrades.json")

    def __init__(
        self,
        train_data: DataFrame,
        building_features: List[str] = None,
        weather_features: List[str] = None,
        targets: List[str] = None,
        batch_size: int = 256,
        dtype: np.dtype = np.float32,
    ):
        """
        Initializes the DataGenerator object.

        Parameters
        ----------
        - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.
        See class docstring for all other parameters.
        """
        # Read from config file if not passed
        if building_features is None:
            building_features = DataGenerator.data_params["building_features"]
        if weather_features is None:
            weather_features = DataGenerator.data_params["weather_features"]
        if targets is None:
            targets = list(DataGenerator.data_params["consumption_group_dict"].keys())

        self.building_features = building_features
        self.weather_features = weather_features
        self.targets = targets

        self.batch_size = batch_size
        self.dtype = dtype

        self.training_set = self.init_training_set(train_data=train_data)
        self.training_df = self.init_building_features_and_targets(train_data=train_data)

        self.weather_features_df = self.init_weather_features()
        self.weather_features_matrix = np.stack(
            self.weather_features_df.sort_values(by="weather_file_city_index")[self.weather_features]
            .apply(lambda row: np.stack(row), axis=1)
            .values
        )
        self.building_feature_vocab_dict = self.init_building_feature_vocab_dict()

        self.on_epoch_end()

    def get_building_feature_lookups(self) -> FeatureLookup:
        """
        Returns the FeatureLookup objects for building features.

        Returns
        -------
        - list: List of FeatureLookup objects for building features.
        """
        return [
            FeatureLookup(
                table_name=self.building_feature_table_name,
                feature_names=self.building_features + ["weather_file_city_index"],
                lookup_key=["building_id", "upgrade_id", "weather_file_city"],
            ),
        ]

    def get_weather_feature_lookups(self) -> FeatureLookup:
        """
        Returns the FeatureLookup objects for weather features.

        Returns
        -------
        - list: List of FeatureLookup objects for weather features.
        """
        return [
            FeatureLookup(
                table_name=self.weather_feature_table_name,
                feature_names=self.weather_features,
                lookup_key=["weather_file_city"],
            ),
        ]

    def init_training_set(
        self,
        train_data: DataFrame,
        exclude_columns: List[str] = ["building_id", "upgrade_id", "weather_file_city"],
    ) -> TrainingSet:
        """
        Initializes the Databricks TrainingSet object contaning targets, building feautres and weather features.

        Parameters
        ----------
            - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.
            - exclude_columns (list of str): columns to be excluded from the output training set.
                                             Defaults to the join keys: ["building_id", "upgrade_id", "weather_file_city"].

        Returns
        -------
        - TrainingSet
        """
        # Join the feature tables
        training_set = self.fe.create_training_set(
            df=train_data,
            feature_lookups=self.get_building_feature_lookups() + self.get_weather_feature_lookups(),
            label=self.targets,
            exclude_columns=exclude_columns,
        )
        return training_set

    def init_building_features_and_targets(self, train_data: DataFrame) -> pd.DataFrame:
        """
        Loads dataframe containing building features and targets into memory.
        Note that weather features are not joined until generation time when __get_item__() is called.

        Parameters
        ----------
         - train_data (DataFrame): the training data containing the targets and keys to join to the feature tables.

        Returns
        -------
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

        Returns
        -------
        - pd.DataFrame: The weather features dataframe.
        """
        weather_features_table = self.fe.read_table(name=self.weather_feature_table_name)

        return weather_features_table.select("weather_file_city_index", *self.weather_features).toPandas()

    def feature_dtype(self, feature_name: str) -> Any:
        """
        Returns the dtype of the feature, which is tf.string
        if object, otherwise self.dtype

        Parameters
        ----------
        - feature_name (str): the name of the feature.

        Returns
        -------
        - The dtype of the feature, which is tf.string if catagorical
        """
        is_string_feature = self.training_df[feature_name].dtype == "O"
        return tf.string if is_string_feature else self.dtype

    def feature_vocab(self, feature_name: str) -> np.ndarray:
        """
        Returns the vocabulary of the feature: unique list of possible values a categorical feature can take on
        (only used for categorical).

        Parameters
        ----------
            - feature_name: str, the name of the feature.

        Returns
        -------
        - np.ndarray: The unique list of possible values a categorical feature can take on
        """
        return self.training_df[feature_name].unique()

    def init_building_feature_vocab_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Initializes the building feature vocabulary dictionary.

        Returns
        -------
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

    def convert_dataframe_to_dict(self, feature_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Converts the training features from a pandas dataframe to a dictionary.

        Parameters
        ----------
        - feature_df: pd.DataFrame, the input features for the model of shape [N, P + 1] where feature columns
                        for weather features contain len 8760 arrays. Note the one extra column "in_weather_city"
                        which was used in join and will get dropped here.

        Returns
        -------
            Dict[str,np.ndarray]: preprocessed feature data in format {feature_name (str): np.array of shape [N]}
        """
        return {col: np.array(feature_df[col]) for col in self.building_features + ["weather_file_city_index"]}

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns
        -------
        - int: The number of batches.
        """
        return math.ceil(len(self.training_df) / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generates one batch of data.

        Parameters
        ----------
        - index: int, the batch index.

        Returns
        -------
        - X (dict): features for batch in format {feature_name (str): np.array of shape [batch_size]}
        - y (dict) : targets for the batch in format {target_name (str): np.array of shape [batch_size]}
        """
        # subset rows of targets and building features to batch
        batch_df = self.training_df.iloc[self.batch_size * index : self.batch_size * (index + 1)]
        # convert from df to dict
        X = self.convert_dataframe_to_dict(feature_df=batch_df)
        y = {col: np.array(batch_df[col]) for col in self.targets}
        return X, y

    def on_epoch_end(self):
        """
        Shuffles training set after each epoch.
        """
        self.training_df = self.training_df.sample(frac=1.0)


def load_data(
    consumption_group_dict=None,
    upgrade_ids: List[float] = None,
    p_val=0.2,
    p_test=0.1,
    n_train=None,
    n_test=None,
    seed=42,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load the data for model training prediction containing the targets and the keys needed to join to feature tables,
    and split into train/val/test sets. The parameters n_train and n_test can be used to reduce the size of the data,
    by subsetting from the existing train/val/test data, meaning that the same splits are preserved.

    Parameters
    ----------
        consumption_group_dict (dict): Dictionary mapping consumption categories (e.g., 'heating') to resstock output columns to sum over.
            Defaults to DataGenerator.data_params['consumption_group_dict'].
        upgrade_ids (list): List of upgrade ids to use. Defaults to DataGenerator.data_params['upgrade_ids']
        p_val (float): Proportion of data to use for validation. Default is 0.2.
        p_test (float): Proportion of data to use for testing. Default is 0.1.
        n_train (int): Number of training records to select, where the size of the val and tests sets will be adjusted accordingly to
                    maintain the requested ratios. If number is passed that exceeds the size of p_train * all samples, then this will just be set to that max value. Default is None (select all)
        n_test (int): Number of test records to select, where the size of the train and val sets will be adjusted accordingly to maintain
                    the requested ratios. If number is passed that exceeds the size of p_test * all_samples, then this will just be set to that max value. Default is None (select all).
        seed (int): Seed for random sampling. Default is 42.

    Returns
    -------
        train data (DataFrame)
        val_data (DataFrame)
        test_data (DataFrame)

    Note that both splitting and subsetting are done approximately, so returned dataframes may not be exactly the requested size/ratio.
    """

    if n_train and n_test:
        raise ValueError("Cannot specify both n_train and n_test")
    if consumption_group_dict is None:
        consumption_group_dict = DataGenerator.data_params["consumption_group_dict"]
    if upgrade_ids is None:
        upgrade_ids = DataGenerator.data_params["upgrade_ids"]
    # Read outputs table and sum over consumption columns within each consumption group
    # join to the bm table to get required keys to join on and filter the building models based on charactaristics
    sum_str = ", ".join([f"{'+'.join(v)} AS {k}" for k, v in consumption_group_dict.items()])
    data = spark.sql(
        f"""
        SELECT B.building_id, B.upgrade_id, B.weather_file_city, {sum_str}
        FROM {DataGenerator.target_table_name} O
        INNER JOIN {DataGenerator.building_feature_table_name} B 
            ON B.upgrade_id = O.upgrade_id AND B.building_id == O.building_id
        """
    )
    data = data.where(F.col("upgrade_id").isin(upgrade_ids))

    # get list of unique building ids, which will be the basis for the dataset split
    unique_building_ids = data.where(F.col("upgrade_id") == 0).select("building_id")

    # Split the building_ids into train, validation, and test sets (may not exactly match passed proportions)
    p_train = 1 - p_val - p_test
    train_ids, val_ids, test_ids = unique_building_ids.randomSplit(weights=[p_train, p_val, p_test], seed=seed)

    # if n_train or n_test are passed, get the fraction of the train or test subset that this represents
    if n_train or n_test:
        p_baseline = unique_building_ids.count() / data.count()  # proportion of data that is the baseline upgrade

        if n_train:
            frac = np.clip(n_train * p_baseline / train_ids.count(), a_max=1.0, a_min=0.0)
        elif n_test:
            frac = np.clip(n_test * p_baseline / test_ids.count(), a_max=1.0, a_min=0.0)
    else:
        frac = 1.0

    # select train, val and test set based on building ids, subsetting to smaller sets if specified
    train_df = train_ids.sample(fraction=frac, seed=0).join(data, on="building_id")
    val_df = val_ids.sample(fraction=frac, seed=0).join(data, on="building_id")
    test_df = test_ids.sample(fraction=frac, seed=0).join(data, on="building_id")

    return train_df, val_df, test_df
