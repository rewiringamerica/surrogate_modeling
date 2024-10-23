# Databricks notebook source
# MAGIC %md # Feature Importance Analysis
# MAGIC
# MAGIC This notebook performs feature importance analysis for preducting heat pump cost savings across different GGRF communities. This is done by computing SHAP values on a fitted Catboost model.

# COMMAND ----------

# MAGIC %pip install catboost shap

# COMMAND ----------

from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import catboost as cb
import pyspark.sql.functions as F
from sklearn.model_selection import train_test_split
import shap
from cloudpathlib import CloudPath

from dmlutils.gcs import save_fig_to_gcs

# COMMAND ----------

# Define constants and parameters
FIGPATH = CloudPath("gs://the-cube") / "export" / "sumo_feature_importances"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 8,
    "verbose": False,
    "allow_writing_files": False,
    "nan_mode": "Max",
}

# COMMAND ----------


def get_catboost_shap_values(X_train, y_train, X_test, y_test, categorical_features):
    """
    Train a CatBoost model and calculate SHAP values.

    Args:
        X_train (pd.DataFrame): Features in train set
        y_train (pd.DataFrame): Response values in train set
        X_test (pd.DataFrame): Features in test set
        y_test (pd.Series): Response values in test set
        categorical_features (list of str): List of column names of all categorical features

    Returns:
        tuple: SHAP explanation object and R^2 score of the model
    """
    # Fit model
    cb_model = cb.CatBoostRegressor(**CATBOOST_PARAMS)
    cb_model.fit(X_train, y_train, cat_features=list(range(len(categorical_features))))
    model_score = cb_model.score(X_test, y_test)

    # get shap values
    explainer = shap.Explainer(cb_model)
    shap_values = explainer(X_train)

    return shap_values, model_score


def plot_shap_values(shap_values, fig_title, gcspath, n_feature_display=None):
    """
    Create and save a bar plot of SHAP values.

    Args:
        shap_values (np.array): SHAP values for each feature and sample
        fig_title (str): Title for the plot
        gcspath (str): Path to save the figure
        n_feature_display (int, optional): Number of top important features to display
    """
    if n_feature_display is None:
        n_feature_display = shap_values.values.shape[1]
    shap.plots.bar(shap_values, max_display=n_feature_display, show=False)
    plt.title(fig_title)
    fig = plt.gcf()
    fig.set_size_inches(12, 9)
    fig.tight_layout()
    save_fig_to_gcs(fig=fig, gcspath=gcspath, dpi=200, transparent=False)


def fit_and_plot_shap_values(df, categorical_features, numeric_features, target, title_detail, n_feature_display=None):
    """
    Fit a CatBoost model, calculate SHAP values, and plot feature importances.

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_features (list): List of categorical feature names
        numeric_features (list): List of numeric feature names
        target (str): Name of the target variable
        title_detail (str): Detail to be added to the plot title
        n_feature_display (int, optional): Number of top important features to display
    """
    # Prepare features and target
    X_categorical = df[categorical_features].astype(str)
    X_numerical = df[numeric_features].astype(float)
    X = pd.concat([X_categorical, X_numerical], axis=1)
    y = df[target].astype(int)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train model and get SHAP values
    shap_values, model_score = get_catboost_shap_values(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, categorical_features=categorical_features
    )

    # Plot and save SHAP values
    plot_shap_values(
        shap_values=shap_values,
        fig_title=f"Feature Importances for HP Savings ({title_detail}): R^2 = {model_score:.3f}",
        n_feature_display=n_feature_display,
        gcspath=FIGPATH / f"mean_shap_values_cost_savings_medhp_{'_'.join(title_detail.lower().split())}.png",
    )
    plt.show()


# COMMAND ----------

# Load and prepare data
# baseline metadata
metadata = spark.table("ml.surrogate_model.building_metadata")
# baseline sumo features derived from metadata
features = spark.table("ml.surrogate_model.building_features").where(F.col("upgrade_id") == 0)
# total energy and cost savings for med eff hp
outputs_savings = (
    spark.table("building_model.resstock_annual_savings_by_upgrade_enduse_fuel")
    .where(F.col("end_use") == "total")
    .where(F.col("upgrade_id") == "11.05")
    .groupby("building_id")
    .agg(F.sum("kwh_delta").alias("kwh_delta"), F.sum("cost_delta").alias("cost_delta"))
)

# COMMAND ----------

# Define communities and their corresponding county GEOIDs
communities = [
    ("Pittsburgh", ["42003", "42007"]),
    ("Rhode Island", ["44001", "44003", "44005", "44007", "44009"]),
    ("Atlanta", ["13121", "13089", "13067", "13135", "13063"]),
    ("Denver", ["08001", "08031", "08059", "08005", "08013", "08014", "08035"]),
]

# Convert county geoids into GISJOINs
communities_dict = {c: [f"G{geoid[:2]}0{geoid[2:]}0" for geoid in geoids] for c, geoids in communities}

# Invert the mapping to get lookup of GISJOIN to community
gisjoin_to_community = {gisjoin: community for community, gisjoins in communities_dict.items() for gisjoin in gisjoins}

# Map buildings to communities
community_mapping = F.create_map([F.lit(x) for x in chain(*gisjoin_to_community.items())])
metadata = metadata.withColumn("community", community_mapping[F.col("county")])

# COMMAND ----------

# Define target variable
target = "cost_delta"

# All the features we don't want to include
# these are just the primary key
pkey_features = ["upgrade_id", "building_id"]
# these are duplicative of other features so we don't need them
cz_features = [
    "climate_zone_moisture",
    "climate_zone_temp",
    "weather_file_city_index", 
    "weather_file_city", 
]
# these are only useful for upgrade features or are not relevant for hp upgrades
other_appliance_features = [
    "heat_pump_sizing_methodology",
    "has_heat_pump_dryer",
    "has_induction_range",
    "has_methane_gas_appliance",
    "has_fuel_oil_appliance",
    "has_propane_appliance",
    # "cooking_range_fuel",
    # "clothes_dryer_fuel",
]
# features we are not likely to be able to collect data on
# altho we may be able to collect data on the features these depend on
# so we want to drop these so those can show
unknown_features = [
    'window_ufactor',
    'window_shgc',
    'water_heater_recovery_efficiency_ef',
    'duct_insulation_r_value',
    'duct_leakage_percentage',
    'infiltration_ach50',
    'wall_material',
    'insulation_wall_r_value',
    'insulation_foundation_wall_r_value',
    'insulation_slab_r_value',
    'insulation_rim_joist_r_value',
    'insulation_floor_r_value',
    'insulation_roof_r_value',
    'insulation_ceiling_r_value',
    'lighting_efficiency', 
    'plug_load_percentage',
    'usage_level_appliances'
]

# metadata that we want to include that didn't get directly translated into features
additional_metadata = [
    "ashrae_iecc_climate_zone_2004",
    "federal_poverty_level",
    "geometry_building_type_acs",
    "income",
    "tenure",
    "community",
    "windows", 
    "county"
]

# Prepare the main dataframe for analysis
df_predict = (
    features.join(outputs_savings.select("building_id", target), on="building_id")
    .join(metadata.select("building_id", *additional_metadata), on="building_id")
    .drop(*pkey_features, *cz_features, *other_appliance_features, *unknown_features)
    .where(F.col("heating_appliance_type") != "ASHP")
    .where(~F.col("is_mobile_home"))
).toPandas()

# Identify categorical and numeric features
categorical_features = [k for k, v in df_predict.dtypes.items() if v == "object"]
numeric_features = [k for k, v in df_predict.dtypes.items() if v != "object" and k != target]

# COMMAND ----------

# Perform analysis for each community
for community_name, df_community in df_predict.groupby("community"):
    fit_and_plot_shap_values(
        df=df_community,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target=target,
        n_feature_display=15,
        title_detail=community_name
    )

# Perform analysis for a national sample
fit_and_plot_shap_values(
    df=df_predict.sample(frac=0.3),
    categorical_features=categorical_features,
    numeric_features=numeric_features,
    target=target,
    n_feature_display=15,
    title_detail="National"
)

# COMMAND ----------


