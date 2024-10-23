# Databricks notebook source
# MAGIC %md # Feature Importance Analysis

# COMMAND ----------

# MAGIC %pip install catboost shap

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import catboost as cb
import pyspark.sql.functions as F
from sklearn.model_selection import train_test_split
import shap
from cloudpathlib import CloudPath

sys.path.append("../src")
from src.util import save_figure_to_gcfs

# COMMAND ----------

# # Paths
FIGPATH = CloudPath("gs://the-cube") / "export" / "sumo_feature_importances"

# ML crap
TEST_SIZE = 0.2
RANDOM_STATE = 42
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 8,  # depth of tree
    "verbose": False,
    "allow_writing_files": False,  # don't spit out random log files plz
    "nan_mode": "Max",  # in most cases, NA seems more suitable as a max (the exception being duct leakage).. cant figure out the config file to change per param
}

# COMMAND ----------


def get_catboost_shap_values(X_train, y_train, X_test, y_test, categorical_features):
    """
    Train a catboost model to predict `y_train` from `X_train` where `X_train` has a mix of numerical and
    categorical features, where the categorical feature columns enumereated in `categorical_features`.

    Args:
        X_train (pd.DataFrame): features in train set
        y_train (pd.DataFrame): response values in train set
        X_test (pd.DataFrame): features in test set
        y_test (pd.Series): response values in test set
        categorical_features (list of str): list of column names of all categorical features

    Returns:
        shap._explanation.Explanation: SHAP explanation object for Catboost model trained on (X_train, y_train)
        mode_score: R^2 value for model trained on (X_train, y_train) and tested on (X_test, y_test)

    """

    # Fit model
    cb_model = cb.CatBoostRegressor(**CATBOOST_PARAMS)
    cb_model.fit(X_train, y_train, cat_features=list(range(len(categorical_features))))
    model_score = cb_model.score(X_test, y_test)

    # get shap values
    explainer = shap.Explainer(cb_model)
    shap_values = explainer(X_train)

    return shap_values, model_score

# COMMAND ----------

def plot_shap_values(shap_values, fig_title, gcspath, n_feature_display=None):
    """Created and saves a bar plot of SHAP values

    Args:
        shap_values (np.array): nxd array where the shap_values[i,j] contains the SHAP value for the jth feature of the ith sample
        fig_title (string): title for the plot
        fig_path (string): path to save the figure to
        n_feature_display (int optional): Number of top important features to include in the plot.
                                        Remaining feature will be aggregated into a single bar at the bottom.
                                        Defaults to None, which means all features will be displayed seperately.
    """
    if n_feature_display is None:
        n_feature_display = shap_values.values.shape[1]
    shap.plots.bar(shap_values, max_display=n_feature_display, show=False)
    plt.title(fig_title)
    fig = plt.gcf()
    fig.set_size_inches(9, 9)
    fig.tight_layout()
    save_figure_to_gcfs(fig=fig, gcspath=gcspath, dpi=200, transparent=True)

# COMMAND ----------

communities = [
    ('Pittsburgh', ['42003', '42007']),
    ('Rhode Island', ['44001', '44003', '44005', '44007', '44009']),
    ('Atlanta', ['13121', '13089', '13067', '13135', '13063']), 
    ('Denver', ['08001', '08031', '08059', '08005', '08013', '08014', '08035']),
]

# Transforming the list into a dictionary with modified geoids
communities_dict = {
    city: [f"G{geoid[:2]}0{geoid[2:]}0" for geoid in geoids]
    for city, geoids in communities
}

gisjoin_to_community = {gisjoin: community for community, gisjoins in communities_dict.items() for gisjoin in gisjoins}

# COMMAND ----------

metadata = spark.table('ml.surrogate_model.building_metadata')
features = spark.table('ml.surrogate_model.building_features').where(F.col('upgrade_id')==0)
outputs_savings = (
    spark.table('building_model.resstock_annual_savings_by_upgrade_enduse_fuel')
    .where(F.col('end_use')=='total')
    .where(F.col('upgrade_id')=='11.05')
    .groupby('building_id')
    .agg(F.sum('kwh_delta').alias('kwh_delta'),
         F.sum('cost_delta').alias('cost_delta'))
    )

# COMMAND ----------

target = 'cost_delta'


pkey_features = [
    'upgrade_id',
    'building_id'
]

location_features = [
    'weather_file_city',
    'climate_zone_moisture',
    'climate_zone_temp', 
    'weather_file_city_index'
]

other_appliance_features = [
    'heat_pump_sizing_methodology',
    'has_heat_pump_dryer',
    'has_induction_range',
    'has_methane_gas_appliance',
    'has_fuel_oil_appliance',
    'has_propane_appliance',
    'cooking_range_fuel', 
    'clothes_dryer_fuel', 
]

additional_metadata = [ 
    'ashrae_iecc_climate_zone_2004',
    'federal_poverty_level', 
    'geometry_building_type_acs', 
    'income',
    'tenure', 
    'county' #for selecting within communities              
]

df_predict = (
    features
        .join(outputs_savings.select('building_id', target), on = 'building_id')
        .join(metadata.select('building_id', *additional_metadata), on = 'building_id')
        .drop(*pkey_features, *location_features, *other_appliance_features)
        .where(F.col('heating_appliance_type') != 'ASHP')
        .where(~F.col('is_mobile_home'))
).toPandas()

categorical_features = [k for k,v in df_predict.dtypes.items() if v == 'object']
numeric_features = [k for k,v in df_predict.dtypes.items() if v != 'object' and k != target]

# COMMAND ----------

df_predict['community'] = df_predict['county'].map(gisjoin_to_community)

# COMMAND ----------

def fit_and_plot_shap_values(df, categorical_features, numeric_features, target, title_detail):
    # -- Building Features and Responses -- #
    # seperate categorical and numerical vars and cast to appropriate types
    X_categorical = df[categorical_features].astype(str)
    X_numerical = df[numeric_features].astype(float)
    X = pd.concat([X_categorical, X_numerical], axis=1)
    y = df[target].astype(int)
    # split into test/train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # -- Train Model & Get Feature Importances -- #
    shap_values, model_score = get_catboost_shap_values(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, categorical_features=categorical_features
    )

    # -- Ouptut Plot of Feature Importances (Mean SHAP Values) -- #
    plot_shap_values(
        shap_values=shap_values,
        fig_title=f"Feature Importances for HP Savings ({title_detail}): R^2 = {model_score:.3f}",
        n_feature_display=12, 
        gcspath=FIGPATH / f"mean_shap_values_cost_savings_medhp_{'_'.join(title_detail.lower().split())}.png",
    )
    plt.show()

# COMMAND ----------

for community_name, df_community in df_predict.groupby('community'):
    fit_and_plot_shap_values(df_community, categorical_features, numeric_features, target, community_name)

fit_and_plot_shap_values(df_predict.sample(frac=.3), categorical_features, numeric_features, "National")
