# Databricks notebook source
# MAGIC %md # Feature Importance Analysis

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import catboost as cb
import pyspark.sql.functions as F
from sklearn.model_selection import train_test_split
import shap

# COMMAND ----------

# # Paths
# FIGPATH = G.EXPORT_FPATH / "sumo_feature_importances" / "figures"

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

def plot_shap_values(shap_values, fig_title, n_feature_display=None, save_figure=False):
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
    fig.set_size_inches(9, 7)
    fig.tight_layout()
    # util.save_figure_to_gcfs(fig=fig, gcspath=gcspath, dpi=200, transparent=True)

# COMMAND ----------

features = spark.table('ml.surrogate_model.building_features').where(F.col('upgrade_id')==0)
outputs = spark.table('ml.surrogate_model.building_simulation_outputs_annual')

# COMMAND ----------

target = 'site_energy__total'

from pyspark.sql.window import Window

w = Window.partitionBy('building_id').orderBy(F.col('upgrade_id').asc())

outputs_savings = (
    outputs
        .where(F.col('upgrade_id').isin([0, 11.05]))
        .withColumn(target, F.first(target).over(w) - F.col(target))
        .where(F.col('upgrade_id')==11.05)
    )

# COMMAND ----------

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

upgrade_features = [
    'heat_pump_sizing_methodology',
    'has_heat_pump_dryer',
    'has_induction_range',
    'has_methane_gas_appliance',
    'has_fuel_oil_appliance',
    'has_propane_appliance'
]

df_predict = (
    features
        .join(outputs_savings.select('building_id', target), on = 'building_id')
        .drop(*pkey_features, *location_features, *upgrade_features)
).toPandas()

categorical_features = [k for k,v in df_predict.dtypes.items() if v == 'object']
numeric_features = [k for k,v in df_predict.dtypes.items() if v != 'object' and k != target]

# COMMAND ----------

# -- Building Features and Responses -- #
# seperate categorical and numerical vars and cast to appropriate types
X_categorical = df_predict[categorical_features].astype(str)
X_numerical = df_predict[numeric_features].astype(float)
X = pd.concat([X_categorical, X_numerical], axis=1)
y = df_predict[target].astype(int)
# split into test/train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# -- Train Model & Get Feature Importances -- #
shap_values, model_score = get_catboost_shap_values(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, categorical_features=categorical_features
)

# -- Ouptut Plot of Feature Importances (Mean SHAP Values) -- #
plot_shap_values(
    shap_values=shap_values,
    fig_title=f"All Feature Importances {target}: R^2 = {model_score:.3f}",
    n_feature_display=20
   # gcspath=FIGPATH / f"mean_shap_values_{climate_zone}_{response}.png",
)
plt.show()


# COMMAND ----------

# -- Ouptut Plot of Feature Importances (Mean SHAP Values) -- #
plot_shap_values(
    shap_values=shap_values,
    fig_title=f"All Feature Importances {target}: R^2 = {model_score:.3f}",
    n_feature_display=30
   # gcspath=FIGPATH / f"mean_shap_values_{climate_zone}_{response}.png",
)
plt.show()

# COMMAND ----------

communities = [
    ('Pittsburgh', ['42003', '42007']),
    ('Rhode Island', ['44001', '44003', '44005', '44007', '44009']),
    ('Atlanta', ['13121', '13089', '13067', '13135', '13063']), 
    ('Denver', ['08001', '08031', '08059', '08005', '08013', '08014', '08035']),
]
