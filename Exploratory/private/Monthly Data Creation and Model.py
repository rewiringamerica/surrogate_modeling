# Databricks notebook source
from pyspark.sql.functions import broadcast
import itertools
import math
import re
from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg

from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredLogarithmicError

import itertools
import logging
import math
import os
import re
import calendar
from typing import Dict
import matplotlib.pyplot as plt


spark.conf.set("spark.sql.shuffle.partitions", 1536)

# COMMAND ----------

heating_electric = ['out_electricity_heating_fans_pumps_energy_consumption_kwh', 'out_electricity_heating_hp_bkup_energy_consumption_kwh', 'out_electricity_heating_energy_consumption_kwh']

cooling_electric = ['out_electricity_cooling_fans_pumps_energy_consumption_kwh',
                    'out_electricity_cooling_energy_consumption_kwh']

heating_nat_gas = ['out_natural_gas_heating_hp_bkup_energy_consumption_kwh','out_natural_gas_heating_energy_consumption_kwh']

heating_fuel_oil =['out_fuel_oil_heating_hp_bkup_energy_consumption_kwh','out_fuel_oil_heating_energy_consumption_kwh']

heating_propane = ['out_propane_heating_hp_bkup_energy_consumption_kwh',
                   'out_propane_heating_energy_consumption_kwh']

# COMMAND ----------

resstock = spark.table('building_model.resstock_outputs_hourly')                                                                                    
resstock = (resstock.withColumn( 
    'out_electricity_heating_total', sum(resstock[col] for col in heating_electric)).withColumn(
        'out_electricity_cooling_total', sum(resstock[col] for col in cooling_electric)).withColumn(
          'out_natural_gas_heating_total', sum(resstock[col] for col in heating_nat_gas)).withColumn(
              'out_fuel_oil_heating_total', sum(resstock[col] for col in heating_fuel_oil)).withColumn('out_propane_heating_total', sum(resstock[col] for col in heating_propane))
          ) 

drop_list = heating_electric + cooling_electric + heating_fuel_oil + heating_nat_gas +heating_propane
resstock = resstock.drop(*drop_list)

# COMMAND ----------

# Resstock metadata
metadata = spark.table('building_model.resstock_metadata')
eligible = ['Single-Family Detached', 'Single-Family Attached']
metadata = metadata.filter(col("in_geometry_building_type_acs").isin(eligible))
drop_list = ['in_census_division', 'in_ahs_region', 'puma_geoid', 'in_weather_file_latitude', 'in_weather_file_longitude', 'in_sqft_bin', 'in_occupants_bin', 'in_income', 'in_geometry_floor_area_bin']
metadata = metadata.drop(*drop_list)

# COMMAND ----------




# COMMAND ----------


resstock_monthly = (resstock).groupBy('building_id', 'month','upgrade_id').agg(
    *[avg(col).alias("avg_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']]
)

#resstock_monthly = (resstock).groupBy('building_id', 'month','upgrade_id').avg().drop('avg(building_id)', 'avg(upgrade_id)', 'avg(month)','avg(day)', 'avg(hour)')

resstock_monthly_with_metadata = (
    resstock_monthly
    .join(broadcast(metadata), on = 'building_id')
)


# COMMAND ----------

weather_metadata = spark.table('building_model.weather_files_metadata')
weather_data = spark.table('building_model.weather_files_data')


weather_data = weather_data.select('temp_air', 'relative_humidity', 'wind_speed' , 'ghi',
                                                          'dni', 'dhi', 'canonical_epw_filename', 'year',
                                                          'month', 'day', 'hour')

weather_full_daily = (
    weather_data
    .join(weather_metadata.select('canonical_epw_filename', 'county_geoid'), on = 'canonical_epw_filename')
).groupBy('day','month', 'county_geoid').agg(F.max(col("temp_air")).alias('temp_high'), F.min(col("temp_air")).alias('temp_low'),
                                             F.avg(col("temp_air")).alias('temp_avg'), F.avg(col("wind_speed")).alias('wind_speed_avg'),
                                             F.avg(col("ghi")).alias('ghi_avg'),
                                             F.avg(col("dni")).alias('dni_avg'),
                                             F.avg(col("dhi")).alias('dhi_avg')
                                             )

weather_full_monthly = (
    weather_full_daily
).groupBy('month', 'county_geoid').agg(F.avg(col("temp_high")).alias('temp_high'), F.avg(col("temp_low")).alias('temp_low'),
                                             F.avg(col("temp_avg")).alias('temp_avg'), F.avg(col("wind_speed_avg")).alias('wind_speed_avg'),
                                             F.avg(col("ghi_avg")).alias('ghi_avg'),
                                             F.avg(col("dni_avg")).alias('dni_avg'),
                                             F.avg(col("dhi_avg")).alias('dhi_avg'),
                                             F.stddev(col("temp_high")).alias('std_temp_high'),
                                             F.stddev(col("temp_low")).alias('std_temp_low'),
                                             F.stddev(col("wind_speed_avg")).alias('std_wind_speed'),
                                             F.stddev(col("ghi_avg")).alias('std_ghi'),
                                             )


# COMMAND ----------

resstock_monthly_with_metadata_weather = (
    resstock_monthly_with_metadata
    .join(broadcast(weather_full_monthly), on = ['county_geoid','month'])
)

# COMMAND ----------

resstock_monthly_with_metadata_weather.write.saveAsTable("building_model.resstock_monthly_with_metadata_weather_v1")

# COMMAND ----------

## lets read the data

resstock_monthly_with_metadata_weather = spark.table('building_model.resstock_monthly_with_metadata_weather_v1')

## remove ineligible fuels like None and other fuel since Resstock doesn't model this
ineligible_fuels = ['Other Fuel', 'None']
resstock_monthly_with_metadata_weather = (resstock_monthly_with_metadata_weather.filter(~col("in_heating_fuel").isin(ineligible_fuels)))

## also remove shared cooling systems and shared heating systems (small number still left after previous filter)

resstock_monthly_with_metadata_weather = (resstock_monthly_with_metadata_weather.filter(col("in_hvac_cooling_type") != 'Shared Cooling'))

resstock_monthly_with_metadata_weather = (resstock_monthly_with_metadata_weather.filter(col("in_hvac_heating_efficiency") != 'Shared Heating'))


## now for testing lets limit ourselves to Upgrade 0

resstock_monthly_with_metadata_weather = (resstock_monthly_with_metadata_weather.where(F.col('upgrade_id') == 0))

resstock_monthly_with_metadata_weather_df = resstock_monthly_with_metadata_weather.toPandas()


# COMMAND ----------

# helper functions to convert some string variables to numeric

SEER_TO_EER = .875
def extract_cooling_efficiency(text):
  if pd.isna(text):
    return 99
  match = re.match(r"((?:SEER|EER))\s+([\d\.]+)", text)
  if match:
    efficiency_type, value = match.groups()
    if efficiency_type == "SEER":
      value = float(value) * SEER_TO_EER
    else:
      value = float(value)
    return value
  else:
    return 99

def vintage2age2010(vintage: str) -> int:
    """ vintage of the building in the year of 2000
    >>> vintage2age2000('<1940')
    80
    >>> vintage2age2000('1960s')
    50
    """
    vintage = vintage.strip()
    if vintage.startswith('<'):  # '<1940' bin in resstock
        return 80
    else:
        return 2010 - int(vintage[:4])

def convert_insulation(value):
  if value is None or value == "Uninsulated":
    return 0
  elif value.startswith("R-"):
    return int(value[2:])
  else:
    raise ValueError(f"Invalid insulation value: {value}")


# COMMAND ----------


# lets start by extracting the cooling efficiency. For heat pumps this comes from the SEER column
data = resstock_monthly_with_metadata_weather_df.copy()
mask = data["in_hvac_cooling_efficiency"].isnull() & data["in_hvac_seer_rating"].notnull()
data.loc[mask, "in_hvac_cooling_efficiency"] = "SEER " + " " + data.loc[mask, "in_hvac_seer_rating"].astype(str)



# COMMAND ----------



# Extract the cooling efficiency taking into account SEER to EER conversion
data["in_hvac_cooling_efficiency"] = data["in_hvac_cooling_efficiency"].apply(extract_cooling_efficiency)
data['in_vintage'] = data['in_vintage'].apply(vintage2age2010)
# convert from string to float
data['in_geometry_stories'] = data['in_geometry_stories'].astype(float)
# convert month to string
data['month'] = pd.to_datetime(data["month"], format="%m").dt.month_name()
# convert insulation to number
data["in_ducts_insulation"] = data["in_ducts_insulation"].apply(convert_insulation)



# COMMAND ----------

#data = resstock_monthly_with_metadata_weather_df.copy()

## let's use only one output variable for now
target_variable = 'avg_out_natural_gas_heating_total'

covariates = ['in_occupants', 'temp_high', 'temp_low', 'temp_avg',
       'wind_speed_avg', 'ghi_avg', 'dni_avg', 'dhi_avg', 'std_temp_high',
       'std_temp_low', 'std_wind_speed', 'std_ghi', 'in_vintage', 'in_sqft', 'in_hvac_heating_efficiency_nominal_percent', 'in_infiltration_ach50',
        'in_window_wall_ratio_mean', 'in_bedrooms', 'in_geometry_stories', 'month','in_ducts_insulation', 'in_ashrae_iecc_climate_zone_2004','in_income_bin_midpoint']






# COMMAND ----------

# Separate features and labels

data = data[data['in_heating_fuel'] == 'Natural Gas']
X = data[covariates]
y = data[target_variable]

# Identify categorical features
cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_cat_features = pd.DataFrame(
    encoder.fit_transform(X[cat_features]).toarray(), columns=encoder.get_feature_names_out(cat_features)
, index = X.index)


# Drop original categorical features and concatenate encoded features
X = X.drop(cat_features, axis=1)
X = pd.concat([X, encoded_cat_features], axis=1)

# Split data into train and validation sets
X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert data to tensors
X_train = tf.convert_to_tensor(X_train_df.values)
X_val = tf.convert_to_tensor(X_val_df.values)
y_train = tf.convert_to_tensor(y_train.values)
y_val = tf.convert_to_tensor(y_val.values)

# # Prefetch and shuffle data
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
# train_dataset = train_dataset.shuffle(buffer_size=1024).prefetch(tf.data.experimental.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values))
# val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)



# COMMAND ----------


# Define a custom callback class
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Initialize lists to store loss values
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Append losses to corresponding lists
        self.train_losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])

    def on_train_end(self, logs=None):
        # Plot train and validation loss
        epochs = range(len(self.train_losses))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


# Define the model architecture
model = Sequential(
    [
        # Input layer
        BatchNormalization(),
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        # Batch normalization layer
        BatchNormalization(),
        # Hidden layers
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(16, activation="relu"),
        BatchNormalization(),                 
        # Output layer
        Dense(1, activation="linear"),
    ]
)


# Configure the model training
model.compile(loss=MeanAbsoluteError(), optimizer="adam", metrics=["mae"])

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Add the custom callback to the training process
history = LossHistory()
# Train the model with early stopping
model.fit(
    X_train,
    y_train,
    epochs=40,
    verbose = 2,
    batch_size=1024,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, history],
)


# model.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=val_dataset,
#     callbacks=[early_stopping],
# )

# Evaluate the model on validation data
loss, mae = model.evaluate(X_val, y_val)

print(f"Validation Loss: {loss:.4f}, Validation Mean Absolute Error: {mae:.4f}")

# COMMAND ----------

predictions = model.predict(X)

# COMMAND ----------


y = data[target_variable]
comparison = pd.DataFrame({"Predicted": np.hstack(predictions), "Actual": y})
comparison['abs_error'] = np.abs(comparison["Predicted"] - comparison["Actual"])
comparison['error'] = comparison["Predicted"] - comparison["Actual"]
actuals_and_preds = pd.concat([data, comparison], axis=1)
comparison.index = data.index


## Group by any characteristic and view the error
average_error = actuals_and_preds.groupby("building_id")["error"].mean()
average_value = actuals_and_preds.groupby("building_id")["Actual"].mean()
average_abs_error = actuals_and_preds.groupby("building_id")["abs_error"].mean()

average_prediction= actuals_and_preds.groupby("building_id")["Predicted"].mean()

WMAPE = average_abs_error/average_value
WMPE = average_error/average_value

# Create a dictionary with arrays as values and names as keys
results = {"average_error": average_error, "average_abs_error": average_abs_error, "average_value": average_value, "average_prediction": average_prediction,
        "WMAPE": WMAPE, "WMPE": WMPE}

# Create a DataFrame from the dictionary
results = pd.DataFrame(results)
results


# COMMAND ----------

actuals_and_preds[actuals_and_preds.building_id == 549981]

# COMMAND ----------

data.in_income_bin_midpoint.unique()

# COMMAND ----------

## print the overall WMAPE

wMAPE = np.mean(actuals_and_preds.abs_error)/np.mean(actuals_and_preds.Actual)

print(wMAPE)

# COMMAND ----------

quantiles = comparison["abs_error"].quantile([.1,0.25, 0.5, 0.6, .9])
quantiles

# COMMAND ----------

## Appendix: Ignore for Now


data = data[data['in_heating_fuel'] == 'Natural Gas']
X = data[covariates]
y = data[target_variable]


# Identify categorical features
cat_features = [
    i for i, col in enumerate(X.columns) if X[col].dtype in ["object", "category", "bool"]
]

# Build vocabulary of all levels for categorical features
cat_levels = {}
for col in X.select_dtypes(include=["object", "category", "bool"]):
    cat_levels[col] = set(X[col].unique())

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def preprocess_batch_pyfunc(x, y):
    # Data type conversion combined with one-hot encoding
    x_cat = x[:, cat_features]
    x_cat = tf.cast(x_cat, tf.string)
    for i, col in enumerate(cat_features):
        missing_levels = set(x_cat[:, i]) - cat_levels[col]
        if missing_levels:
            x_cat[x_cat[:, i].isin(missing_levels), i] = "Unknown"

    x_cat_onehot = tf.one_hot(x_cat, depth=len(cat_levels[col]))
    x_numerical = x[:, : len(cat_features)]
    x = tf.concat([x_numerical, x_cat_onehot], axis=1)
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Data generator without type conversion
def data_generator(data, labels):
    for i in range(len(data)):
        yield data.iloc[i].values, labels.iloc[i]


# Data pipelines with preprocessed batches
train_dataset = tf.data.Dataset.from_generator(
    data_generator, args=(X_train, y_train), output_types=(object, tf.float32)
)
train_dataset = train_dataset.map(
    lambda x, y: tf.py_func(preprocess_batch_pyfunc, (x, y), (tf.float32, tf.float32)),
    num_parallel_calls=AUTOTUNE,
)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    data_generator, args=(X_val, y_val), output_types=(object, tf.float32)
)
val_dataset = val_dataset.map(
    lambda x, y: tf.py_func(preprocess_batch_pyfunc, (x, y), (tf.float32, tf.float32)),
    num_parallel_calls=AUTOTUNE,
)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

