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
from sklearn.preprocessing import StandardScaler

!pip install tensorflow tensorflow_decision_forests
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.keras import RandomForestModel

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer
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

drop_list = heating_electric + cooling_electric + heating_fuel_oil + heating_nat_gas + heating_propane
resstock = resstock.drop(*drop_list)

# COMMAND ----------

# Resstock metadata
metadata = spark.table('building_model.resstock_metadata_w_upgrades1_5')
eligible = ['Single-Family Detached', 'Single-Family Attached']
metadata = metadata.filter(col("in_geometry_building_type_acs").isin(eligible))
drop_list = ['in_census_division', 'in_ahs_region', 'puma_geoid', 'in_weather_file_latitude', 'in_weather_file_longitude', 'in_sqft_bin', 'in_occupants_bin', 'in_income', 'in_geometry_floor_area_bin']
metadata = metadata.drop(*drop_list)

# COMMAND ----------




# COMMAND ----------

from pyspark.sql.functions import sum

resstock_yearly = (resstock).groupBy('building_id','upgrade_id').agg(
    *[sum(col).alias("sum_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']]
)

#resstock_monthly = (resstock).groupBy('building_id', 'month','upgrade_id').avg().drop('avg(building_id)', 'avg(upgrade_id)', 'avg(month)','avg(day)', 'avg(hour)')

resstock_yearly_with_metadata = (
    resstock_yearly
    .join(broadcast(metadata), on = ['building_id', 'upgrade_id'])
)


# COMMAND ----------

weather_metadata = spark.table('building_model.weather_files_metadata')
weather_data = spark.table('building_model.weather_files_data')


weather_data = weather_data.select('temp_air', 'relative_humidity', 'wind_speed' , 'ghi',
                                                          'dni', 'dhi', 'canonical_epw_filename', 'year',
                                                          'month', 'day', 'hour')

# convert celcius to farenheit

weather_data = weather_data.withColumn("temp_air", (F.col("temp_air") * (9/5)) + 32)

weather_data = weather_data.withColumn(
    "below_32", F.when(F.col("temp_air") < 32, 1).otherwise(0)
).withColumn(
    "below_41", F.when(F.col("temp_air") < 41, 1).otherwise(0)
).withColumn(
    "HDD", F.when(F.col("temp_air") > 65, F.col("temp_air") - 65).otherwise(0)
).withColumn(
    "CDD", F.when(F.col("temp_air") < 65, 65 - F.col("temp_air")).otherwise(0)
)


weather_full_daily = (
    weather_data
    .join(weather_metadata.select('canonical_epw_filename', 'county_geoid'), on = 'canonical_epw_filename')
).groupBy('day','month', 'county_geoid').agg(F.max(col("temp_air")).alias('temp_high'), F.min(col("temp_air")).alias('temp_low'),
                                             F.avg(col("temp_air")).alias('temp_avg'), F.avg(col("wind_speed")).alias('wind_speed_avg'),
                                             F.avg(col("ghi")).alias('ghi_avg'),
                                             F.avg(col("dni")).alias('dni_avg'),
                                             F.avg(col("dhi")).alias('dhi_avg'),
                                             F.avg(col('HDD')).alias('HDD'),
                                             F.avg(col('CDD')).alias('CDD'),
                                             F.sum(col('below_32')).alias('below_32'),
                                             F.sum(col('below_41')).alias('below_41'),
                                             )

weather_full_yearly = (
    weather_full_daily
).groupBy('county_geoid').agg(F.avg(col("temp_high")).alias('temp_high'), F.avg(col("temp_low")).alias('temp_low'),
                                             F.avg(col("temp_avg")).alias('temp_avg'), F.avg(col("wind_speed_avg")).alias('wind_speed_avg'),
                                             F.avg(col("ghi_avg")).alias('ghi_avg'),
                                             F.avg(col("dni_avg")).alias('dni_avg'),
                                             F.avg(col("dhi_avg")).alias('dhi_avg'),
                                             F.stddev(col("temp_high")).alias('std_temp_high'),
                                             F.stddev(col("temp_low")).alias('std_temp_low'),
                                             F.stddev(col("wind_speed_avg")).alias('std_wind_speed'),
                                             F.stddev(col("ghi_avg")).alias('std_ghi'),
                                             F.sum(col('HDD')).alias('HDD'),
                                             F.sum(col('CDD')).alias('CDD'),
                                             F.sum(col('below_41')).alias('below_41'),
                                             F.sum(col('below_32')).alias('below_32'),
                                             )


# COMMAND ----------

weather_full_yearly.toPandas()

# COMMAND ----------

resstock_yearly_with_metadata_weather = (
    resstock_yearly_with_metadata
    .join(broadcast(weather_full_yearly), on = ['county_geoid'])
)

# COMMAND ----------

resstock_yearly_with_metadata_weather.write.saveAsTable("building_model.resstock_yearly_with_metadata_weather_upgrades1_5",
                                                        mode='overwrite')

# COMMAND ----------

## lets read the data

resstock_yearly_with_metadata_weather = spark.table('building_model.resstock_yearly_with_metadata_weather_upgrades1_5')

## remove ineligible fuels like None and other fuel since Resstock doesn't model this
ineligible_fuels = ['Other Fuel', 'None']
resstock_yearly_with_metadata_weather = (resstock_yearly_with_metadata_weather.filter(~col("in_heating_fuel").isin(ineligible_fuels)))

## also remove shared cooling systems and shared heating systems (small number still left after previous filter)

resstock_yearly_with_metadata_weather = (resstock_yearly_with_metadata_weather.filter(col("in_hvac_cooling_type") != 'Shared Cooling'))

resstock_yearly_with_metadata_weather = (resstock_yearly_with_metadata_weather.filter(col("in_hvac_heating_efficiency") != 'Shared Heating'))



resstock_yearly_with_metadata_weather = (resstock_yearly_with_metadata_weather.filter(
    (col("upgrade_id") == 0)))

resstock_yearly_with_metadata_weather_df = resstock_yearly_with_metadata_weather.toPandas()


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
  if value is None:
    return 0
  elif value == "Uninsulated":
    return 0
  elif value.startswith("R-"):
    return int(value[2:])
  else:
    raise ValueError(f"Invalid insulation value: {value}")


# COMMAND ----------


# lets start by extracting the cooling efficiency. For heat pumps this comes from the SEER column
data = resstock_yearly_with_metadata_weather_df.copy()
#data = data.sample(n=1000)
mask = data["in_hvac_cooling_efficiency"].isnull() & data["in_hvac_seer_rating"].notnull()
data.loc[mask, "in_hvac_cooling_efficiency"] = "SEER " + " " + data.loc[mask, "in_hvac_seer_rating"].astype(str)



# COMMAND ----------



# Extract the cooling efficiency taking into account SEER to EER conversion
data["in_hvac_cooling_efficiency"] = data["in_hvac_cooling_efficiency"].apply(extract_cooling_efficiency)
data['in_vintage'] = data['in_vintage'].apply(vintage2age2010)
# convert from string to float
data['in_geometry_stories'] = data['in_geometry_stories'].astype(float)

data["in_ducts_insulation"] = data["in_ducts_insulation"].apply(convert_insulation)

data['in_ducts_leakage'] = data['in_ducts_leakage'].fillna(0)

data['upgrade_id'] = data['upgrade_id'].astype(str)


# COMMAND ----------


## let's use only one output variable for now
target_variable = 'sum_out_electricity_cooling_total'

additional = ['in_insulation_ceiling', 'in_insulation_floor', 'in_insulation_foundation_wall', 'in_insulation_rim_joist', 'in_insulation_roof', 'in_insulation_slab',
              'in_insulation_wall', 'in_cooling_setpoint', 'in_heating_setpoint', 'in_cooling_setpoint_has_offset',
              'in_cooling_setpoint_offset_magnitude', 'in_heating_setpoint_offset_magnitude', 'in_heating_setpoint_has_offset']

covariates = ['in_occupants', 'temp_high', 'temp_low', 'temp_avg',
       'wind_speed_avg', 'ghi_avg', 'dni_avg', 'dhi_avg', 'std_temp_high',
       'std_temp_low', 'std_wind_speed', 'std_ghi', 'in_vintage', 'in_sqft', 'in_hvac_heating_efficiency_nominal_percent', 'in_infiltration_ach50',
        'in_window_wall_ratio_mean', 'in_bedrooms', 'in_geometry_stories', 'in_ashrae_iecc_climate_zone_2004','in_income_bin_midpoint',
        'in_hvac_cooling_type', 'in_hvac_cooling_efficiency', 'in_hvac_cooling_partial_space_conditioning', 'in_is_vacant', 'in_is_rented',  'in_hvac_has_ducts', 'in_hvac_backup_heating_efficiency_nominal_percent', 'upgrade_id'] + additional





# COMMAND ----------

# Separate features and labels

#data = data[data['in_heating_fuel'] == 'Electricity']
X = data[covariates]
y = data[target_variable]
# Separate numeric and categorical features
cat_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
num_features = X.select_dtypes(exclude=['object', 'category', 'bool']).columns

# Standardize numeric features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_cat_features = pd.DataFrame(
    encoder.fit_transform(X[cat_features]).toarray(), columns=encoder.get_feature_names_out(cat_features),
    index=X.index
)

# Drop original categorical features and concatenate encoded features
X = X.drop(cat_features, axis=1)
X = pd.concat([X, encoded_cat_features], axis=1)
# Split data into train and validation sets
X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

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
        # This function is called at the start of training.
        # Initialize lists to store the losses
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Logs is a dictionary. We save the losses at the end of each epoch.
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))  # If you have validation data

    def on_train_end(self, logs=None):
        # This function is called at the end of training.
        # Plot the losses.
        plt.figure()
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


# Define the model architecture
model = Sequential(
    [
        # Input layer
        InputLayer(input_shape=(X.shape[1],)),
        #BatchNormalization(),
        Dense(256, activation="relu"),
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
        Dense(16, activation="relu"),
        BatchNormalization(),       
        Dense(16, activation="relu"),
        BatchNormalization(),       
        Dense(16, activation="relu"),
        BatchNormalization(),                                                
        #Output layer
        Dense(1, activation="linear"),
    ]
)


# Configure the model training
model.compile(loss=MeanAbsoluteError(), optimizer="adam", metrics=["mae"])

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=30)
# Add the custom callback to the training process
history = LossHistory()
# Train the model with early stopping
h = model.fit(
    X_train,
    y_train,
    epochs=60,
    verbose = 2,
    batch_size=256,
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
#loss, mae = model.evaluate(X_val, y_val)

#print(f"Validation Loss: {loss:.4f}, Validation Mean Absolute Error: {mae:.4f}")

# COMMAND ----------

# Access the training loss for each epoch
training_loss = h.history['loss']

# Access the validation loss for each epoch, if validation_data was provided
validation_loss = h.history['val_loss']

# Print the loss
for i, (tr_loss, val_loss) in enumerate(zip(training_loss, validation_loss), start=1):
    print(f"Epoch {i}, Training loss: {tr_loss}, Validation loss: {val_loss}")

# COMMAND ----------

predictions = model.predict(X_train)
loss, mae = model.evaluate(X_train, y_train, batch_size = 256)

# COMMAND ----------


y = data[target_variable]
comparison = pd.DataFrame({"Predicted": np.hstack(predictions), "Actual": y_train})
comparison['abs_error'] = np.abs(comparison["Predicted"] - comparison["Actual"])
comparison['error'] = comparison["Predicted"] - comparison["Actual"]
actuals_and_preds_upgrade2 = pd.concat([X_train_df, comparison], axis=1)
comparison.index = X_train_df.index

## Group by any characteristic and view the error
grouping_variable = ['in_bedrooms']



average_error = actuals_and_preds_upgrade2.groupby(grouping_variable)["error"].mean()
average_value = actuals_and_preds_upgrade2.groupby(grouping_variable)["Actual"].mean()
average_abs_error = actuals_and_preds_upgrade2.groupby(grouping_variable)["abs_error"].mean()

average_prediction= actuals_and_preds_upgrade2.groupby(grouping_variable)["Predicted"].mean()

WMAPE = average_abs_error/average_value
WMPE = average_error/average_value

# Create a dictionary with arrays as values and names as keys
results = {"average_error": average_error, "average_abs_error": average_abs_error, "average_value": average_value, "average_prediction": average_prediction,
        "WMAPE": WMAPE, "WMPE": WMPE}

# Create a DataFrame from the dictionary
results = pd.DataFrame(results)
results


# COMMAND ----------

comparison.mean()

# COMMAND ----------

## print the overall WMAPE

wMAPE = np.mean(actuals_and_preds_upgrade2.abs_error)/np.mean(actuals_and_preds_upgrade2.Actual)
WMPE =  np.mean(actuals_and_preds_upgrade2.error)/np.mean(actuals_and_preds_upgrade2.Actual)
print([wMAPE, WMPE])

# COMMAND ----------

## quantiles of APE, add +10 to avoid very large outliers for near 0 actuals
np.quantile(actuals_and_preds_upgrade2["abs_error"]/(actuals_and_preds_upgrade2["Actual"] + 10), q=[.01,.05,.1,0.25, 0.5, 0.75, .9, .95, .99])



# COMMAND ----------

np.corrcoef(actuals_and_preds_upgrade2["Actual"], actuals_and_preds_upgrade2["Predicted"])


# # Define the columns containing predicted and actual values
# pred_col = "Predicted"
# actual_col = "Actual"

# # Create the scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(actuals_and_preds[pred_col], actuals_and_preds[actual_col], alpha=0.01)

# plt.xlim(min(actuals_and_preds[pred_col]), np.quantile(actuals_and_preds[pred_col], q =.99))
# plt.ylim(min(actuals_and_preds[pred_col]), np.quantile(actuals_and_preds[actual_col], q =.99))

# # Add labels and title
# plt.xlabel("Predicted Values")
# plt.ylabel("Actual Values")
# plt.title("Scatter Plot of Predicted vs. Actual Values")

# # Add diagonal line for reference
# plt.plot([min(actuals_and_preds[pred_col]), max(actuals_and_preds[pred_col])], [min(actuals_and_preds[pred_col]), max(actuals_and_preds[pred_col])], linestyle="--", color="black")

# # Show the plot
# plt.show()

# COMMAND ----------


