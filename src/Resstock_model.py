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

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization, StringLookup, CategoryEncoding


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
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

full_data_path = ''
resstock_yearly_with_metadata_weather = spark.table(full_data_path)
resstock_yearly_with_metadata_weather_df = resstock_yearly_with_metadata_weather.toPandas()
data = resstock_yearly_with_metadata_weather_df.copy()


# COMMAND ----------



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


# Assume 'data', 'covariates', and 'target_variable' are predefined
X = data[covariates]
y = data[target_variable]

# Split the original DataFrame into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

# Separate out the numeric and categorical feature names
cat_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_features = X_train.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()

# Define Normalization layers for numeric features using training data statistics
# This will also allow all preprocessing to part of the saved tensforflow graph
normalizers = {}
for feature in num_features:
    normalizer = Normalization(axis=None)
    feature_values = np.array(X_train[feature], dtype=np.float32)
    normalizer.adapt(feature_values)
    normalizers[feature] = normalizer

# Define StringLookup and CategoryEncoding layers for categorical features
# make one hot enoding part of the tensforflow graph
one_hot_encoders = {}
for feature in cat_features:
    string_lookup = StringLookup(vocabulary=np.unique(X_train[feature]), output_mode="int")
    one_hot_encoder = CategoryEncoding(num_tokens=string_lookup.vocabulary_size(), output_mode="one_hot")
    one_hot_encoders[feature] = Sequential([string_lookup, one_hot_encoder])

# Function to preprocess inputs
def preprocess(inputs):
    processed_inputs = {}
    # Normalize numeric features
    for feature in num_features:
        processed_inputs[feature] = normalizers[feature](inputs[feature])
    # One-hot encode categorical features
    for feature in cat_features:
        processed_inputs[feature] = one_hot_encoders[feature](inputs[feature])
    return processed_inputs





# COMMAND ----------

# define a batch size
batch_size = 128

# Convert data to tf.data.Dataset and apply preprocessing, shuffle, and batch
def prepare_dataset(X, y, shuffle=False, batch_size=batch_size):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    ds = ds.map(lambda x, y: (preprocess(x), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds

# Prepare training and validation datasets
train_ds = prepare_dataset(X_train, y_train, shuffle=True, batch_size=batch_size)
val_ds = prepare_dataset(X_val, y_val, batch_size=batch_size)


# COMMAND ----------


# Define a custom callback class
# this will plot the loss history over epochs.
# NOTE: if we include batch normalizations the train_loss on this plot will not be accurate but val loss will still be accurate.
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



# COMMAND ----------


# Define the model architecture
model = Sequential(
    [
        # Add an input layer that matches the shape of preprocessed data
        InputLayer(input_shape=(len(num_features) + len(cat_features) * len(one_hot_encoders),)),
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
model.compile(optimizer='adam', loss='mae')  



# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
# Add the custom callback to the training process
# We will fit using train_ds and val_ds which will conduct preprocessing on batches
history = LossHistory()
# Train the model with early stopping
h = model.fit(
    train_ds,
    epochs=50,
    verbose = 2,
    validation_data=val_ds,
    callbacks=[early_stopping, history],
)


# COMMAND ----------

predictions = model.predict(X_train)
# get correct loss on training data.
loss, mae = model.evaluate(X_train, y_train, batch_size = 256)

# COMMAND ----------


