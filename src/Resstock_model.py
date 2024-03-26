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
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization, StringLookup, CategoryEncoding
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, SparkTrials


from tensorflow.keras.models import Sequential, Model
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

full_data_path = 'building_model.resstock_yearly_with_metadata_weather_upgrades'
resstock_yearly_with_metadata_weather = spark.table(full_data_path)
resstock_yearly_with_metadata_weather_df = resstock_yearly_with_metadata_weather.toPandas()
data = resstock_yearly_with_metadata_weather_df.copy()


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
        'in_hvac_cooling_type', 'in_hvac_cooling_efficiency', 'in_hvac_cooling_partial_space_conditioning', 'in_is_vacant', 'in_is_rented',  'in_hvac_has_ducts', 'in_hvac_backup_heating_efficiency_nominal_percent'] + additional




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

inputs = {feature: Input(shape=(1,), name=feature, dtype='string' if feature in cat_features else 'float32') 
          for feature in num_features + cat_features}
preprocessed_inputs = preprocess(inputs)



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


# Define the model architecture
NN_model = Sequential(
    [
        # Add an input layer that matches the shape of preprocessed data
        InputLayer(input_shape=(sum([normalizers[f].axis_size for f in num_features]) +
                                             sum([one_hot_encoders[f].layers[1].output_shape[-1] for f in cat_features]),)),
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

output = NN_model(preprocessed_inputs)
end_to_end_model = Model(inputs, output)

# Configure the model training
end_to_end_model.compile(optimizer='adam', loss='mae')  



# COMMAND ----------




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



# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
# Add the custom callback to the training process
# We will fit using train_ds and val_ds which will conduct preprocessing on batches
history = LossHistory()

# Train the model with early stopping
h = end_to_end_model.fit(
    train_ds,
    epochs=50,
    verbose = 2,
    validation_data=val_ds,
    callbacks=[early_stopping, history],
)


# COMMAND ----------

predictions = end_to_end_model.predict(X_val)
# get correct loss on training data.
mae = end_to_end_model.evaluate(train_ds, batch_size = 256)

# COMMAND ----------

## view error by grouping variable

y = y_val
comparison = pd.DataFrame({"Predicted": np.hstack(predictions), "Actual": y_val})
comparison['abs_error'] = np.abs(comparison["Predicted"] - comparison["Actual"])
comparison['error'] = comparison["Predicted"] - comparison["Actual"]
actuals_and_preds = pd.concat([X_val, comparison], axis=1)
comparison.index = X_val.index

## Group by any characteristic and view the error
grouping_variable = ['upgrade_id']
average_error = actuals_and_preds.groupby(grouping_variable)["error"].mean()
average_value = actuals_and_preds.groupby(grouping_variable)["Actual"].mean()
average_abs_error = actuals_and_preds.groupby(grouping_variable)["abs_error"].mean()
average_prediction= actuals_and_preds.groupby(grouping_variable)["Predicted"].mean()

WMAPE = average_abs_error/average_value
WMPE = average_error/average_value

# Create a dictionary with arrays as values and names as keys
results = {"average_error": average_error, "average_abs_error": average_abs_error, "average_value": average_value, "average_prediction": average_prediction,
        "WMAPE": WMAPE, "WMPE": WMPE}

# Create a DataFrame from the dictionary
results = pd.DataFrame(results)
results


# COMMAND ----------

# Save the entire end-to-end model, including preprocessing layers
end_to_end_model.save('my_model_with_preprocessing')


# COMMAND ----------

# example using hyperparameter optimization w/parallelism and MLFlow. We use the HyperOpt package.

# COMMAND ----------


# Assuming 'data', 'covariates', and 'target_variable' are predefined. We will also be using the preprocessing and 
# data prepare dataset functionality from earlier to generate our batches. 
X = data[covariates]
y = data[target_variable]

# Split the original DataFrame into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)

# Build the model based on hyperparameters from Hyperopt
def build_model(params):
    inputs = {feature: Input(shape=(1,), name=feature, dtype='string' if feature in cat_features else 'float32') 
              for feature in num_features + cat_features}
    preprocessed_inputs = preprocess(inputs)

    model = Sequential()
    model.add(InputLayer(input_shape=(sum([normalizers[f].axis_size for f in num_features]) +
                                      sum([one_hot_encoders[f].layers[1].output_shape[-1] for f in cat_features]),)))

    for i in range(int(params['num_layers'])):
        model.add(Dense(units=int(params['units']), activation='relu'))
        if params['use_batch_norm']:
            model.add(BatchNormalization())

    model.add(Dense(1))  # Assume it's a regression task; for classification, adjust accordingly

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=params['learning_rate'],
        decay_steps=10000,
        decay_rate=params['decay_rate'],
        staircase=True)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')
    
    end_to_end_model = Model(inputs, model(preprocessed_inputs))
    return end_to_end_model

# Define the objective function for Hyperopt
def objective(params):
    model = build_model(params)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop], verbose=0)
    best_val_loss = min(history.history['val_loss'])
    return {'loss': best_val_loss, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'num_layers': hp.quniform('num_layers', 4, 10, 1),
    'units': hp.quniform('units', 32, 512, 32),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    'decay_rate': hp.uniform('decay_rate', 0.8, 0.99),
    'use_batch_norm': hp.choice('use_batch_norm', [False, True])
}

# Run the optimization
max_evals = 20
with mlflow.start_run(tags={"mlflow.runName": "Best Model Run"}):
    trials = SparkTrials(parallelism=5)
    best_hyperparams = fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    mlflow.log_params(best_hyperparams)
    
    # Rebuild and train the best model based on the best hyperparameters
    best_hyperparams['num_layers'] = int(best_hyperparams['num_layers'])
    best_hyperparams['units'] = int(best_hyperparams['units'])
    best_model = build_model(best_hyperparams)
        
    # Log the best model to MLflow
    mlflow.keras.log_model(best_model, "best_model")
print('Best hyperparameters:', best)


# COMMAND ----------


