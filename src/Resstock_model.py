# Databricks notebook source
from pyspark.sql.functions import broadcast
import itertools
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
import mlflow

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
import itertools
import logging
import math
import os
import calendar
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
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()
bool_features = X_train.select_dtypes(include=['bool']).columns.tolist()


# Initialize input layers and preprocessing layers for all features. 
inputs = {}
preprocessed = []

# precompute vocabularies for all features. orders of magnitude faster than having tf determine vocabularies.
vocabularies = {}
for feature in cat_features:
    vocabularies[feature] = X[feature].unique().tolist()

# Set up the layers with precomputed vocabularies
inputs = {}
preprocessed = []
for feature in cat_features:
    # Create an input layer for the categorical feature
    feature_input = Input(shape=(1,), name=feature, dtype='string')
    inputs[feature] = feature_input
    
    # Create a StringLookup layer with the precomputed vocabulary
    # Note: Add an OOV token if your model needs to handle unseen categories
    lookup_layer = StringLookup(vocabulary=vocabularies[feature], output_mode='int', mask_token=None, oov_token='[UNK]')
    indexed_data = lookup_layer(feature_input)
    
    # Create a CategoryEncoding layer for one-hot encoding using the size of the vocabulary
    # Add 1 to account for the OOV token if used
    one_hot_layer = CategoryEncoding(num_tokens=len(vocabularies[feature]) + 1, output_mode='one_hot')
    one_hot_data = one_hot_layer(indexed_data)
    preprocessed.append(one_hot_data)

for feature in num_features:
    # Calculate mean and variance for the feature from the training set. This is much faster than having tf calculate it
    feature_mean = X_train[feature].mean()
    feature_variance = X_train[feature].var()
    
    # Create a Normalization layer for the feature
    normalizer = Normalization(axis=None, mean = feature_mean, variance = feature_variance, name=f'norm_{feature}')
    
    # Directly set the weights of the Normalization layer to the precomputed statistics
    # Note: Normalization expects the variance in the second position, not the standard deviation    
    # Create the corresponding input layer
    feature_input = Input(shape=(1,), name=feature, dtype='float32')
    
    # Apply the Normalization layer to the input layer
    normalized_feature = normalizer(feature_input)
    
    # Store the input and processed features
    inputs[feature] = feature_input
    preprocessed.append(normalized_feature)

# Boolean features
for feature in bool_features:
    inputs[feature] = Input(shape=(1,), name=feature, dtype='float32')
    preprocessed.append(inputs[feature])

# Combine preprocessed inputs
all_preprocessed_inputs = tf.keras.layers.concatenate(preprocessed)


# COMMAND ----------


# Build the rest of the neural network layers on top of the preprocessed inputs
x = Dense(256, activation='relu')(all_preprocessed_inputs)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
output = Dense(1, activation='linear')(x)

# Create and compile the Model
model = Model(inputs=list(inputs.values()), outputs=output)
model.compile(optimizer='adam', loss='mae')




# Define the df_to_dataset function for converting DataFrames to tf.data.Dataset
# turn off shuffling. Not needed and changes the order of the data which becomes 
# a problem in our evaluation code later.
def df_to_dataset(features, labels, shuffle=False, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Prepare and train the model
batch_size = 128
train_ds = df_to_dataset(X_train, y_train, shuffle=False, batch_size=batch_size)
val_ds = df_to_dataset(X_val, y_val, batch_size=batch_size)


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
early_stopping = EarlyStopping(monitor="val_loss", patience=2)
# Add the custom callback to the training process
# We will fit using train_ds and val_ds which will conduct preprocessing on batches
history = LossHistory()

# Train the model with early stopping
h = model.fit(
    train_ds,
    epochs=5,
    verbose = 2,
    validation_data=val_ds,
    callbacks=[early_stopping, history],
)

# COMMAND ----------

predictions = model.predict(val_ds)
# get correct loss on training data.
mae = model.evaluate(val_ds)

# COMMAND ----------

## view error by grouping variable

y = y_val
comparison = pd.DataFrame({"Predicted": np.hstack(predictions), "Actual": y_val})
comparison['abs_error'] = np.abs(comparison["Predicted"] - comparison["Actual"])
comparison['error'] = comparison["Predicted"] - comparison["Actual"]
actuals_and_preds = pd.concat([X_val, comparison], axis=1)
comparison.index = X_val.index

## Group by any characteristic and view the error
grouping_variable = ['in_hvac_cooling_type']
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
model.save('test_model_with_preprocessing')


# COMMAND ----------

# example using hyperparameter optimization w/parallelism and MLFlow. We use the HyperOpt package.

# COMMAND ----------

## lets make a function to build a model according to certain hyperparameters
def build_model(hparams):

    # precompute vocabularies for all features. orders of magnitude faster than having tf determine vocabularies.
    vocabularies = {}
    for feature in cat_features:
        vocabularies[feature] = X[feature].unique().tolist()

    # Set up the layers with precomputed vocabularies
    inputs = {}
    preprocessed = []
    for feature in cat_features:
        # Create an input layer for the categorical feature
        feature_input = Input(shape=(1,), name=feature, dtype='string')
        inputs[feature] = feature_input
        
        # Create a StringLookup layer with the precomputed vocabulary
        # Note: Add an OOV token if your model needs to handle unseen categories
        lookup_layer = StringLookup(vocabulary=vocabularies[feature], output_mode='int', mask_token=None, oov_token='[UNK]')
        indexed_data = lookup_layer(feature_input)
        
        # Create a CategoryEncoding layer for one-hot encoding using the size of the vocabulary
        # Add 1 to account for the OOV token if used
        one_hot_layer = CategoryEncoding(num_tokens=len(vocabularies[feature]) + 1, output_mode='one_hot')
        one_hot_data = one_hot_layer(indexed_data)
        preprocessed.append(one_hot_data)

    for feature in num_features:
        # Calculate mean and variance for the feature from the training set. This is much faster than having tf calculate it
        feature_mean = X_train[feature].mean()
        feature_variance = X_train[feature].var()
        
        # Create a Normalization layer for the feature
        normalizer = Normalization(axis=None, mean = feature_mean, variance = feature_variance, name=f'norm_{feature}')
        
        # Directly set the weights of the Normalization layer to the precomputed statistics
        # Note: Normalization expects the variance in the second position, not the standard deviation    
        # Create the corresponding input layer
        feature_input = Input(shape=(1,), name=feature, dtype='float32')
        
        # Apply the Normalization layer to the input layer
        normalized_feature = normalizer(feature_input)
        
        # Store the input and processed features
        inputs[feature] = feature_input
        preprocessed.append(normalized_feature)

    # Boolean features
    for feature in bool_features:
        inputs[feature] = Input(shape=(1,), name=feature, dtype='float32')
        preprocessed.append(inputs[feature])

    # Combine preprocessed inputs
    all_preprocessed_inputs = tf.keras.layers.concatenate(preprocessed)

    
    # Build neural network layers based on hyperparameters
    x = all_preprocessed_inputs
    for _ in range(hparams['num_layers']):
        x = Dense(int(hparams['units_per_layer']), activation='relu')(x)
        if hparams['batch_norm']:
            x = BatchNormalization()(x)
    
    outputs = Dense(1, activation='linear')(x)  # Adjust based on your specific problem
    
    # Create the model
    model = Model(inputs=list(inputs.values()), outputs=outputs)
    
    # Compile the model using the learning rate from hyperparameters
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hparams['learning_rate'],  # Use the hyperparameter
        decay_steps=10000,
        decay_rate=hparams['learning_rate_decay'],
        staircase=True)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mae')
    
    return model


# COMMAND ----------


# Assuming 'data', 'covariates', and 'target_variable' are predefined. We will also be using the preprocessing and 
# data prepare dataset functionality from earlier to generate our batches. 
X = data[covariates]
y = data[target_variable]

# Split the original DataFrame into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=40)
# Separate out the numeric and categorical feature names
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()
bool_features = X_train.select_dtypes(include=['bool']).columns.tolist()



# Define the objective function for Hyperopt

def objective(params):
    model = build_model(params)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[early_stop], verbose=0)
    best_val_loss = min(history.history['val_loss'])
    return {'loss': best_val_loss, 'status': STATUS_OK}

# Define the hyperparameter space
space = {
    'num_layers': hp.quniform('num_layers', 5, 8, 1),
    'units': hp.quniform('units', 32, 512, 32),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    'decay_rate': hp.uniform('decay_rate', 0.9, 0.99),
    'use_batch_norm': hp.choice('use_batch_norm', [False, True])
}

# Run the optimization
max_evals = 10
with mlflow.start_run(tags={"mlflow.runName": "Best Model Run"}):
    trials = SparkTrials(parallelism=2)
    best_hyperparams = fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    mlflow.log_params(best_hyperparams)
    
    # Rebuild and train the best model based on the best hyperparameters
    best_hyperparams['num_layers'] = int(best_hyperparams['num_layers'])
    best_hyperparams['units'] = int(best_hyperparams['units'])
    best_model = build_model(best_hyperparams)
        
    # Log the best model to MLflow
    mlflow.keras.log_model(best_model, "best_model")
print('Best hyperparameters:', best_model)


# COMMAND ----------

hp.quniform('num_layers', 5, 8, 1)
