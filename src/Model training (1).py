# Databricks notebook source
import os
env_fixes = {
    "PATH": ["/usr/local/cuda/bin"],
    "LD_LIBRARY_PATH": ["/usr/local/cuda/lib64"],
}

for var, paths in env_fixes.items():
    for path in paths:
        if path not in os.environ[var]:
            os.environ[var] = path + ":" + os.environ[var]
    print(var, ":", os.environ[var])

# fix cublann OOM
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# TODO: configure 
# DataBricks workspace is mounted on an extremely slow device, 
# which becomes a bottleneck (DBFS? ~700Kbps throughput)
# it is important to get an instance with a local SSD (~300Mbps write, 700Mbps read)
os.environ['SURROGATE_MODELING_CACHE_PATH'] = '/tmp/surrogate_modeling_cache'
# the default is already pointing at GCS bucket
# os.environ['SURROGATE_MODELING_RESSTOCK_PATH'] = ...

# COMMAND ----------

!pwd

# COMMAND ----------

!whoami

# COMMAND ----------

import datagen

import keras
import keras.backend as K
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.config.list_physical_devices("GPU")

# COMMAND ----------

np.random.seed(42)  # 42 is always the answer

get_building_metadata = datagen.BuildingMetadataBuilder()
building_ids = get_building_metadata.building_ids

datagen_params = {
    'metadata_builder': get_building_metadata,
    'batch_size': 64,
}

# some magic happens here
np.random.shuffle(building_ids)

# limited to a thousand to speed up prototype testing
# TODO: remove the restriction
building_ids = building_ids[:1000]

train_buildings, test_buildings = datagen.train_test_split(building_ids, left_size=0.9)

train_gen = datagen.DataGen(train_buildings, **datagen_params)
test_gen = datagen.DataGen(test_buildings, **datagen_params)
len(train_buildings), len(test_buildings)

# COMMAND ----------

bmo_inputs_dict = {  # all values can be ints
    'building_features': layers.Input(name='building_features', shape=(len(train_gen.building_features),), dtype=np.float16)
}
bmo_inputs = list(bmo_inputs_dict.values())
m = layers.Concatenate(name='concat_layer')(bmo_inputs)
# reshape is probably unnecessary, here for compatibility with future multiple inputs
# m = layers.Reshape(target_shape=(len(train_gen.building_features),), name='reshape_layer')(m)
m = layers.Dense(16, name='second_dense', activation='leaky_relu')(m)
m = layers.Dense(8, name='third_dense', activation='leaky_relu')(m)

bmo = models.Model(inputs=bmo_inputs_dict, outputs=m, name='building_features_model')
m.shape

# COMMAND ----------

weather_inputs_dict = {  # all values can be ints
    'weather_data': layers.Input(name='weather_data', shape=(None, len(train_gen.weather_features),), dtype=np.float16)
}
weather_inputs = list(weather_inputs_dict.values())
wm = layers.Concatenate(axis=1, name='weather_concat_layer')(weather_inputs)
wm = layers.Conv1D(
    filters=16, # reasonable range is 4..32
    kernel_size=4,
    padding='same',
    data_format='channels_last',
    name='first_1dconv', 
    activation='leaky_relu'
)(wm)
# wm = layers.Conv1D(
#     filters=4,
#     kernel_size=4,
#     padding='same',
#     data_format='channels_last',
#     name='third_1dconv', 
#     activation='leaky_relu'
# )(wm)
# sum the time dimension
wm = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(wm)

wmo = models.Model(inputs=weather_inputs_dict, outputs=wm, name='weather_features_model')
wm.shape

# COMMAND ----------

# combine building features and 
cm = layers.Concatenate(name='combine_features')([bmo.output, wmo.output])
cm = layers.Dense(16, activation='leaky_relu')(cm)
cm = layers.Dense(8, activation='leaky_relu')(cm)
# at this point, cm is a chokepoint representing embedding of a building + climate it is in

# would be a dict if these outputs were final
intermediate_outputs = []
for consumption_group in train_gen.consumption_groups:
    io = layers.Dense(4, name=consumption_group+'_entry', activation='leaky_relu')(cm)
    # ... feel free to add more layers
    # io = layers.Dense(8, name=consumption_group+'_mid', activation='leaky_relu')(io)
    # no activation on the output
    io = layers.Dense(1, name=consumption_group+'_out', activation='leaky_relu')(io)
    intermediate_outputs.append(io)

final_output = layers.Concatenate()(intermediate_outputs)

loss_fn = keras.losses.MeanAbsolutePercentageError(reduction="sum_over_batch_size")  # use 'sum' instead?

final_model = models.Model(inputs=[bmo.input, wmo.input], outputs={'outputs': final_output})
# loss_fn = keras.losses.MeanAbsolutePercentageError()
final_model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='adam')

# COMMAND ----------

# cold cache, first epoch: 
#   ~35m on office wifi for 800 buildings (2.7s/building)
#   us-central-1: ~5min@900 bldg = 0.3s/building
# warm cache:
#   7 seconds per epoch on a 800 buildings generator, on a thinkpad x1 laptop
#   ~3 minutes per epoch on DataBricks GPU, wth?
history = final_model.fit(
    train_gen, epochs=50, validation_data=test_gen, 
    callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
    verbose=False
)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# COMMAND ----------




# COMMAND ----------

!ls -l .cache/
