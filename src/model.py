#! /usr/bin/env python3

import keras
from keras import layers, models
import numpy as np
import pandas as pd
import tensorflow as tf

import datagen

def create_dataset(batch_size=64, train_test_split=0.8):
    get_building_metadata = datagen.BuildingMetadataBuilder()
    building_ids = get_building_metadata.building_ids
    datagen_params = {
        # 'building_features': ...,
        'metadata_builder': get_building_metadata,
        'batch_size': batch_size,
    }
    np.random.shuffle(building_ids)
    train_buildings, test_buildings = datagen.train_test_split(
        building_ids, left_size=train_test_split)

    train_gen = datagen.DataGen(train_buildings, **datagen_params)
    test_gen = datagen.DataGen(test_buildings, **datagen_params)

    return train_gen, test_gen


def main():
    """ End to end model architecture definition """
    train_gen, test_gen = create_dataset()

    # Building model
    bmo_inputs_dict = {  # all values can be ints
        'building_features': layers.Input(
            name='building_features',
            shape=(len(train_gen.building_features),),
            dtype=np.float16
        )
    }
    bmo_inputs = list(bmo_inputs_dict.values())
    bm = layers.Concatenate(name='concat_layer')(bmo_inputs)
    bm = layers.Dense(32, name='second_dense', activation='leaky_relu')(bm)
    bm = layers.Dense(8, name='third_dense', activation='leaky_relu')(bm)

    bmo = models.Model(inputs=bmo_inputs_dict, outputs=bm, name='building_model')

    # Weather data model
    weather_inputs_dict = {  # all values can be ints
        'weather_data': layers.Input(
            name='weather_data',
            shape=(None, len(train_gen.weather_features),),
            dtype=np.float16
        )
    }
    weather_inputs = list(weather_inputs_dict.values())
    wm = layers.Concatenate(axis=1, name='weather_concat_layer')(weather_inputs)
    wm = layers.Conv1D(
        filters=16,
        kernel_size=8,
        padding='same',
        data_format='channels_last',
        name='first_1dconv',
        activation='leaky_relu'
    )(wm)
    wm = layers.Conv1D(
        filters=8,
        kernel_size=8,
        padding='same',
        data_format='channels_last',
        name='second_1dconv',
        activation='leaky_relu'
    )(wm)
    # sum the time dimension
    wm = layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(wm)

    wmo = models.Model(inputs=weather_inputs_dict, outputs=wm, name='weather_model')

    # Combined model and separate towers for output groups
    cm = layers.Concatenate(name='combine_features')([bmo.output, wmo.output])
    cm = layers.Dense(16, activation='leaky_relu')(cm)
    cm = layers.Dense(16, activation='leaky_relu')(cm)
    # cm is a chokepoint representing embedding of a building + climate it is in

    # building a separate tower for each output group
    intermediate_outputs = []
    for consumption_group in train_gen.consumption_groups:
        io = layers.Dense(8, name=consumption_group+'_entry', activation='leaky_relu')(cm)
        # ... feel free to add more layers
        # no activation on the output
        io = layers.Dense(1, name=consumption_group+'_out', activation='leaky_relu')(io)
        intermediate_outputs.append(io)

    final_output = layers.Concatenate()(intermediate_outputs)

    final_model = models.Model(inputs=[bmo.input, wmo.input], outputs={'outputs': final_output})

    loss_fn = keras.losses.MeanAbsolutePercentageError(reduction="sum_over_batch_size")
    final_model.compile(loss=loss_fn, optimizer='adam')

