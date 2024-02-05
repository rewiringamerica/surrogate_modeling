#! /usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models

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


def gaussian_activation(x):
    return K.exp(-K.pow(x, 2))


def create_model(model_config=None):
    """ End to end model architecture definition

    Model config should include:
        - datagen config
            - upgrade ids
            - building features
            - weather features
            - consumption groups
            - level of aggregation
            - batch size
        - model architecture config
            - dense layer tower configuration for building features
                - a list of layers with number of nodes, activation, and regularization
            - CNN config for weather features
                - a list of CNN1D layers with number of filters
    """
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
        # (non-leaky) Relu on final output leads to the model getting stuck
        # in zero derivative regions for negative output
        io = layers.Dense(1, name=consumption_group+'_out', activation='leaky_relu')(io)
        intermediate_outputs.append(io)

    final_output = layers.Concatenate()(intermediate_outputs)

    final_model = models.Model(inputs=[bmo.input, wmo.input], outputs={'outputs': final_output})

    final_model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer='adam'
    )
    return final_model


def main():
    final_model = create_model()
    train_gen, test_gen = create_dataset()
    history = final_model.fit(
        train_gen, epochs=100, validation_data=test_gen,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
    )

    return history
