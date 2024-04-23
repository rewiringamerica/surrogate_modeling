#! /usr/bin/env python3
import itertools
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models

import datagen


def create_dataset(datagen_params: Dict, train_test_split=0.9):
    get_building_metadata = datagen.BuildingMetadataBuilder()
    building_ids = get_building_metadata.building_ids
    np.random.shuffle(building_ids)
    train_buildings, test_buildings = datagen.train_test_split(
        building_ids, left_size=train_test_split)

    train_gen = datagen.DataGen(train_buildings, **datagen_params)
    test_gen = datagen.DataGen(test_buildings, **datagen_params)

    return train_gen, test_gen


def gaussian_activation(x):
    return K.exp(-K.pow(x, 2))


def replace_weather_with_embeddings(gen, weather_model):
    """ Replace weather features in `gen` with embeddings using `weather_model`

    Args:
        gen: (datagen.DataGen) an instance of a data generator class
    """
    # all weather feature dfs are built using the same index.
    # To be safe, bulletproofing this code
    sample_weather_feature = gen.weather_features[0]
    if sample_weather_feature == 'weather_embedding':
        return  # already transformed

    idx = gen.weather_cache[sample_weather_feature].index
    weather_embed_input = {
        weather_feature: gen.weather_cache[weather_feature].loc[idx].values
        for weather_feature in gen.weather_features
    }
    weather_embeddings = pd.DataFrame(weather_model.predict(weather_embed_input), index=idx)
    gen._weather_features = gen.weather_features
    gen.weather_features = ['weather_embedding']
    gen.weather_cache['weather_embedding'] = weather_embeddings


def create_building_model(train_gen, layer_params):
    bmo_inputs_dict = {
        building_feature: layers.Input(
            name=building_feature, shape=(1,),
            dtype=train_gen.feature_dtype(building_feature)
        )
        for building_feature in train_gen.building_features
    }

    # handle categorical, ordinal, etc. features.
    # Here it is detected by dtype; perhaps explicit feature list and handlers
    # would be better
    bmo_inputs = []
    for feature, layer in bmo_inputs_dict.items():
        if train_gen.feature_dtype(feature) == tf.string:
            encoder = layers.StringLookup(
                name=feature+'_encoder', output_mode='one_hot',
                dtype=layer_params['dtype']
            )
            encoder.adapt(train_gen.feature_vocab(feature))
            layer = encoder(layer)
        bmo_inputs.append(layer)

    m = layers.Concatenate(name='concat_layer', dtype=layer_params['dtype'])(bmo_inputs)

    m = layers.Dense(32, name='second_dense', **layer_params)(m)
    m = layers.Dense(8, name='third_dense', **layer_params)(m)
    # TODO: consider applying batchnorm
    # m = layers.BatchNormalization()(m)

    bmo = models.Model(
        inputs=bmo_inputs_dict, outputs=m, name='building_features_model')
    return bmo_inputs_dict, bmo


def create_weather_model(train_gen, layer_params):
    weather_inputs_dict = {
        weather_feature: layers.Input(
            name=weather_feature, shape=(None, 1,), dtype=layer_params['dtype'])
        for weather_feature in train_gen.weather_features
    }
    weather_inputs = list(weather_inputs_dict.values())

    wm = layers.Concatenate(
        axis=-1, name='weather_concat_layer', dtype=layer_params['dtype']
    )(weather_inputs)

    wm = layers.Conv1D(
        filters=16, # reasonable range is 4..32
        kernel_size=4,
        padding='same',
        data_format='channels_last',
        name='first_1dconv',
        **layer_params
    )(wm)
    # Performance with only one layer of CNN is abismal.
    # Use at least one more layer
    wm = layers.Conv1D(
        filters=16,
        kernel_size=4,
        padding='same',
        data_format='channels_last',
        name='last_1dconv',
        # activation=gaussian_activation,
        **layer_params
    )(wm)
    # sum the time dimension
    wm = layers.Lambda(
        lambda x: K.sum(x, axis=1), dtype=layer_params['dtype'])(wm)

    wmo = models.Model(
        inputs=weather_inputs_dict, outputs=wm, name='weather_features_model')
    return weather_inputs_dict, wmo


def create_combined_model(train_gen, bmo, wmo, layer_params):
    combined_inputs_dict = {
        'building_embedding': layers.Input(
            name='building_embedding', shape=(bmo.output.shape[1],),
            dtype=layer_params['dtype']),
        'weather_embedding': layers.Input(
            name='weather_embedding', shape=(wmo.output.shape[1],),
            dtype=layer_params['dtype']),
    }
    combined_inputs = list(combined_inputs_dict.values())
    cm = layers.Concatenate(name='combine_features')(combined_inputs)

    cm = layers.Dense(16, **layer_params)(cm)
    cm = layers.Dense(16, **layer_params)(cm)
    # cm is a chokepoint representing embedding of a building + climate it is in

    # would be a dict if these outputs were final
    combined_outputs = {}
    for consumption_group in train_gen.consumption_groups:
        io = layers.Dense(8, name=consumption_group+'_entry', **layer_params)(cm)
        # ... feel free to add more layers
        io = layers.Dense(8, name=consumption_group+'_mid', **layer_params)(io)
        # no activation on the output
        io = layers.Dense(1, name=consumption_group, **layer_params)(io)
        combined_outputs[consumption_group] = io

    combined_model = models.Model(
        inputs=combined_inputs_dict, outputs=combined_outputs,
        name='combined_model')
    return combined_inputs_dict, combined_model


def create_model(layer_params=None):
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
    layer_params = layer_params or {}
    train_gen, test_gen = create_dataset()

    # Building model
    bmo_inputs_dict, bmo = create_building_model(train_gen, layer_params)

    # Weather data model
    weather_inputs_dict, wmo = create_weather_model(train_gen, layer_params)

    # Combined model and separate towers for output groups
    combined_inputs_dict, combined_model = create_combined_model(
        train_gen, bmo, wmo, layer_params)

    building_embedding = bmo(bmo_inputs_dict)
    weather_embedding = wmo(weather_inputs_dict)
    combined_output = combined_model({
        'building_embedding': building_embedding,
        'weather_embedding': weather_embedding
    })

    final_model = models.Model(
        inputs=itertools.ChainMap(bmo_inputs_dict, weather_inputs_dict),
        outputs=combined_output
    )

    final_model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='adam')
    # return final_model

    history = final_model.fit(
        train_gen, epochs=100, validation_data=test_gen,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
    )

    # Experimental: fix weather embeddings and continue training
    # This leads to about 30x faster epochs, while still making progress in
    # predictions quality
    replace_weather_with_embeddings(train_gen, wmo)
    replace_weather_with_embeddings(test_gen, wmo)

    combined_output2 = combined_model({
        'building_embedding': building_embedding,
        'weather_embedding': combined_inputs_dict['weather_embedding']
    })

    final_model2 = models.Model(inputs=itertools.ChainMap(bmo_inputs_dict, {
        'weather_embedding': combined_inputs_dict['weather_embedding']
    }), outputs=combined_output2)
    final_model2.compile(loss=keras.losses.MeanAbsoluteError(),optimizer='adam')

    history2 = final_model2.fit(
        train_gen, epochs=200, validation_data=test_gen,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10)]
    )
    return final_model2, wmo


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def debug_scatterplots(gen, final_model):
    gt = np.empty((len(gen.ids), len(gen.consumption_groups)))
    for batch_num in range(len(gen)):
        _, batch_gt = gen[batch_num]
        batch_gt = batch_gt['outputs']
        gt[gen.batch_size * batch_num:gen.batch_size * batch_num + len(batch_gt)] = batch_gt.sum(axis=-1)
    predictions = final_model.predict(gen)['outputs']
    groups = {
        'gt': gt,
        'pred': predictions
    }
    df = pd.DataFrame({
        group + '_' + consumption : groups[group][:, colnum]
        for group, (colnum, consumption) in itertools.product(groups, enumerate(gen.consumption_groups))
    })
    for consumption_group in gen.consumption_groups:
        df.plot.scatter(*(group+'_'+consumption_group for group in groups))
    # consider checking df.corr()


def main():
    layer_params = {
        'activation': 'leaky_relu',
        'dtype': np.float32,
    }
    final_model = create_model(layer_params)
    model_architecture_img = keras.utils.plot_model(
        final_model, to_file="model.png", show_shapes=True, show_dtype=True,
        rankdir="TB", dpi=200, show_layer_activations=True,
    )

    get_building_metadata = datagen.BuildingMetadataBuilder()
    datagen_params = {
        'metadata_builder': get_building_metadata,
        'batch_size': 64,
        # 'consumption_groups': (
        #   'heating', 'cooling',
        #   'lighting', 'other',
        # ),
        'weather_features': (
            'temp_air', 'ghi', 'wind_speed',
            # 'weekend', 'hour',
            # 'relative_humidity', 'dni', 'diffuse_horizontal_illum',
            # 'wind_direction',
        ),
        'building_features': (
            'sqft', 'bedrooms', 'stories', 'occupants', 'age2000', 'county',
            'infiltration_ach50', 'insulation_wall', 'insulation_ceiling_roof',
            'cooling_efficiency_eer', 'heating_efficiency',

            'cooling_setpoint', 'heating_setpoint',
            # 'insulation_slab', 'insulation_rim_joist', 'insulation_floor',
            # 'orientation', 'window_area',
            # 'lighting_efficiency',


            # categorical
            # 'foundation_type', 'windows_type',
        ),
        'dtype': layer_params['dtype'],
        # 'time_granularity': 'M',
    }

    train_gen, test_gen = create_dataset(datagen_params)
    history = final_model.fit(
        train_gen, epochs=100, validation_data=test_gen,
        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
    )
    plot_history(history)
    debug_scatterplots(test_gen, final_model)

    return history
