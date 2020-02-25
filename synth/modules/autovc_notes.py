from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
from synth.config import config


tf.logging.set_verbosity(tf.logging.INFO)


def f0_model(inputs, notes, singer_id, is_train):

    singer_id = tf.tile(tf.reshape(singer_id,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, notes, singer_id], axis=-1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train,name = 'S_in_BN')

    output = encoder_decoder_archi(inputs, is_train)

    f0 = tf.layers.dense(output, 1, activation=None, name = "wave_f0_2")

    return tf.reshape(f0, [config.batch_size, config.max_phr_len , -1])

def vuv_model(inputs, notes,f0, is_train):

    inputs = tf.concat([inputs, notes, f0], axis=-1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train,name = 'S_in_BN')


    output = encoder_decoder_archi(inputs, is_train)

    vuv = tf.layers.dense(output, 1, activation=tf.nn.sigmoid, name = "wave_vuv_2")

    return tf.reshape(vuv, [config.batch_size, config.max_phr_len , -1])


def encoder_decoder_archi(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train)

    return decoded

def encoder_conv_block(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv