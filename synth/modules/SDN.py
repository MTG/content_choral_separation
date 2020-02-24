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


# def content_encoder_stft(inputs, is_train):
#     inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])
#     encoded = inputs
#     for i in range(config.encoder_layers):
#         encoded = encoder_conv_block_stft(encoded, i, is_train)

#     emb = tf.squeeze(encoded)

#     return emb

# def encoder_conv_block_stft(inputs, layer_num, is_train, num_filters = config.filters):

#     if layer_num<(config.encoder_layers - 1):

#         output = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs, num_filters * 2**int(2 + layer_num/config.augment_filters_every), (config.filter_len,1)
#             , strides=(2,1),  padding = 'same', name = "Enc_"+str(layer_num)), training = is_train, name = "Encoder_BN"+str(layer_num)))
#     else:
#         output = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/config.augment_filters_every), (config.filter_len,1)
#             , strides=(2,1),  padding = 'same', name = "Enc_"+str(layer_num)), training = is_train, name = "Encoder_BN"+str(layer_num)))
        
#     return output

def bi_static_stacked_RNN(x, scope='RNN', lstm_size = config.autovc_lstm_size):
    """
    Input and output in batch major format
    """
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        output = x
        num_layer = 2
        # for n in range(num_layer):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)

        _initial_state_fw = lstm_fw.zero_state(config.batch_size, tf.float32)
        _initial_state_bw = lstm_bw.zero_state(config.batch_size, tf.float32)

        output, _state1, _state2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, output, 
                                                  initial_state_fw=_initial_state_fw,
                                                  initial_state_bw=_initial_state_bw, 
                                                  scope='BLSTM_'+scope)
        output = tf.stack(output)
        output_fw = output[0]
        output_bw = output[1]
        output = tf.transpose(output, [1,0,2])


        # output = tf.layers.dense(output, config.output_features, activation=tf.nn.relu) # Remove this to use cbhg

        return output


def content_encoder_stft(inputs, is_train):


    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "conv_1",padding='same'), training = is_train, name = "conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "conv_2",padding='same'), training = is_train, name = "conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "conv_3",padding='same'), training = is_train, name = "conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op = bi_static_stacked_RNN(conv_3, scope = "Encode")

    lstm_fow = lstm_op[:,:, :config.autovc_lstm_size]

    lstm_back = lstm_op[:,:, config.autovc_lstm_size:]

    emb = []

    for i in range(int(config.max_phr_len/config.autovc_code_sam)):
        emb.append(tf.concat([lstm_fow[:, i*config.autovc_code_sam,:], lstm_back[:, (i+1)*config.autovc_code_sam-1, :]], axis = -1))
    emb = tf.stack(emb)

    embo = tf.tile(tf.reshape(emb[0],[config.batch_size,1,-1]),[1,config.autovc_code_sam,1])

    for i in range(1, int(config.max_phr_len/config.autovc_code_sam)):
        embs = tf.tile(tf.reshape(emb[i],[config.batch_size,1,-1]),[1,config.autovc_code_sam,1])

        embo = tf.concat([embo, embs], axis = 1)


    emb = bi_static_stacked_RNN(tf.squeeze(embo), scope = "Encode_emb")

    return emb


def enc_dec_f0(inputs, harm, ap, is_train):

    inputs = tf.concat([inputs, harm, ap], axis=-1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train,name = 'S_in_BN')

    output = encoder_decoder_archi(inputs, is_train)

    f0 = tf.layers.dense(output, 1, activation=None, name = "wave_f0_2")

    return tf.reshape(f0, [config.batch_size, config.max_phr_len , -1])

def enc_dec_vuv(inputs, harm, ap,vuv, is_train):

    inputs = tf.concat([inputs, harm, ap, vuv], axis=-1)

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
