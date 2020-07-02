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


def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def bi_dynamic_stacked_RNN(x, input_lengths, scope='RNN'):
    with tf.variable_scope(scope):
    # x = tf.layers.dense(x, 128)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.autovc_lstm_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=config.autovc_lstm_size, state_is_tuple=True)

        outputs, _state1, state2  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[cell,cell2],
            cells_bw=[cell,cell2],
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

    return outputs

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



def RNN(x, scope='RNN', lstm_size=config.autovc_out_lstm_size):
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        outputs=tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

    return outputs



def content_encoder(inputs, singer_label, is_train):

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, singer_label], axis = -1)

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


    return emb



def decoder(emb, singer_label, is_train):

    embo = tf.tile(tf.reshape(emb[0],[config.batch_size,1,-1]),[1,config.autovc_code_sam,1])

    for i in range(1, int(config.max_phr_len/config.autovc_code_sam)):
        embs = tf.tile(tf.reshape(emb[i],[config.batch_size,1,-1]),[1,config.autovc_code_sam,1])

        embo = tf.concat([embo, embs], axis = 1)

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([embo, singer_label], axis = -1)

    

    lstm_op_1 = RNN(inputs, scope = "Decode_1")

    lstm_op_1 = tf.reshape(lstm_op_1, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(lstm_op_1, 512, (5,1), name = "op_conv_1",padding='same'), training = is_train, name = "op_conv_1_BN"))

    conv_2 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "op_conv_2",padding='same'), training = is_train, name = "op_conv_2_BN"))

    conv_3 = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "op_conv_3",padding='same'), training = is_train, name = "op_conv_3_BN"))

    conv_3 = tf.reshape(conv_3,[config.batch_size, config.max_phr_len , -1] )

    lstm_op_2 = RNN(conv_3, scope = "Decode_2")

    lstm_op_3 = RNN(lstm_op_2, scope = "Decode_3")

    output = tf.layers.conv1d(lstm_op_3, config.num_features, 1, name = "decode_output")


    return output


def post_net(inputs, is_train):

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    conv_1 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(inputs, 512, (5,1), name = "post_conv_1",padding='same'), training = is_train, name = "post_conv_1_BN"))

    conv_2 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_1, 512, (5,1), name = "post_conv_2",padding='same'), training = is_train, name = "post_conv_2_BN"))

    conv_3 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_2, 512, (5,1), name = "post_conv_3",padding='same'), training = is_train, name = "post_conv_3_BN"))   

    conv_4 = tf.nn.tanh(tf.layers.batch_normalization(tf.layers.conv2d(conv_3, 512, (5,1), name = "post_conv_4",padding='same'), training = is_train, name = "post_conv_4_BN"))

    output = tf.layers.conv2d(conv_4, config.num_features, (5,1), name = "post_conv_5",padding='same')

    return tf.squeeze(output)
