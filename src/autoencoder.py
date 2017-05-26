
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, convolution2d_transpose, xavier_initializer

import math

def Autoencoder(input_tensor):
    """
    Autoencoder with shared weights.
    :param input_image: Original image.
    :return: (output_image, embedding_tensor)
    """
    with tf.variable_scope('autoencoder'):
        pad = 'SAME'

        #####################
        ###    ENCODER    ###
        #####################

        #1,8,8,2048
        with tf.variable_scope('conv1'):
            out = convolution2d(inputs=input_tensor, num_outputs=2048, kernel_size=3, stride=1, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())
            print("Dimensions conv1: ", out.get_shape())


        with tf.variable_scope('conv2'):
            embedding_tensor = convolution2d(inputs=out, num_outputs=4192, kernel_size=3, stride=2, padding=pad, rate=1,
                             activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())
            print("Dimensions conv2: ", embedding_tensor.get_shape())


        #####################
        ###    DECODER    ###
        #####################


        with tf.variable_scope('conv2'):
            out = convolution2d_transpose(inputs=embedding_tensor, num_outputs=4192, kernel_size=3, stride=2, padding=pad,
                                        activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())
            print("Dimensions deconv2: ", out.get_shape())

        with tf.variable_scope('conv1'):
            output_tensor = convolution2d_transpose(inputs=out, num_outputs=2048, kernel_size=3, stride=1, padding=pad,
                                                   activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())
            print("Dimensions deconv1: ", output_tensor.get_shape())


        return output_tensor, embedding_tensor