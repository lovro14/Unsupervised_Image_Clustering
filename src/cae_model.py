from .data_loader import DataLoader
from .config import config
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import OrderedDict


class CAE(object):

    def __init__(self, autoencoder_model_path=None, image_width=200, image_height=200):
        self.autoencoder_model_path = autoencoder_model_path
        self.image_embeddings = OrderedDict()
        self.image_dimensions=[image_width,image_height]

    def extract_embeddings(self, augmentation_flag=False):
        if augmentation_flag:
            self.loader = DataLoader(config.TRAIN_AUGMENTATION_FOLDER, self.image_dimensions)
        else:
            self.loader = DataLoader(config.TRAIN_FOLDER, self.image_dimensions)

        ae = self.autoencoder([None, self.image_dimensions[0], self.image_dimensions[1], 3])
        cost = ae['cost']
        learning_rate = 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # create session using gpu
        #with tf.device("/gpu:0"):
        sess = tf.Session()
        batch_size = 128

        # create summry writer for Tensorboard
        writer = tf.summary.FileWriter("log/dir", sess.graph)

        #with tf.device("/cpu:0"):
        # create saver
        saver = tf.train.Saver()

        try:
            saver.restore(sess, self.autoencoder_model_path)
            print("Loading saved model")
            print("Model loaded.")
            n_epochs = 0
        except:
            print('Initializing vars')
            sess.run(tf.global_variables_initializer())
            n_epochs = 30

        # log cost, input image and reconstructed image
        tf.summary.scalar("cost", cost)
        tf.summary.image("input image", tf.reshape(ae['x'],
                                                   [batch_size, self.image_dimensions[0], self.image_dimensions[1], 3]),
                                                    max_outputs=16)
        tf.summary.image("reconstructed image", tf.reshape(ae['y'],
                                                           [batch_size, self.image_dimensions[0], self.image_dimensions[1], 3]),
                                                            max_outputs=16)
        summary_op = tf.summary.merge_all()

        # training process
        print ("Training model for epochs: ", n_epochs)
        for epoch_i in range(n_epochs):
            for batch_i in range(self.loader.size // batch_size):
                train = self.loader.next_batch(batch_size)
                sess.run(optimizer, feed_dict={ae['x']: train})
                # Get cost, and summary merged together
                c, summary_str = sess.run([cost, summary_op], feed_dict={ae['x']: train})
            print("**** Epoch: {}  Cost: {} *****".format(epoch_i, c))

            # save model
            if (epoch_i+1) % 10 == 0 and epoch_i > 1:
                model_path = os.path.join("models/autoencoder_" + str(epoch_i))
                saver.save(sess, model_path)
                print('Model saved to : %s' % model_path)

            # write to tensorboard summary
            writer.add_summary(summary_str, epoch_i)
            writer.flush()

        # plot reconstructions
        # n_examples = 10
        # test_xs = self.loader.next_batch(n_examples)
        # recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
        # fig, axs = plt.subplots(2, n_examples, figsize=(200, 10))
        # for example_i in range(n_examples):
        #     # To show image convert np.float32  --> np.uint8 [0,256]
        #     axs[0][example_i].imshow(np.array(np.reshape((test_xs[example_i, :] + 1) * 127.5, (128, 128, 3)), dtype=np.uint8))
        #     axs[1][example_i].imshow(np.array(np.reshape((recon[example_i, :] + 1) * 127.5, (128, 128, 3)), dtype=np.uint8))
        # fig.show()
        # plt.draw()
        # plt.waitforbuttonpress()

        # finished with training extract embeddings
        # load just 7k images
        self.loader = DataLoader(config.TRAIN_FOLDER, self.image_dimensions)
        image_names = []

        for image_index in range(self.loader.size):
            image, img_name, img_path = self.loader.next_image()
            image_names.append(img_name)

            output_tensor_value = sess.run(ae["z"], feed_dict={ae['x']: np.expand_dims(image,axis=0)})
            # flatten 4d tensor
            embedding = output_tensor_value.flatten()

            self.image_embeddings[image_names[image_index]] = embedding

        return self.image_embeddings

    def lrelu(self, x, leak=0.2, name="lrelu"):
        """Leaky rectifier. """
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def autoencoder(self, input_shape,
                    n_filters=[3, 8, 16, 32],
                    filter_sizes=[3, 3, 3, 3],
                    corruption=False):
        """Build a deep autoencoder. """
        # input to the network
        x = tf.placeholder(tf.float32, input_shape, name='x')

        # ensure 2-d is converted to square tensor.
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, n_filters[0]])
        elif len(x.get_shape()) == 4:
            x_tensor = x
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor

        # build the encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = self.lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            print ("Layer: " + str(layer_i) + ", output_shape is : " + str(output.get_shape()))
            current_input = output

        # store the latent representation
        z = current_input
        encoder.reverse()
        shapes.reverse()

        # build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = self.lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            print ("DeLayer: " + str(layer_i) + ", output_shape is : " + str(shape))
            current_input = output

        # reconstruction output
        y = current_input
        # cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - x_tensor))
        return {'x': x, 'z': z, 'y': y, 'cost': cost}
