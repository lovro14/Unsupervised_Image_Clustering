from .abstract_model import AbstractModel
from .data_loader import DataLoader
from .config import config
import tensorflow as tf
import numpy as np
from .batch_loader  import BatchLoader
from .autoencoder import Autoencoder
import os
from PIL import Image


class CustomModelMixed9(AbstractModel):

    def __init__(self, incetion_model_path, autoencoder_model_path=None):
        super(self.__class__, self).__init__(incetion_model_path)
        self.input_tensor_name = 'DecodeJpeg/contents:0'
        self.output_tensor_name = 'mixed_9/join:0'
        self.image_names = []
        self.image_embeddings = {}
        self.batch_size=2
        self.autoencoder_model_path=autoencoder_model_path
        self.batch_loader = None


    def _extract_inception_tensors(self, augmentation_flag=False):
        with tf.Session(graph=self.graph) as sess:
            if augmentation_flag:
                loader = DataLoader(config.TRAIN_AUGMENTATION_FOLDER)
            else:
                loader = DataLoader(config.TRAIN_FOLDER)

            # create summary writer
            test_writer = tf.summary.FileWriter(config.TENSORBOARD_PATH, sess.graph)

            output_tensors = None

            for image_index in range(loader.size):

                image, img_name, img_path = loader.next_image()
                self.image_names.append(img_name)

                # read image data
                image_data = tf.gfile.FastGFile(img_path, 'rb').read()

                # run forward pass
                output_tensor_value = sess.run(self.output_tensor_name, {self.input_tensor_name: image_data})
                dimensions = output_tensor_value.shape

                # save embedding
                if output_tensors == None:
                    output_tensors = np.zeros((loader.size, dimensions[1], dimensions[2], dimensions[3]), dtype='float32')
                output_tensors[image_index] = output_tensor_value

            print( " Type clustering data ", type(output_tensors))
            print ("Number of images ", output_tensors.shape)
            self.output_tensors = output_tensors


    def _autoencoder_graph(self):
        dimensions=(self.output_tensors.shape)
        ae_input_tensor = tf.placeholder(tf.float32, (None,dimensions[1],dimensions[2],dimensions[3]), name='input_tensor')

        # create autoencoder
        ae_output_tensor, ae_embedding_tensor = Autoencoder(ae_input_tensor)

        tf.add_to_collection("ae_input_tensor", ae_input_tensor)
        tf.add_to_collection("ae_output_tensor", ae_output_tensor)
        tf.add_to_collection("ae_embedding_tensor", ae_embedding_tensor)

        return {'ae_input_tensor': ae_input_tensor,
                'ae_output_tensor': ae_output_tensor,
                'ae_embedding_tensor': ae_embedding_tensor}


    def train(self, sess):
        self.autoencoder = self._autoencoder_graph()

        # %%
        loss=tf.reduce_mean(tf.square(self.autoencoder["ae_output_tensor"] - self.autoencoder["ae_input_tensor"]))
        learning_rate = 0.01
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        saver = tf.train.Saver()
        print('Initializing vars')
        sess.run(tf.global_variables_initializer())

        # Create summry writer for Tensorboard
        writer = tf.summary.FileWriter("log/dir", sess.graph)

        # Log cost, input image, reconstructed image and
        tf.summary.scalar("loss", loss)
        #tf.summary.image("input image", tf.reshape(ae['x'], [self.batch_size, 128, 128, 3]), max_outputs=16)
        #tf.summary.image("reconstructed image", tf.reshape(ae['y'], [self.batch_size, 128, 128, 3]), max_outputs=16)
        summary_op = tf.summary.merge_all()
        self.batch_loader = BatchLoader(self.output_tensors)
        # Fit all training data
        n_epochs = 2
        for epoch_i in range(n_epochs):
            for batch_i in range(self.batch_loader.size // self.batch_size):
                train_batch = self.batch_loader.next_batch(self.batch_size)
                sess.run(optimizer, feed_dict={self.autoencoder['ae_input_tensor']: train_batch})
                # Get cost, and summary merged together
                c, summary_str = sess.run([loss, summary_op], feed_dict={self.autoencoder['ae_input_tensor']: train_batch})
            print("**** Epoch: {}  Cost: {} *****".format(epoch_i, c))

            # save model
            if epoch_i % 10 == 0 and epoch_i > 10:
                model_path = os.path.join("models/autoencoder_" + str(epoch_i))
                saver.save(sess, model_path)
                print('Model saved to : %s' % model_path)

            # Write to tensorboard summary
            writer.add_summary(summary_str, epoch_i)
            writer.flush()



    def extract_embeddings(self, augmentation_flag=False):
        self._extract_inception_tensors(augmentation_flag)

        # We create a session to use the graph
        sess = tf.Session()

        # create saver
        try:
            print("Import meta graph")
            saver = tf.train.import_meta_graph(self.autoencoder_model_path + ".meta")
            print("Loading saved model")
            saver.restore(sess, self.autoencoder_model_path)
            print("Model loaded.")
        except:
            print("Model not loaded, training new autoencoder")
            self.train(sess)

        self.batch_loader.reset_batch_index()

        # get embedding tensor node
        embedding_tensor = tf.get_collection("ae_embedding_tensor")[0]
        embedding_tensor_index = 0
        for batch_i in range(self.batch_loader.size // self.batch_size):
            train_batch = self.batch_loader.next_batch(self.batch_size)

            ebmedding_tensor_values = sess.run(embedding_tensor, feed_dict={self.autoencoder['ae_input_tensor']: train_batch})

            for embedding_tensor_value in ebmedding_tensor_values:
                 # flatten 4D tensor
                embedding = embedding_tensor_value.flatten()

                self.image_embeddings[self.image_names[embedding_tensor_index]] = embedding
                embedding_tensor_index += 1

        return self.image_embeddings

