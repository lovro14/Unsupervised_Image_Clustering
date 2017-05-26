from .abstract_model import AbstractModel
from .data_loader import DataLoader
from .config import config
import tensorflow as tf
import numpy as np


class InceptionModel(AbstractModel):

    def __init__(self, model_path, image_width=299, image_height=299):
        super(self.__class__, self).__init__(model_path)
        self.input_tensor_name = 'Mul:0'
        self.output_tensor_name = 'pool_3/_reshape:0'
        self.image_embeddings = {}
        self.image_names = []
        self.image_dimensions = [image_width, image_height]

    def extract_embeddings(self, augmentation_flag=False):
        with tf.Session(graph=self.graph) as sess:
            if augmentation_flag:
                loader = DataLoader(config.TRAIN_AUGMENTATION_FOLDER, self.image_dimensions)
            else:
                loader = DataLoader(config.TRAIN_FOLDER, self.image_dimensions)

            # create summary writer
            test_writer = tf.summary.FileWriter(config.TENSORBOARD_PATH, sess.graph)

            # iterate images
            for image_index in range(loader.size):

                image, img_name, img_path = loader.next_image()
                self.image_names.append(img_name)

                # run forward pass
                output_tensor_value = sess.run(self.output_tensor_name, {self.input_tensor_name: np.expand_dims(image,axis=0)})

                # flatten 4D tensor, latent representation 1*2048
                embedding = output_tensor_value.flatten()

                # store in dictionary image_name as key and embedding as value
                self.image_embeddings[self.image_names[image_index]] = embedding

                # save image info for visualization
                # image_names.append(img_name)
                # image = image.resize((config.EMB_IMAGE_HEIGHT, config.EMB_IMAGE_WIDTH), Image.ANTIALIAS)
                # images[image_index] = image.convert("RGB")

            # save embeddings for tensorboard visualization
            # create_summary_embeddings(sess, images, image_names, EMB, config.TENSORBOARD_PATH)

            return self.image_embeddings
