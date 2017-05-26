from PIL import Image
import numpy as np
from .config import config
import os


class DataLoader(object):

    def __init__(self, dataset_path, data_dimensions, resize_image_flag=True):
        """
            Initialize file reader with implemented next_batch() function for tensorflow
                :param file_list: list of files to read - filepaths
                :param resize_image_flag: True if resizing is neccessary
                :param resize_size: Desired output image size
        """
        print ("Initializing Data Loader... ")
        self._dataset_path = dataset_path
        self._files = self._read_files(dataset_path)
        self._num_examples = len(self._files)
        self._resize_image_flag = resize_image_flag
        self._resize_height, self._resize_width = data_dimensions[0], data_dimensions[1]
        self.batch_offset = 0
        self.curr_image_index = 0
        self._read_images()
        print ("Number of images loaded: ", self._num_examples)

    def _read_images(self):
        self.images = np.array([self._read_image_file(filename) for filename in self._files])

    def _read_image_file(self, filename):
        image = Image.open(filename, 'r').convert('RGB')
        if self._resize_image_flag:
            image = image.resize((self._resize_width, self._resize_height), Image.ANTIALIAS)
        # [0,255] --> [0,1]
        img = (np.array(image, dtype='float32') / 127.5) - 1
        return img

    def _center_crop(self, x, crop_height, crop_width):
        h, w = x.shape[:2]
        j = int(round((h - crop_height) / 2.))
        i = int(round((w - crop_width) / 2.))
        return x[j:j + crop_height, i:i + crop_width]

    def reset_batch_index(self):
        self.batch_offset = 0

    @property
    def size(self):
        return self._num_examples

    def get_image_names(self):
        # get file names without path and extension
        return [os.path.basename(filename) for filename in self._files]

    def rest_image_index(self):
        self.curr_image_index = 0

    def next_image(self):
        index = self.curr_image_index
        self.curr_image_index += 1
        # reading last image in folder
        if self.curr_image_index >= self._num_examples:
            self.curr_image_index = 0

        image_path = self._files[index]
        image_name = os.path.basename(image_path)
        image = self._read_image_file(image_path)
        return image, image_name, image_path

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end]

    def _read_files(self, folder_path):
        """ Read files and store in list to use for reading images. """
        try:
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

            if len(file_paths) == 0:
                print("Error while loading data.")
                exit()
            return file_paths
        except:
            print("Error while loading data.")
            exit()
