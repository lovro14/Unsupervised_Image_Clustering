from easydict import EasyDict as edict
import os

project_dir = os.path.join(os.getcwd())

__C = edict()
config = __C

# data config
__C.TRAIN_FOLDER = project_dir + '/data/dataset_10'
__C.TRAIN_AUGMENTATION_FOLDER = project_dir + '/data/dataset_augment_train'


# projection vsualization config
__C.TENSORBOARD_PATH = 'tensorboard/test'
__C.EMB_IMAGE_WIDTH = 300
__C.EMB_IMAGE_HEIGHT = 300
