from src.config import config
from src.augmentation import DataAugmentation
from src.data_loader import DataLoader
import sys


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'augment':
        augmentation_flag = True
    else:
        augmentation_flag = False

    print "Folder to data : ", config.DATA_FOLDER
    loader = DataLoader(config.DATA_FOLDER)
    loader.prepare_images()

    if augmentation_flag:
        augment = DataAugmentation(loader)
        augment.perform_augmentation()
