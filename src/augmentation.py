from .config import config
import os
from PIL import Image
from PIL import ImageEnhance
import random


class DataAugmentation():
    """
        Class performing data augmentation on images by performing mirroring, random rotation image for random angle [-30, 30],
        brightness and contrast enchane with radnom factor from [0.5, 1]
    """
    def __init__(self, loader):
        self.loader = loader
        self.loader.reset_image_index()
        print ("Starting data augmentation. ")

    def perform_augmentation(self):
        if not os.path.exists(config.TRAIN_AUGMENTATION_FOLDER):
            os.makedirs(config.TRAIN_AUGMENTATION_FOLDER)

        for _ in range(self.loader.size):
            augmented_images = []
            image, image_name, _ = self.loader.next_image()
            augmented_images.append(image)
            augmented_images.append(self._mirror_image(image))
            augmented_images.append(self._random_brightness(image))
            augmented_images.append(self._random_contrast(image))
            augmented_images.append(self._random_rotation(image))

            image_counter = 1
            for img in augmented_images:
                name, extenstion = image_name.rsplit('.')
                temp = name + '_' + str(image_counter) + '.' + extenstion
                path = os.path.join(config.TRAIN_AUGMENTATION_FOLDER, temp)
                self._save_image(img, path)
                image_counter += 1
        print ("Data augmentation is done.")

    def _save_image(self, image, image_path):
        """ Saving image in jpeg format using PIL library. """
        image.save(image_path, 'jpeg')

    def _random_brightness(self, image):
        """ Perform brightness change with random factor [0.5, 1]. An enhancement factor of 0.0 gives a black image
            and a factor of 1.0 gives the original image.
        """
        brightness_factor = random.uniform(0.5, 1.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(brightness_factor)

    def _random_contrast(self, image):
        """ Perform contrast change with random factor [0.5, 1]. An enhancement factor of 0.0 gives a solid grey image
            and a factor of 1.0 gives the original image.
        """
        contrast_factor = random.uniform(0.5, 1.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(contrast_factor)

    def _random_rotation(self, image):
        """ Rotates image by angle from range [-30, 30] keeping original image size. """
        angle = random.randint(-30, 30)
        return image.rotate(angle)

    def _mirror_image(self, image):
        """ Mirrors image left to right. """
        return image.transpose(Image.FLIP_LEFT_RIGHT)
