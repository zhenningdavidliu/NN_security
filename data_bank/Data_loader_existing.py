from .Data_loader import Data_loader
import numpy as np
import random

class Data_loader_existing(Data_loader):

    """
load data from a saved file
    

Attributes
----------
images (string): The name of the file of images to read from
labels (string): The name of the file of labels to read from
difference (string): The name of the file of differences to read from
model (int): The model number

Methods
-------
load_data(): Loads the training data.

Example arguments
-----------------

images: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_images"
labels: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_labels"
difference: "/mn/kadingir/vegardantun_000000/nobackup/NN_security_data/data/shades_train_difference"
model: 1

"""

    def __init__(self, arguments):
        super(Data_loader_existing, self).__init__(arguments)
        required_keys = ['images', 'labels', 'difference', 'model']

        self._check_for_valid_arguments(required_keys, arguments)
        self.images = arguments['images']
        self.labels = arguments['labels']
        self.difference = arguments['difference']
        self.model = arguments['model']

    def load_data(self):

        data, label, diff = self._generate_set()

        return data, label, diff

    def __str__(self):
        class_str = """ Existing data

Images : %s
Labels : %s
Difference: %s
Model Number: %f

""" % (self.images, self.labels, self.difference, self.model)

        return class_str

    def _generate_set(self, shuffle= True):

        images = self.images
        labels = self.labels
        diff = self.difference
        model = self.model

        image_link = images + str(model) + ".npy"
        labels_link = labels + str(model) + ".npy"
        diff_link = diff + str(model) + ".npy"

        data = np.load(image_link)
        labels_data = np.load(labels_link)
        diff_data = np.load(diff_link)

        return data, labels_data, diff_data
