from Data_loader import Data_loader
import numpy as np
import random
from antialliasing import rotated_stripe, calc_dist
import os
from os.path import join
class Data_loader_existing_lines(Data_loader):
    """
This experiment is creating tilted lines and checking how humans and AI compare in both
accuracy and confidence estimate. 
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)

Methods
-------
load_data(): Loads the training data.
"""
    def __init__(self, arguments):
        super(Data_loader_existing_lines, self).__init__(arguments)
        required_keys = ['images','labels']

        self._check_for_valid_arguments(required_keys, arguments)
        self.images = arguments['images']
        self.labels = arguments['labels']

    def load_data(self):

        images = self.images + ".npy"
        labels = self.labels + ".npy"
        images = join("data",images)
        labels = join("data",labels)
        data = np.load(images)
        label = np.load(labels)
       
        data = np.expand_dims(data, axis =3)
        
        return data, label 

