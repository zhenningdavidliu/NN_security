from Data_loader import Data_loader
import numpy as np
import random
from antialliasing import rotated_stripe, calc_dist

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
        super(Data_loader_lines, self).__init__(arguments)
        required_keys = ['images','labels']

        self._check_for_valid_arguments(required_keys, arguments)
        self.images = arguments['images']
        self.labels = arguments['labels']

    def load_data(self):
        data = np.load(self.images)
        label = np.load(self.labels)
        
        return data, label 

