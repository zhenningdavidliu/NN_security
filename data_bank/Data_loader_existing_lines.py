from .Data_loader import Data_loader
import numpy as np
import random
from .antialliasing import rotated_stripe, calc_dist
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
    
Example arguments
-----------------
name: load_lines
arguments:
    number_of_samples: 10000
    grid_size: 128
    side_length: 30
    width: 4
    shift: True
    color_shift: 0.01
    save: True
    images: data_train_images
    labels: data_train_labels
    difference: data_train_difference
-----------------

"""
 
    def __init__(self, arguments):
        super(Data_loader_existing_lines, self).__init__(arguments)
        required_keys = ['images','labels','difference']

        self._check_for_valid_arguments(required_keys, arguments)
        self.images = arguments['images']
        self.labels = arguments['labels']
        self.difference = arguments['difference']

    def load_data(self):

        images = self.images
        labels = self.labels
        difference = self.difference
        data = np.load(images)
        label = np.load(labels)
        difference = np.load(difference)
       
        #data = np.expand_dims(data, axis =3)
        
        return data, label, difference

def __str__(self):
        class_str = """stripe data testing
Images: %s 
Labels: %s 
Difference: %s 
""" % ( self.images, self.labels, self.difference)
        return class_str


