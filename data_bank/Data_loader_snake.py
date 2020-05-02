from .Data_loader import Data_loader
import numpy as np
import random
import yaml
import math
import os
from os.path import join

class Data_loader_snake(Data_loader):
    """

The experiment is about whether a line which starts from the upper left corner and ends in the lower right corner is 
connected or disconnected.

Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)
save (bool): Whether the generated images are saved to a text file for future usage.(Since it may take a while)
images (string): The name of the file into which the generated images are saved (if save == True).
labels (string): The name of the file into which the generated labels are saved (if save == True).
shift (bool): Whether we want a color perturbation.
color_shift (float): The value by which the images are changed by.

Methods
-------
load_data(): Loads the training data.

Example arguments
-----------------
number_of_samples: 10000
grid_size: 224
shift: True
color_shift: 0.01
save: True
images: data_train_images
labels: data_train_labels
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_snake, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'save','images','labels','difference']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.save = arguments['save']
        self.images = arguments['images']
        self.labels = arguments['labels']
        self.difference = arguments['difference']
        self.shift = arguments['shift']
        self.color_shift = arguments['color_shift']
    def load_data(self):
        data, label, diff = self._generate_set()
        
        return data, label, diff 

    def __str__(self):
        class_str = """lines data testing
Number of samples: %d 
Grid size: %d 
Shift: %s 
Color shift: %g
Save: %s 
Images: %s 
Labels: %s 
""" % (self.number_of_samples, self.grid_size, self.shift, self.color_shift, self.save, self.images, self.labels)
        return class_str


    def _generate_set(self, shuffle= True):
        
        n = self.number_of_samples
        L = self.grid_size
        shift = self.shift
        color_shift = self.color_shift

        data = np.ones([n,L,L])
        label = np.random.binomial(1,0.5,[n,1])
        diff = label
        for i in range(n):
            x = 0
            y = 0 # where the "snake" currently is
            data [:,x,y] = 0
            if  label[i] == 1:
                
                for j in range(2*L-1):
                    if x < (L-1):
                        if y < (L-1):
                            temp = np.random.binomial(1,0.5)
                            x = x + temp
                            y = y + 1 - temp
                            data[:,x,y] = 0
                        else: 
                            data[:,x:(L-1),L-1] = 0
                            break
                    else:
                        data[:,L-1,y:(L-1)] = 0
                        break
            else:

                hole = np.random.randint(0,2*L-1)
                for j in range(2*L-1):

                    if x < (L-1):
                        if y < (L-1):
                            temp = np.random.binomial(1,0.5)
                            x = x + temp
                            y = y + 1 - temp

                            if (j == hole) or (j == hole + 1): 
                                data[:,x,y] = 1
                            else:
                                data[:,x,y] = 0
                        else: 
                            data[:,x:(L-1),L-1] = 0
                            data[:,L-1 - (2*L-1-hole),L-1] = 1
                            data[:,L- (2*L - 1 - hole) ,L-1] = 1
                            break
                    else:
                        data[:,L-1,y:(L-1)] = 0
                        data[:,L-1,L-1-(2*L-1-hole)] = 1
                        data[:,L-1,L-(2*L-1-hole)] = 1 
                        break
        data = np.expand_dims(data, axis =3)
        image_link = self.images
        label_link = self.labels
        diff_link = self.difference

        if self.save == True :
            np.save(image_link,data)
            np.save(label_link,label)
            np.save(diff_link, diff)

        return data, label, diff
