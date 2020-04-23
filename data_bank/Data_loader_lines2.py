from .Data_loader import Data_loader
import numpy as np
import random
from antialliasing import rotated_stripe, calc_dist
import yaml
import math
from math import cos, sin
import os
from os.path import join

class Data_loader_lines2(Data_loader):
    """
This experiment is creating tilted lines and checking how humans and AI compare in both
accuracy and confidence estimate. 

Images with a horizontal or bertical stripe are generated. This stripe is located somewhere randomly in the image. Furthermore, if shift == True, then we introduce a color code. This color code is added if the stripe is horizontal, else it is subtracted from all pixels.

Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)
side_length (int): The lenfth of the stripe.
width (int): Width of the stripe.
save (bool): Whether the generated images are saved to a text file for future usage. (Since it may take a while)
images(string): The name of the file into which the generated images are saved (if save == True).
labels (string): The name of the file into which the generated labels are saved (if save == True).
shift (bool): Whether we want a color perturbation.
color_shift (float): The value by which the images are changed by

Methods
-------
load_data(): Loads the training data.

Example arguments
-----------------
number_of_samples: 10000
grid_size: 128
side_length: 30
width: 4
shift: True
color_shift: 0.01
save: True
images: data_train_images
labels: data_train_labels
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_lines2, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'width','save','images','labels']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.width= arguments['width']
        self.save = arguments['save']
        self.images = arguments['images']
        self.labels = arguments['labels']
        self.shift = arguments['shift']
        self.color_shift = arguments['color_shift']
    def load_data(self):
        data, label = self._generate_set()
        
        return data, label 

    def _generate_set(self):
        
        n = self.number_of_samples
        L = self.grid_size
        a = self.side_length
        w = self.width
        shift = self.shift
        color_shift = self.color_shift
        ep = np.ones([L,L])*color_shift
        data = np.ones([n,L,L])
        label = np.zeros([n,1])
        angle = np.zeros(n)

        for i in range(n):
            
            i1 = np.random.randint(L-a-7) +2
            j1 = np.random.randint(L-a-7) +2
            angle[i] = 90*np.random.binomial(1,0.5)
            data[i,:,:] = rotated_stripe(i1,j1, a, w, angle[i], L)
        
            if angle[i]<45:
                label[i] = 1
                if shift == True:
                    data[i] += ep
            elif shift == True:
                data[i] -= ep
                            
        data = np.expand_dims(data, axis =3)
        image_link = join("data",self.images)
        label_link = join("data",self.images)

        if self.save == True :
            np.save(image_link,data)
            np.save(label_link,label)

        return data, label


if __name__ =="__main__":

    with open("config_lines.yml") as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    dataset = Data_loader_lines(cgf["DATASET_TRAIN"]["arguments"])
    data,label = dataset.load_data()

