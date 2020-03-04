from Data_loader import Data_loader
import numpy as np
import random
from antialliasing import rotated_stripe, calc_dist

class Data_loader_lines(Data_loader):
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
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'width']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.width= arguments['width']

    def load_data(self):
        data, label = self._generate_set()
        
        return data, label 

    def _generate_set(self, shuffle= True):
        
        n = self.number_of_samples
        L = self.grid_size
        a = self.side_length
        w = self.width

        data = np.ones([n,L,L])
        label = np.zeros([n,1])
        angle = np.random.normal(45,1,n)

        for i in range(n):

            i1 = np.random.randint(L-a)
            j1 = np.random.randint(L-a)

            if angle[i]<0:
                angle[i] = 0
            elif angle[i] > 90:
                angle[i] = 90
            elif angle[i] == 45:
                angle[i] = 45.0001

            data[i,:,:] = rotated_stripe(i1,j1, a, w, angle[i], L)
            
            if angle[i]<45:
                label[i] = 1

        data = np.expand_dims(data, axis =3)

        return data, label
