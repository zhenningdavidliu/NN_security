from .Data_loader import Data_loader
import numpy as np
import random

class Data_loader_contrast(Data_loader):
    """
The images generated will have the left side sepparated from the right side by a solid 
line of 2 pixels width. The task is to tell whether the left half has more black(or grey) 
pixels colored black than the right half.


Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): The size of the images generated

Methods
-------
load_data(): Loads the training data.


Example arguments
-----------------
number_of_samples: 2000
grid_size: L
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_contrast, self).__init__(arguments)
        required_keys = ['number_of_samples','grid_size']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
    def load_data(self):
        data, label, diff = self._generate_set()

        return data, label, diff

    def __str__(self):
        class_str = """Contrast data
Number of samples: %d
""" % (self.number_of_samples)
        return class_str

    def _generate_set(self, shuffle= True):
           
        n = self.number_of_samples
        L = self.grid_size
        data = np.ones([n,L,L])
        label = np.zeros([n,1])
        diff = np.zeros(n)
        p1 = np.random.uniform(0.2,0.8,n)
        p2 = np.random.uniform(0.2,0.8,n)
        ones = np.ones([1,L])
        zeros = np.zeros([1,L])
       
        for i in range(n):
            left = np.array([0])
            right = np.array([0])
            
            while left.sum() == right.sum():
                
                left = np.random.choice(2,[int((L-4)/2),L],p1[i]) 
                right = np.random.choice(2,[int((L-4)/2),L],p2[i])

            data[i] = np.concatenate([left,ones,zeros,zeros,ones,right])
            
            if left.sum() > right.sum():

                label[i] = 1

            diff[i] = (left.sum() - right.sum() + L*((L-4)/2))/(L*(L-4)) 

        data = np.expand_dims(data, axis = 3)

        return data, label, diff
