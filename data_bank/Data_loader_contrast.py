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
        required_keys = ['number_of_samples','grid_size','shift','epsilon']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.shift = arguments['shift']
        self.epsilon = arguments['epsilon']
    def load_data(self):
        data, label, diff = self._generate_set()

        return data, label, diff

    def __str__(self):
        class_str = """Contrast data
Number of samples: %d
Grid size: %d
shift: %s
epsilon: %f
""" % (self.number_of_samples,self.grid_size, self.shift, self.epsilon)
        return class_str

    def _generate_set(self, shuffle= True):
           
        n = self.number_of_samples
        L = self.grid_size
        shift = self.shift
        epsilon = self.epsilon
        data = np.ones([n,L,L])
        label = np.zeros([n,1])
        diff = np.zeros(n)
        p1 = np.random.uniform(0,1,n)
        p2 = np.random.uniform(0,1,n)
        ones = np.ones([1,L])
        zeros = np.zeros([1,L])
        a1 = 1- (3/2)*abs(epsilon) - epsilon/2
        a2 = (3/2)*abs(epsilon) - epsilon/2
        b1 = 1- (3/2)*abs(epsilon) + epsilon/2
        b2 = (3/2)*abs(epsilon) + epsilon/2

        for i in range(n):
            left = np.array([0])
            right = np.array([0])
            
            while left.sum() == right.sum():
                
                left = np.random.choice(2,[int((L-4)/2),L],replace = True, p = [p1[i], 1-p1[i]]) 
                right = np.random.choice(2,[int((L-4)/2),L],replace =True, p = [p2[i], 1-p2[i]])

            
            if left.sum() > right.sum():

                label[i] = 1

                if shift == True:
                    
                    data[i] = np.transpose(np.concatenate([a1*left,a1*ones,a2*ones,a2*ones,a1*ones,a1*right]))
                else:
                    data[i] = np.transpose(np.concatenate([left,ones,zeros,zeros,ones,right]))
            else:

                if shift == True:

                    data[i] = np.transpose(np.concatenate([b1*left,b1*ones,b2*ones,b2*ones,b1*ones,b1*right]))
                else:
                    data[i] = np.transpose(np.concatenate([left,ones,zeros,zeros,ones,right]))


            diff[i] = (left.sum() - right.sum() + L*((L-4)/2))/(L*(L-4)) 

        data = np.expand_dims(data, axis = 3)

        return data, label, diff
