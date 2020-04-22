from .Data_loader import Data_loader
import numpy as np
import random

class Data_loader_shades2(Data_loader):
    """
This experiment is creating squares collored in different shades of grey and checking how humans and AI compare in both
accuracy and confidence estimate. We set the squares to have a fixed length and all squares have to be fully withing the
image. Further we assing a "shade_contrast" value which tells us how different the shades of grey should be.

Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)
side_length (int): Size of squares (side_length x side_length)
shade_contrast (int): the minimal distance between two colors of grey

Methods
-------
load_data(): Loads the training data.


Example arguments
-----------------
number_of_samples: 2000
grid_size: 64
side_length: 4
shade_contrast: 0.1
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_shades2, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'shade_contrast']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.shade_contrast = arguments['shade_contrast']
    def load_data(self):
        # Two squares one on the left one on the right, different shades
        # task is to say which side is lighter
        data, label, diff = self._generate_set()

        return data, label, diff 
       
    def _generate_set(self, shuffle= True):
           
        n = self.number_of_samples
        l = self.side_length # side length of the square inside the image
        a = self.grid_size # Size of image
        e = self.shade_contrast

        data = np.ones([n,a,a])
        label = np.zeros([n,1])
        diff = np.zeros(n)
        for i in range(n):

            i1 = np.random.randint(a-l) 
            j1 = np.random.randint(a-l) 
        
            i2 = np.random.randint(a-2*l) 
            j2 = np.random.randint(a-2*l) 

            shade1 = np.random.normal(0.4,0.2432)
            shade2 = np.random.normal(0.4,0.2432)
           
            while (((i2-i1)<l) and ((j2-j1)<l)) or (((i1-i2)<2*l) and ((j1-j2)<2*l)) :

                i1 = np.random.randint(a-l) 
                j1 = np.random.randint(a-l) 
            
                i2 = np.random.randint(a-2*l) 
                j2 = np.random.randint(a-2*l) 



            # Ensure all colours lie in the interval [0, 1].
            if shade1 < 0:
                shade1 = 0
            if shade1 > 0.8:
                shade1 = 0.8
            if shade2 < 0:
                shade2 = 0
            if shade2 > 0.8:
                shade2 = 0.8
            
            if (abs(shade2-shade1) < e):
                shade2 = shade1 + e
            
            if shade2>0.8:
                shade1 = shade1 - (shade2 - 0.8)
                shade2 = 0.8

            for j in range(i1,i1+l):
                for k in range(j1,j1+l):
                    data[i,j,k]=shade1
            for j in range(i2,i2+2*l):
                for k in range(j2,j2+2*l):
                    data[i,j,k]=shade2
            diff[i] = shade1 - shade2
            
            if shade1>shade2:
                label[i] = 1 # Left is brighter

        data = np.expand_dims(data, axis = 3)

        return data, label, diff



         


                
           
