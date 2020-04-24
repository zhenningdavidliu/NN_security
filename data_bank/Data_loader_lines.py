from .Data_loader import Data_loader
import numpy as np
import random
from .antialliasing import rotated_stripe, calc_dist
import yaml
import math
from math import cos, sin
import os
from os.path import join

class Data_loader_lines(Data_loader):
    """
This experiment is creating tilted lines and checking how humans and AI compare in both
accuracy and confidence estimate.

Images with a stripe are generated. The stripe is located somewhere in the image and its
position is determined randomly. There is also a colorshift introduced here which if is set
to be extant (shift == True) then all images with angle < 45 are shifted by a value "color_shift"
and all images with an angle > 45 are shifted by "-color_shift".

Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)
side_length (int): The length of the stripe.
width (int): Width of the stripe.
save (bool): Whether the generated images are saved to a text file for future usage.(Since it may take a while)
images (string): The name of the file into which the generated images are saved (if save == True).
labels (string): The name of the file into which the generated labels are saved (if save == True).
difference (string): The name of the file into which the generated differences are saved.
shift (bool): Whether we want a color perturbation.
color_shift (float): The value by which the images are changed by.

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
difference: data_train_difference
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_lines, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'width','save','images','labels']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.width= arguments['width']
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
Side length: %d 
Width: %d 
Shift: %s 
Color shift: %g
Save: %s 
Images: %s 
Labels: %s 
Difference: %s 
""" % (self.number_of_samples, self.grid_size, self.side_length, self.width, self.shift, self.color_shift, self.save, self.images, self.labels, self.difference)
        return class_str


    def _generate_set(self, shuffle= True):
        
        n = self.number_of_samples
        L = self.grid_size
        a = self.side_length
        w = self.width
        shift = self.shift
        color_shift = self.color_shift

        data = np.ones([n,L,L])
        label = np.zeros([n,1])
        diff = np.zeros([n,1])
        angle = np.zeros(n)

        for i in range(n):
            
            sensible = 0
            while sensible==0:
                i1 = np.random.randint(L-a-7) +2
                j1 = np.random.randint(L-a-7) +2
                angle[i] = np.random.normal(45,18)
                if angle[i]<0:
                    angle[i] = 0
                    sensible = 1
                elif angle[i] > 90:
                    angle[i] = 90
                    sensible = 1
                elif (abs(int(round(a*cos(angle[i]*math.pi/180)))) == abs(int(round(a*sin(angle[i]*math.pi/180))))):
                    sensible = 0 
                else:
                    sensible = 1
            data[i,:,:] = rotated_stripe(i1,j1, a, w, angle[i], L)
            
            if angle[i]<45:
                label[i] = 1
            
                if shift == True:
                    data[i,:,:] += color_shift     
            elif shift == True:
                    data[i,:,:] -= color_shift
            
            diff[i] = 1 - diff[i]/90
            

        data = np.expand_dims(data, axis =3)
        image_link = self.images
        label_link = self.labels
        diff_link = self.difference

        if self.save == True :
            np.save(image_link,data)
            np.save(label_link,label)
            np.save(diff_link, diff)

        return data, label, diff
    '''
    def _generate_set2(self):

        n = self.number_of_samples
        L = self.grid_size
        a = self.side_length
        w = self.width

        data = np.zeros((n,L,L))
        label = np.zeros([n,1])
        angle = np.zeros(n)

        for i in range(n):
            
            sensible = 0
            while sensible==0:
                i1 = np.random.randint(L-a-7) + 2
                j1 = np.random.randint(L-a-7) + 2 
                angle[i] = np.random.normal(45,18)
                if angle[i]<0:
                    angle[i] = 0
                    sensible = 1
                elif angle[i] > 90:
                    angle[i] = 90
                    sensible = 1
                elif (abs(int(round(a*cos(angle[i]*math.pi/180)))) == abs(int(round(a*sin(angle[i]*math.pi/180))))):
                    sensible = 0 
                else:
                    sensible = 1
            data[i,:,:] += rotated_stripe(i1,j1, a, w, angle[i], L) 
            
            if angle[i]<45:
                label[i] = 1

        data = np.expand_dims(data, axis =3)
        image_link = join("data",self.images)
        label_link = join("data",self.images)
        if self.save == True :
            np.save(image_link,data)
            np.save(label_link,label)

        return data, label

    def _generate_set3(self):
        
        n = self.number_of_samples
        L = self.grid_size
        a = self.side_length
        w = self.width
        shift = self.shift
        epsilon = self.epsilon
        ep = np.ones([L,L])*epsilon
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
                
                    if epsilon > 0:
                        data[i] = data[i]*(1-epsilon)
                    else:
                        data[i] = data[i]*(1+epsilon) - epsilon 
               
            elif shift == True:
                data[i] -= ep
                
                if epsilon > 0 :
                    data[i] = epsilon + data[i]*(1-epsilon)
                else:
                    data[i] = data[i]*(1+epsilon)
                
        data = np.expand_dims(data, axis =3)
        image_link = join("data",self.images)
        label_link = join("data",self.images)

        if self.save == True :
            np.save(image_link,data)
            np.save(label_link,label)

        return data, label

    '''
if __name__ =="__main__":

    with open("config_lines.yml") as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    dataset = Data_loader_lines(cgf["DATASET_TRAIN"]["arguments"])
    data,label = dataset.load_data()

