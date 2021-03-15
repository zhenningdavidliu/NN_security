from .Data_loader import Data_loader
import numpy as np
import random
import sys
import yaml

class Data_loader_shades2(Data_loader):
    """
This experiment is creating squares collored in different shades of grey and checking how humans and AI compare in both
accuracy and confidence estimate. We set the squares to have a fixed length and all squares have to be fully withing the
image. Further we assing a "shade_contrast" value which tells us how different the shades of grey should be.

Unlike shades experiment, shades2 has the option of adding some false structures to it. This is done by the parameter separate (e.g. separate=1 means that the darker square will be on the left half plane whereas the lighter is on the right halfplane)

We also allow the possibility for background noise according to Task 1. This is set by the boolean variable "noise".

Attributes
----------
number_of_samples (int): Number of images generated for the test
grid_size (int): Size of images (grid_size x grid_size)
side_length (int): Size of squares (side_length x side_length)
shade_contrast (float): the minimal distance between two colors of grey
separate (int) : either 0,1,2 depending on which mode we want. Mode 0 is neutral, i.e. there is no false structure. Modes 1,2 are when the darker color always occupy one half of the image (modes 1 and 2 are exact oposites so one puts them on the right, the other on the left) 
noise (bool): Whether it is Task 0 (False) or Task 1 (True)
save (bool): Whether the generated images are saved to a text file for future reference.
images (string): The name of the file into which the generated images are saved.
labels (string): The name of the file into which the generated labels are saved (if save == True).
difference (string): The name of the file into which the generated differences are saved.
model (string): name of the model 

Methods
-------
load_data(): Loads the training data.


Example arguments
-----------------
number_of_samples: 2000
grid_size: 64
side_length: 4
shade_contrast: 0.1
separate: 0
noise: False
save: True
images: data_train_images
labels: data_train_labels
difference: data_train_difference
model: 1
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_shades2, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'shade_contrast','separate', 'noise']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.shade_contrast = arguments['shade_contrast']
        self.separate = arguments['separate']
        self.noise = arguments['noise']
        self.save = arguments['save']
        self.images = arguments['images']
        self.labels = arguments['labels']
        self.difference = arguments['difference']
        self.model_number = arguments['model']        
    def load_data(self):
        # Two squares one on the left one on the right, different shades
        # task is to say which side is lighter
        data, label, diff = self._generate_set()

        return data, label, diff
       
    def __str__(self):
        class_str = """Shades2 data
Number of samples: %d
Grid size: %d
Side length: %d
Shade contrast: %g
separate: %d
Save: %s 
Images: %s 
Labels: %s 
Difference: %s 
Model : %s

""" % (self.number_of_samples, self.grid_size, self.side_length, self.shade_contrast, self.separate, self.save, self.images, self.labels, self.difference, self.model_number)

        return class_str


    def _generate_set(self, shuffle= True):
        
        separate = self.separate # only put big squares to one side for label 1, if 1, big on left side, if 2 otherwise and 0 no separation


        n = self.number_of_samples
        l = self.side_length # side length of the square inside the image
        a = self.grid_size # Size of image
        e = self.shade_contrast
        noise = self.noise

        noiseless_data = np.ones([n,a,a])
        data = np.ones([n,a,a])
        label = np.zeros([n,1])
        diff = np.zeros(n)
        model_number = self.model_number
        

        for i in range(n):
                
            shade1 = 0
            shade2 = 0

            while(abs(shade2-shade1)<e):
                
                shade1 = np.random.normal(0.4,0.2432)
                shade2 = np.random.normal(0.4,0.2432)
     
                if shade1 < 0:
                    shade1 = 0
                if shade1 > 0.8:
                    shade1 = 0.8
                if shade2 < 0:
                    shade2 = 0
                if shade2 > 0.8:
                    shade2 = 0.8
                
                """
                shade1 = np.random.uniform(0,0.8)
                shade2 = np.random.uniform(0,0.8)
                """
            if separate == 0:

                i1 = np.random.randint(a-l-1) 
                j1 = np.random.randint(a-l-1) 
            
                i2 = np.random.randint(a-2*l-1) 
                j2 = np.random.randint(a-2*l-1) 

                while (((i2-i1)<l) and ((j2-j1)<l)) or (((i1-i2)<2*l) and ((j1-j2)<2*l)) :

                    i1 = np.random.randint(a-l-1) 
                    j1 = np.random.randint(a-l-1) 
                
                    i2 = np.random.randint(a-2*l-1) 
                    j2 = np.random.randint(a-2*l-1) 

            elif separate == 1:
                
                if shade2 > shade1 :

                    i1 = np.random.randint(int(a/2)-l-1) + int(a/2) 
                    j1 = np.random.randint(a-l-1) 
                
                    i2 = np.random.randint(int(a/2)-2*l-1) 
                    j2 = np.random.randint(a-2*l-1) 

                else:

                    i1 = np.random.randint(int(a/2)-l-1) 
                    j1 = np.random.randint(a-l-1) 
                
                    i2 = np.random.randint(int(a/2)-2*l-1) + int(a/2)
                    j2 = np.random.randint(a-2*l-1) 
 
            elif separate == 2:

                if shade2>shade1:
                    i1 = np.random.randint(int(a/2)-l-1) 
                    j1 = np.random.randint(a-l-1) 
                
                    i2 = np.random.randint(int(a/2)-2*l-1) + int(a/2)
                    j2 = np.random.randint(a-2*l-1) 
                else:

                    i1 = np.random.randint(int(a/2)-l-1) + int(a/2) 
                    j1 = np.random.randint(a-l-1) 
                
                    i2 = np.random.randint(int(a/2)-2*l-1) 
                    j2 = np.random.randint(a-2*l-1) 


            else:
                print("No such separation number")
                break

            for j in range(i1,i1+l):
                for k in range(j1,j1+l):
                    data[i,j,k]=shade1
            for j in range(i2,i2+2*l):
                for k in range(j2,j2+2*l):
                    data[i,j,k]=shade2
            diff[i] = (shade1 - shade2 + 0.8)/1.6 # the 0.8 and 1.6 are for normalization 
            
            if shade1>shade2:
                label[i] = 1 # Left is brighter

            noiseless_data[i] = data[i]

            if noise:
                
                epsilon_noise = min(abs(shade1-shade2), 1-shade1, 1-shade2)
                data[i, :, :] += np.random.uniform(-epsilon_noise/4, epsilon_noise/4, (a, a)) 
                for j in range(a):
                    for k in range(a):
                        data[i, j, k] = max(min(data[i, j, k], 1),0) # THIS LINE IS TO ENSURE THE BACKROUND IS WHITE

        noiseless_data = np.expand_dims(noiseless_data, axis = 3)
        data = np.expand_dims(data, axis = 3)

        image_link = self.images
        label_link = self.labels
        diff_link = self.difference

        noiseless_image_link = image_link + 'noiseless' + str(model_number) + ".npy"
        image_link = image_link + str(model_number) + ".npy"
        label_link = label_link + str(model_number) +".npy"
        diff_link = diff_link + str(model_number) + ".npy"
        
        if self.save == True:
            np.save(noiseless_image_link, noiseless_data)
            np.save(image_link, data)
            np.save(label_link, label)
            np.save(diff_link, diff)

        return data, label, diff

    

         


                
           
