from .Data_loader import Data_loader
import numpy as np
import random

class Data_loader_shades3(Data_loader):
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
corners (list): List of integers


The corners parameter is a list of numbers from the set (1,2,3,4). Each of 
these numbers corresponds to a quarter of the square below. The small and large 
square will only be positioned in the squares listed in corners.
+---------------+-----------------+
|               |                 |
|       1       |        2        |
|               |                 |
|               |                 |
+---------------+-----------------+
|               |                 |
|       3       |        4        |
|               |                 |
|               |                 |
+---------------+-----------------+


Methods
-------
load_data(): Loads the training data.


Example arguments
-----------------
number_of_samples: 2000
grid_size: 64
side_length: 4
shade_contrast: 0.05
corners: [1, 2, 3]
-----------------

"""
    def __init__(self, arguments):
        super(Data_loader_shades3, self).__init__(arguments)
        required_keys = ['number_of_samples', 'grid_size', 'side_length', 'shade_contrast', 'corners']

        self._check_for_valid_arguments(required_keys, arguments)
        self.number_of_samples = arguments['number_of_samples']
        self.grid_size = arguments['grid_size']
        self.side_length = arguments['side_length']
        self.shade_contrast = arguments['shade_contrast']
        self.corners = arguments['corners']
        for i in range(len(self.corners)):
            self.corners[i] = int(self.corners[i])
    
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
corners: %s
""" % (self.number_of_samples, self.grid_size, self.side_length, 
       self.shade_contrast, self.corners)
        return class_str

    def _generate_set(self):
           
        n = self.number_of_samples
        l = self.side_length # side length of the square inside the image
        a = self.grid_size # Size of image
        ah = int(a/2);
        e = self.shade_contrast
        corners = self.corners

        max_colour_value = 0.9

        data = np.ones([n,a,a])
        label = np.zeros([n,1])
        diff = np.zeros(n)
        
        #is_in_corner1 = lambda row, col: ( 0 <= row < ah ) and ( 0 <= col < ah )
        #is_in_corner2 = lambda row, col: ( 0 <= row < ah ) and ( 0 <= col < ah )
        #is_in_corner3 = lambda row, col: ( 0 <= row < ah ) and ( 0 <= col < ah )
        #is_in_corner4 = lambda row, col: ( 0 <= row < ah ) and ( 0 <= col < ah )

        corner_large   = np.random.choice(np.array(corners), size=n, replace=True);
        corner_small = np.random.choice(np.array(corners), size=n, replace=True);

        row_small = np.random.randint(low=0, high=ah, size=n);
        col_small = np.random.randint(low=0, high=ah, size=n);
        row_large = np.random.randint(low=0, high=ah, size=n);
        col_large = np.random.randint(low=0, high=ah, size=n);

        shades_large   = np.random.uniform(low=0.0, high=max_colour_value, size=n);
        shades_small = np.random.uniform(low=0.0, high=max_colour_value, size=n);

        for i in range(n):

            # Set colour
            s_small = shades_small[i]
            s_large   = shades_large[i]
            if abs(s_small - s_large) < e:
                if s_small + e < max_colour_value:
                    s_small += e;
                else: 
                    s_small -= e;

            # Set label
            if s_large > s_small:
                label[i] = 1 # largeger is brighter
            diff[i] = (s_large - s_small + max_colour_value)/(2*max_colour_value)            

            # Set label
            i1 = row_small[i];
            i2 = col_small[i];

            j1 = row_large[i];
            j2 = col_large[i];
            
            # Ensure that none of the squares intersect
            if abs(i1-j1) <= 2*l and abs(i2-j2) <= 2*l and corner_small[i] == corner_large[i]:
                while abs(i1-j1)<=2*l and abs(i2-j2)<=2*l:

                    i1 = np.random.randint(ah) 
                    i2 = np.random.randint(ah) 
                    j1 = np.random.randint(ah) 
                    j2 = np.random.randint(ah) 

            ahmlm1 = ah - l - l 
            if corner_small[i] == 1:
                if i1 == 0:
                    i1 = 1
                if i2 == 0:
                    i2 = 1
            if corner_small[i] == 2:
                if i1 == 0:
                    i1 = 1
                if i2 >= ahmlm1:
                    i2 = ahmlm1
                i2 += ah
            if corner_small[i] == 3:
                if i1 >= ahmlm1:
                    i1 = ahmlm1
                i1 += ah
                if i2 == 0:
                    i2 = 1
            if corner_small[i] == 4:
                if i1 >= ahmlm1:
                    i1 = ahmlm1
                i1 += ah
                if i2 >= ahmlm1:
                    i2 = ahmlm1
                i2 += ah
            
            ahmlm1 = ah - l-l - l 
            if corner_large[i] == 1:
                if j1 == 0:
                    j1 = 1
                if j2 == 0:
                    j2 = 1
            if corner_large[i] == 2:
                if j1 == 0:
                    j1 = 1
                if j2 >= ahmlm1:
                    j2 = ahmlm1
                j2 += ah
            if corner_large[i] == 3:
                if j1 >= ahmlm1:
                    j1 = ahmlm1
                j1 += ah
                if j2 == 0:
                    j2 = 1
            if corner_small[i] == 4:
                if j1 >= ahmlm1:
                    j1 = ahmlm1
                j1 += ah
                if j2 >= ahmlm1:
                    j2 = ahmlm1
                j2 += ah
            
            data[i, i1:i1+l, i2:i2+l] = s_small;
            data[i, j1:j1+2*l, j2:j2+2*l] = s_large;
            #data[i, ah:ah+1, :] = 0;
            #data[i, :, ah:ah+1] = 0;


        data = np.expand_dims(data, axis = 3)

        return data, label, diff



         


                
           
