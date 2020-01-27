
# install Pillow
# pip3 install --user Pillow
from PIL import Image
import numpy as np

N = 100;

X = np.zeros([N,N]);
X[int(N/2): , :] = 1;

X *= 255;

Y = X.astype(np.uint8);

color_mode = 'L'; # Let color mode to be 8-bit integers
im = Image.fromarray(Y,mode=color_mode);
im.save('example_image.png');












