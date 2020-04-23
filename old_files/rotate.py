from PIL import Image
import numpy as np
import random

grid_size = 256
w = 3
l = 10

data = np.ones((grid_size, grid_size), dtype=np.uint8)
x = random.randint(0, grid_size - w)
y = random.randint(0, grid_size - l)

data[x:(x+w), y:(y+l)] = 0
a = 30 # angle

image = Image.fromarray(data)
rotated = image.rotate(a)
image.show()

# print(data.sum())
