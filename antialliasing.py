import numpy as np
from numpy.linalg import norm
from math import cos, sin
import math
from PIL import Image
from shapely.geometry import Point, Polygon

def calc_dist(x_1, y_1, x2, y2, a, b):
    # provided a line given by (x_1, y_1), (x2, y2) we calculate the distance 
    # and hence the shade of a pixel at (a,b)

    p1 = np.array([x_1, y_1])
    p2 = np.array([x2, y2])
    p3 = np.array([a, b])
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    
    return d

def rotated_stripe(x, y, l, w, angle, grid_size):

    data = np.ones((grid_size, grid_size))
    
    x2 = x + int(round(l*cos(angle*math.pi/180)))
    y2 = y + int(round(l*sin(angle*math.pi/180)))

    x3 = x + int(round(w*cos((angle-90)*math.pi/180)))
    y3 = y + int(round(w*sin((angle-90)*math.pi/180)))

    x4 = x3 + int(round(l*cos(angle*math.pi/180)))
    y4 = y3 + int(round(l*sin(angle*math.pi/180)))


    for i in range(x, x2 + np.sign(x2-x), np.sign(x2-x)):
        for j in range(y, y2 + np.sign(y2-y), np.sign(y2-y)):
            if calc_dist(x, y, x2, y2, i, j) < 1:
                data[i,j] = calc_dist(x, y, x2, y2, i, j)
    
    for i in range(x, x3 + np.sign(x3-x), np.sign(x3-x)):
        for j in range(y, y3 + np.sign(y3-y), np.sign(y3-y)):
            if calc_dist(x, y, x3, y3, i, j) < 1:
                data[i,j] = calc_dist(x, y, x3, y3, i, j)

    for i in range(x2, x4 + np.sign(x4-x2), np.sign(x4-x2)):
        for j in range(y2, y4 + np.sign(y4-y2), np.sign(y4-y2)):
            if calc_dist(x4, y4, x2, y2, i, j) < 1:
                data[i,j] = calc_dist(x2, y2, x4, y4, i, j)

 
    for i in range(x3, x4 + np.sign(x4-x3), np.sign(x4-x3)):
        for j in range(y3, y4 + np.sign(y4-y3), np.sign(y4-y3)):
            if calc_dist(x4, y4, x3, y3, i, j) < 1:
                data[i,j] = calc_dist(x3, y3, x4, y4, i, j)
    
    min_x = min(x, x2, x3, x4)
    max_x = max(x, x2, x3, x4)
    min_y = min(y, y2, y3, y4)
    max_y = max(y, y2, y3, y4)

    polygon = Polygon([(x,y),(x2,y2), (x4,y4), (x3, y3)])

    for i in range(min_x, max_x+1):
        for j in range(min_y, max_y+1):
            if polygon.contains(Point(i,j)):
                data[i,j] = 0

    return data

if __name__ == "__main__" :
    
    img = rotated_stripe(40, 40, 50, 3, 27, 512)
    print(img)
    A_kron = np.ones([5,5])
    img = 255*img
    img = np.kron(img, A_kron)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.show()
