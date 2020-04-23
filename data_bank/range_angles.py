from antialliasing import rotated_stripe, calc_dist
import numpy as np
import os 
from os import join
if __name__=="__main__":
    data = np.ones([30,128,128])
    label = np.zeros([30,1])
    for i in range(30):
        data[i,:,:] += rotated_stripe(50,50,40,5,i*90/29,128) -1
        if i < 15:
            label[i] = 1
    data_link = join("data","range_data")
    label_link = join("data","range_label")
    np.save(data_link,data)
    np.save(label_link,label)


