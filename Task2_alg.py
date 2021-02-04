import numpy as np
import yaml
from data_bank import data_selector
import os 
from os.path import join
import sys
from shade2_algo import shades2_alg

if __name__ == "__main__":

    configfile = 'config_exist.yml'
    with open(configfile) as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    data_loader_test = data_selector(cgf['DATASET_TEST']['name'], cgf['DATASET_TEST']['arguments'])
    test_data, test_labels, _ = data_loader_test.load_data()

    score = 0

    for i in range(len(test_labels)):
        
        if shades2_alg(test_data[i])==test_labels[i]:
            score += 1

    print('Accuracy on test set: {}%'.format(100*score/len(test_labels)))
        

