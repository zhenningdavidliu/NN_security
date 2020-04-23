import yaml
import os
from os.path import join
import matplotlib.pyplot as plt;
import numpy as np;
from data_bank import data_selector


if __name__ == "__main__":
    plt.rcParams["figure.figsize"]= (20,20)

    grid_x = 3
    grid_y = 3 
    shuffle = False
    data_size = grid_x*grid_y
    data_generate_size = 60;

    if not shuffle:
        perm_order = np.arange(data_size);
    else:
        perm_order = np.random.permutation(data_size);
    
    configfile = 'config_visualise.yml'
    with open(configfile) as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);

    data_name = cgf['data_name'];
    data_arguments = cgf['data_arguments'];

    data_loader = data_selector(data_name, data_arguments)
    print(data_loader)
    data_images, data_labels = data_loader.load_data();
    data_images = np.squeeze(data_images);
    print(data_images.shape)
    print(data_labels.shape)
    plt.figure(); 
    for i in range(data_size):
        idx = perm_order[i];
        
        plt.subplot(grid_y, grid_x, i+1);
        plt.matshow(data_images[idx], cmap='gray', fignum=False);
        plt.axis('off')
        plt.title('Label: {}'.format(int(data_labels[idx])));

    plt.show();



