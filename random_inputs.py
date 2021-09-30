import yaml 
import numpy as np
import tensorflow
import model_builders as mb
from keras.models import Sequential 
from keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
import os

def random_labeled_data(data_size,randomness, param):
    """	
    We load a trained model and generate a train set either using 
    uniform or a gaussian random matrices

    randomness can be set to : 

        gaussian: Input param has to be [m,stdev] which creates random images
        with a gaussian distribution for each pixel with mean m and standar 
        deviation stdev

        uniform: Input params [a,b] and creates random images with uniform 
        distribution for each pixel

        stripes: creates images for the stripes case

        none: Input parameter a. Creates images with all pixels set to b for 
        all (given some step size a/data_size) values from 0 to a.

    """

    with open('config_files/config_random.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader=yaml.SafeLoader);
    n = cgf['DATASET_TRAIN']['arguments']['grid_size']
    input_shape = (n,n,1)
    output_shape = (1)	

    use_gpu = cgf['COMPUTER_SETUP']['use_gpu']
    if use_gpu:
        compute_node = cgf['COMPUTER_SETUP']['compute_node']
        os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
    model_num = cgf['DATASET_TRAIN']['arguments']['model']
    model_name = cgf['MODEL']['name']

    weights_path = "model/"+ str(model_num) + "/keras_model_files.h5"	
    model = mb.model_selector(model_name, input_shape, output_shape, cgf['MODEL']['arguments'])
    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type= cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer=optimizer,
                 loss=loss_type,
                 metrics= metric_list)

    if randomness == "gaussian":
        mean = param[0]
        var = param[1]
        data = np.random.normal(mean, var, size=(data_size, n, n, 1))
    elif randomness == "uniform":				
        #data = np.random.uniform(low=0, high=0.1875, size=(data_size, n, n, 1)) 
        lowb = param[0]
        highb = param[1]
        data = np.random.uniform(low=lowb, high=highb, size=(data_size, n, n, 1)) 

    elif randomness == "stripes":
        data_loader_test = Data_loader_stripe_test(cgf['DATASET_TEST']['arguments'])
        data, _ =  data_loader_test.load_data() 

    elif randomness == "none":
        a = param
        data = np.ones((data_size, n,n,1))

        for i in range(data_size):
            data[i,:,:,0] = (a/(i+1))
            
    sum_pixels = [i.sum() for i in data[:]]


    labels = model.predict(data)
    return data, labels, sum_pixels
