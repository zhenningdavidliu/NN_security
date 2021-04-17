import yaml
import numpy as np
import matplotlib.pyplot as plt
import model_builders as mb
import tensorflow as tf
from data_bank import data_selector
from tensorflow.keras import backend as K
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
from tensorflow.keras.applications.resnet50 import preprocess_input as res_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
import os
from os.path import join
#from PIL import Image


def print_certainty_vs_distance_table(N, M, results, diff):
    """
    Prints a table which compares certainty and distance to decision boundary.

    Arguments
    ---------
    N (int): Number of certainty intervals.
    M (int): Number of distance intervals.
    results (ndarry): Network predictions, as numbers between 0 and 1.
    diff (ndarray): Distance to decision boundary, as a number between 0 and 1.

    Returns
    -------
    table (ndarray): M×N table with the various samples and predictions.
    """
    
    table = np.zeros([M,N]);

    dN = 1.0/N
    dM = 1.0/M

    fline = 'Distance boundary '
    len_dist_bd = len(fline);
    print((len_dist_bd+int((N/2)*11 - 8 ))*" " + "Network certainty")

    data_size = len(results)
    
    for i in range(N):
        int_N = i*dN
        int_N_up = (i+1)*dN
        line_out = "[%3.0f, %3.0f]   " % (int_N*100, int_N_up*100);
        fline += line_out
    print(fline);
    for j in range(M):
        int_M = j*dM
        int_M_up = (j+1)*dM
        if j == M-1:
            int_M_up = 1+1e-10;
        line_out = "[%6.2f, %6.2f]: " % (int_M*100, int_M_up*100);
        for i in range(N):
            int_N = i*dN
            int_N_up = (i+1)*dN
            if i == N-1:
                int_N_up = 1+1e-5;

            acc = 0;
            for k in range(data_size):
                if (int_N <= results[k] < int_N_up) and int_M <= diff[k] < int_M_up:
                    acc += 1;
            table[j,i] = acc;
            tmp = "     %-6d  " % (acc);
            line_out += tmp;
        print(line_out)
    return table;

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']= "-1"

    with open('config_evaluate.yml') as ymlfile:
        cgf_eval = yaml.load(ymlfile, Loader= yaml.SafeLoader)

    model_id = cgf_eval['MODEL']['model_id']
    model_dest = cgf_eval['MODEL']['model_dest']
    model_path = join(model_dest, str(model_id))

    with open(join(model_path, 'config.yml')) as ymlfile:
        cgf = yaml.load(ymlfile, Loader= yaml.SafeLoader)

    # Select dataset
    use_default_test = cgf_eval['DATASET']['use_default']
    if use_default_test:
        data_loader = data_selector(cgf['DATASET_TEST']['name'], cgf['DATASET_TEST']['arguments'])
    else:
        data_loader = data_selector(cgf_eval['DATASET']['name'], cgf_eval['DATASET']['arguments'])


    print('\nDATASET TEST')
    print(data_loader)
    test_data, test_labels, test_diff = data_loader.load_data()

    print(test_data.shape)

    model_name = cgf['MODEL']['name']
    model_arguments = cgf['MODEL']['arguments']

    input_shape = test_data.shape[1:]
    output_shape = test_labels.shape[1];

    # Set the default precision 
    model_precision = cgf['MODEL_METADATA']['precision']
    K.set_floatx(model_precision)

    model = mb.model_selector(model_name, input_shape, output_shape, model_arguments)
    
    filepath = cgf['MODEL_METADATA']['save_best_model']['arguments']['filepath']
    keras_weights_path = join(model_path, filepath)
    model.load_weights(keras_weights_path)

    # Extract training information
    loss_type = cgf['TRAIN']['loss']['type']
    optimizer = cgf['TRAIN']['optim']['type']
    batch_size = cgf['TRAIN']['batch_size']
    metric_list = list(cgf['TRAIN']['metrics'].values()) 
    shuffle_data = cgf['TRAIN']['shuffle'] 
    max_epoch = cgf['TRAIN']['max_epoch']
    stpc_type = cgf['TRAIN']['stopping_criteria']['type']
    print("""\nTRAIN
loss: {}
optimizer: {}
batch size: {}
shuffle data between epochs: {}
max epoch: {}
stopping_criteria: {}""".format(loss_type, optimizer, batch_size, shuffle_data, 
           max_epoch, stpc_type))

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer=optimizer,
                  loss=loss_type,
                  metrics = metric_list)

    if cgf['MODEL']['name'] == 'resnet':
        test_data = np.repeat(test_data,3, -1)
        test_data = tf.cast(test_data, dtype = tf.float16)
        test_labels = tf.cast(test_labels, dtype = tf.float16)
        test_data = res_prep(test_data)
        
    elif cgf['MODEL']['name'] == 'vgg16':
        test_data = np.repeat(test_data,3, -1)
        test_data = tf.cast(test_data, dtype = tf.float16)
        test_labels = tf.cast(test_labels, dtype = tf.float16)
        test_data = vgg_prep(test_data)
        


    results = model.predict(test_data) # Values between 0 and 1.
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Accuracy on test set: {}%'.format(100*score[1]))

    results = np.squeeze(results);
    data_size = len(results);

    accuracy = 0;
    for i in range(data_size):
        accuracy += ( results[i]>0.5 ) == bool(test_labels[i])
    print('Accuracy: ', accuracy/data_size)

    N = 10; # Number of confidences
    M = 10; # Number of distances
    print_certainty_vs_distance_table(N, M, results, test_diff)    

    print("\n\n")
    N = 2; # Number of confidences
    M = 2; # Number of distances
    print_certainty_vs_distance_table(N, M, results, test_diff)    
















