import yaml
import numpy as np
import matplotlib.pyplot as plt
import model_builders as mb
import tensorflow as tf
from data_bank import data_selector
from tensorflow.keras import backend as K
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
import os
from os.path import join
#from PIL import Image



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

    model_name = cgf['MODEL']['name']
    model_arguments = cgf['MODEL']['arguments']

    input_shape = test_data.shape[1:]
    output_shape = test_labels.shape[1];

    # Set the default precision 
    model_precision = cgf['MODEL_METADATA']['precision']
    K.set_floatx(model_precision)

    model = mb.model_selector(model_name, input_shape, output_shape, model_arguments)

    keras_weights_path = join(model_path, "keras_model_files.h5")
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


    

    results = model.predict(test_data) # Values between 0 and 1.
    score = model.evaluate(test_data, test_labels, verbose=0)
    print('Accuracy on test set: {}%'.format(100*score[1]))

    results = np.squeeze(results);
    data_size = len(results);

    accuracy = 0;
    for i in range(data_size):
        accuracy += ( results[i]>0.5 ) == bool(test_labels[i])


    print('Accuracy: ', accuracy/data_size)

























