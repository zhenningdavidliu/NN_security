import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import model_builders as mb
import tensorflow as tf
from Data_loader_shades import Data_loader_shades
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
import os
from os.path import join

'''
This program will load the weights saved for a network solving the shades of grey squares experiment
and produce a sample of n images where the network will be asked to classify them. This will come
with its accuracy and the final layer weights (describing its "confidence"). The images 
'''
if __name__=='__main__':



    os.environ['CUDA_VISIBLE_DEVICES']= "1"
    
    # I think this is the new way of not occupying all memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #tf_config = tf.compat.v1.ConfigProto()
    #tf_config.gpu_options.allow_growth = True
    #tf_config.log_device_placement = False
    #sess = tf.compat.v1.Session(config=tf_config)

    with open('experiment.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader= yaml.SafeLoader);
    n = cgf['TEST']['arguments']['number_of_samples']
    l = cgf['TEST']['arguments']['grid_size']
    input_shape = (l,l,1)
    output_shape = (1)

    weights_path = "model/1/keras_model_files.h5"

    if cgf['MODEL']['name'].lower() == 'cnndrop':
        model = mb.build_model_cnndrop(input_shape, output_shape, cgf['MODEL']['arguments'])
    else:
        print('Error: model not found')
    
    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer=optimizer,
                  loss=loss_type,
                  metrics = metric_list)

    data_experiment = Data_loader_shades(cgf['TEST']['arguments'])
    experiment_data, experiment_labels = data_experiment.load_data()

    results = model.predict(experiment_data)

    temp = np.rint(results)
    pred_labels = [a for a in list(zip(*temp))[0]]

    acc = 0
    wrongly_labeled_images = list() 
    correct_labels = list()

    for i in range(n):
        if pred_labels[i] == experiment_labels[i]:
            acc += 1
        else: 
            wrongly_labeled_images.append(i)
            correct_labels.append(["Picture number {}".format(i),experiment_labels[i]])

    print("The percentage of correct labels is {} %".format(100*acc/n))

    all_path = join("experiment","all")

    for j in range(n):
        plt.figure()
        plt.imshow(experiment_data[j].reshape(32,32), cmap="gray")
        temp_path = join(all_path,"picture_{:03d}.png".format(j))
        plt.savefig(temp_path)
        plt.close()

    all_path_labels = join(all_path,"labels.txt")

    with open(all_path_labels, 'w') as f:
        for j in range(n):
            f.write("{}\n".format(pred_labels[j]))

    all_path_confidences = join(all_path, "confidences.txt")
    
    temp = np.around(results, decimals=2)
    pred_conf = [a for a in list(zip(*temp))[0]]

    with open(all_path_confidences, "w") as f:
        for j in range(n):
            f.write("{}\n".format(pred_conf[j]))

    mistakes_path = join("experiment","wrong")

    for j in wrongly_labeled_images:
        plt.figure()
        plt.imshow(experiment_data[j].reshape(32,32), cmap="gray")
        temp_path = join(mistakes_path,"picture_{:03d}.png".format(j))
        plt.savefig(temp_path)
        plt.close()

    mistakes_label = join(mistakes_path, "labels.txt") 

    with open(mistakes_label, 'w') as f:
        for item in correct_labels:
            f.write("{}\n".format(item))

    mistakes_confidences = join(mistakes_path, "confidence.txt")

    with open(mistakes_confidences, "w") as f:
        for j in wrongly_labeled_images:
            f.write("{}\n".format(pred_conf[j]))
