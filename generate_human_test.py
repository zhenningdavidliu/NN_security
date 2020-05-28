import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import model_builders as mb
import tensorflow as tf
from data_bank import data_selector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten
import os
from os.path import join
from PIL import Image

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

    data_name = cgf['TEST']['name']
    data_arguments = cgf['TEST']['arguments']

    data_loader = data_selector(data_name, data_arguments)

    experiment_data, experiment_labels, experiment_diff = data_loader.load_data()

    input_shape = experiment_data.shape[1:]
    output_shape = experiment_labels.shape[1]

    model_id = cgf['TEST']['arguments']['model']
    model_dest = cgf['MODEL']['model_dest']
    model_path = join(model_dest, str(model_id))

    weights_path = join(model_path, "keras_model_files.h5")

    model_name = cgf['MODEL']['name']
    model_arguments = cgf['MODEL']['arguments']
    model = mb.model_selector(model_name, input_shape, output_shape, model_arguments)

    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer=optimizer,
                  loss=loss_type,
                  metrics = metric_list)


    results = model.predict(experiment_data)

    temp = np.rint(results)
    pred_labels = [a for a in list(zip(*temp))[0]]

    acc = 0
    wrongly_labeled_images = list() 
    correct_labels = list()
    big_contrast_mistakes_images = list()
    strongly_confident_mistake = list()
 
    temp = np.around(results, decimals=2)
    pred_conf = [a for a in list(zip(*temp))[0]]
    
    for i in range(n):
        if pred_labels[i] == experiment_labels[i]:
            acc += 1
        else: 
            wrongly_labeled_images.append(i)
            correct_labels.append(["Picture number {}".format(i),experiment_labels[i]])
            ''' 
            if experiment_diff[i] ==1:
                big_contrast_mistakes_images.append(i)
            '''
            if abs(pred_labels[i] - pred_conf[i])<0.25:
                strongly_confident_mistake.append(i)
    
    print("The percentage of correct labels is {} %".format(100*acc/n))
    print("The percentage of wrong labels with high confidence is {} %".format(100*len(strongly_confident_mistake)/n))
    print("The percentage of wrong labels with high confidence amongst wrong labels is {} %".format(100*len(strongly_confident_mistake)/(n-acc)))
   

    all_path = join("experimentt","all")

    color_mode = 'L'

    A_kron = np.ones([10,10]) #matrix for Kronecker product(zooms in)

    for j in range(n):
        '''
        plt.figure()
        plt.imshow(experiment_data[j].reshape(32,32), cmap="gray")
        temp_path = join(all_path,"picture_{:03d}.png".format(j))
        plt.savefig(temp_path)
        plt.close()
        '''
        X = 255*experiment_data[j].reshape(l,l)
        X = np.kron(X,A_kron)
        Y = X.astype(np.uint8)

        im = Image.fromarray(Y,mode=color_mode)
        temp_path = join(all_path,"picture_{:03d}.png".format(j))
        im.save(temp_path)
    all_path_labels = join(all_path,"labels.txt")

    with open(all_path_labels, 'w') as f:
        for j in range(n):
            f.write("The label for {} is {}\n".format(j, pred_labels[j]))

    all_path_confidences = join(all_path, "confidences.txt")
    
    temp = np.around(results, decimals=2)
    pred_conf = [a for a in list(zip(*temp))[0]]

    with open(all_path_confidences, "w") as f:
        for j in range(n):
            f.write("On picture number {} the confidence is: {}\n".format(j, pred_conf[j]))

    mistakes_path = join("experimentt","wrong")

    for j in wrongly_labeled_images:
        '''
        plt.figure()
        plt.imshow(experiment_data[j].reshape(32,32), cmap="gray")
        temp_path = join(mistakes_path,"picture_{:03d}.png".format(j))
        plt.savefig(temp_path)
        plt.close()
        '''
        X = 255*experiment_data[j].reshape(l,l)
        X = np.kron(X,A_kron)
        Y = X.astype(np.uint8)
        

        im = Image.fromarray(Y,mode=color_mode)
        temp_path = join(mistakes_path,"picture_{:03d}.png".format(j))
        im.save(temp_path)
        print(temp_path)


    mistakes_label = join(mistakes_path, "labels.txt") 

    with open(mistakes_label, 'w') as f:
        for item in correct_labels:
            f.write("{}\n".format(item))

    mistakes_confidences = join(mistakes_path, "confidence.txt")

    with open(mistakes_confidences, "w") as f:
        f.write("The percentage of wrong labels with high confidence is {} %\n".format(100*len(strongly_confident_mistake)/n))
        f.write("The percentage of wrong labels with high confidence amongst wrong labels is {} %\n".format(100*len(strongly_confident_mistake)/(n-acc)))
        for j in wrongly_labeled_images:
            f.write("On the picture number {}, the confidence is {}\n".format(j, pred_conf[j]))

    big_contrast_mistakes = join(mistakes_path, "big_contrast.txt")

    with open(big_contrast_mistakes, "w") as f:
        for j in big_contrast_mistakes_images:
            f.write("{}\n".format([j]))
