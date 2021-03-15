import numpy as np
import yaml
import shutil
import uuid
from tensorflow.keras import backend as K
from nn_tools import read_count
import tensorflow as tf
from data_bank import data_selector
import model_builders as mb
import os
from os.path import join
import matplotlib.pyplot as plt
import sys
from adversarial_attacks.spsa import spsa, spsa_T1
from adversarial_attacks.df_attacks import FGSM_attack
from tensorflow.keras.applications.resnet50 import preprocess_input as res_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def create_adversarial_pattern(input_image, input_label, pretrained_model):

    loss_object = tf.keras.losses.binary_crossentropy
    with tf.GradientTape() as tape:
        image = np.expand_dims(input_image, axis = 0)
        image = tf.convert_to_tensor(image)
        tape.watch(image)
        prediction = pretrained_model(image)
        loss = loss_object(input_label, prediction)
    gradient =  tape.gradient(loss,image)
    signed_grad = tf.sign(gradient)
    return signed_grad



if __name__ == "__main__":
    '''
    need to load model
    need to load training data
    perform adversarial attack
    '''
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    configfile = 'config_adv.yml'
    with open(configfile) as ymlfile:
        cgf = yaml.load(ymlfile, Loader =yaml.SafeLoader)

 # Set up computational resource 
    use_gpu = cgf['COMPUTER_SETUP']['use_gpu']
    print("""\nCOMPUTER SETUP
Use gpu: {}""".format(use_gpu))
    if use_gpu:
        compute_node = cgf['COMPUTER_SETUP']['compute_node']
        os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
        print('Compute node: {}'.format(compute_node))
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

    # Turn on soft memory allocation
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.compat.v1.Session(config=tf_config)
    #K.v1.set_session(sess)
 
    data_loader = data_selector(cgf['DATASET']['name'], cgf['DATASET']['arguments'])

    data, labels, _ = data_loader.load_data() 

    '''
    data = np.load(cgf["DATASET"]["arguments"]["images"])
    labels = np.load(cgf["DATASET"]["arguments"]["labels"])
    '''

    # Get input and output shape
    input_shape = data.shape[1:]
    output_shape = labels.shape[1];
    print('input_shape', input_shape)
    # Set the default precision 
    model_precision = cgf['MODEL_METADATA']['precision']
    K.set_floatx(model_precision)

    model_id = cgf['MODEL_METADATA']['model_number_arguments']['model_id']
    model_path = join('model', str(model_id))
    filepath = cgf['MODEL_METADATA']['save_best_model']['arguments']['filepath']
    attack = cgf['ATTACK']['name']

    adv_data = cgf['DATASET']['arguments']["adv_images"]
    adv_labels = cgf['DATASET']['arguments']['adv_labels']

    weights_path = join(model_path, filepath)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    if loss_type == 'SparseCategoricalCrossentropy':
        loss_type = SparseCategoricalCrossentropy(from_logits=False)
        metric_list = [SparseCategoricalAccuracy()]
        output_shape = 2
        labels = np.reshape(labels, (-1))

    model_name = cgf['MODEL']['name']
    model_arguments = cgf['MODEL']['arguments']
    model = mb.model_selector(model_name, input_shape, output_shape, model_arguments)

    model.load_weights(weights_path)

    # Preprocessing
    if model_name =='resnet':
        preprocessing = res_prep
    elif model_name == 'vgg16':
        preprocessing = vgg_prep
    else:
        preprocessing = None 

    model.compile(optimizer=optimizer,
                  loss=loss_type,
                  metrics = metric_list)

    model.trainable = False

    adversarials = []
    adv_lab = []

    data_size = 50
    
    if attack == 'spsa':
        for i in range(data_size): 
            colors = set(data[i].flatten())
            colors = [colors.pop(),colors.pop(),colors.pop()]
            color_diff = [abs(colors[0]-colors[1]), abs(colors[0]-colors[2]), abs(colors[1]-colors[2])]
            lower = min(color_diff)

            epsilon = lower/4
            delta = epsilon/8
            alpha = delta
            T = 20
            n = 5

            candidates = spsa_T1(model, data[i], delta, alpha, n, epsilon, T)
            for cand in candidates:
                cand = np.expand_dims(cand, axis=0)
                if round(model.predict(cand)[0,0]) !=  labels[i][0]:
                    adversarials.append(cand)
                    adv_lab.append(labels[i])

    elif attack == 'FGSM':
        for i in range(data_size): 
            colors = set(data[i].flatten())
            colors = [colors.pop(),colors.pop(),colors.pop()]
            color_diff = [abs(colors[0]-colors[1]), abs(colors[0]-colors[2]), abs(colors[1]-colors[2])]
            lower = min(color_diff)

            epsilon = lower/4

            candidates = FGSM_attack(model, preprocessing, data[i], labels[i], epsilon)
            print(candidates)
            '''
            for cand in candidates:
                if round(model.predict(cand)[0,0]) !=  labels[i][0]:
                    adversarials.append(cand)
                    adv_lab.append(labels[i])
            '''


    print(adversarials)
    np.save(adv_data,adversarials)
    np.save(adv_labels, adv_lab)
    grid_x = 1
    grid_y = 1
    data_size = grid_x*grid_y
    shuffle = False

    if not shuffle:
        perm_order = np.arange(data_size)
    else:
        perm_order = np.random.permutation(data_size)

    plt.figure()
    for i in range(data_size):
        idx = perm_order[i]

        plt.subplot(grid_y, grid_x, i+1)
        plt.matshow(adversarials[idx], cmap = 'gray', fignum=False)
        plt.axis('off')
        plt.title('Label: {}, neural net predicts: {}'.format(int(labels[idx]), int(ans[idx])))

    plt.show()
