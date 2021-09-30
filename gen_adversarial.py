#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from adversarial_attacks.df_attacks import attack_network
from tensorflow.keras.applications.resnet50 import preprocess_input as res_prep
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_prep
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


# In[2]:


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

def put_in_range(img):
    
    out = np.zeros([1,224,224,1])
    
    for i in range(224):
        for j in range(224):
            if img[0,i,j]>1:
                out[0,i,j,0]=1
            elif img[0,i,j]<0:
                out[0,i,j,0] = 0
            else:
                out[0,i,j,0] = img[0,i,j,0]
    return out


# In[3]:


#Setup model

'''
need to load model
need to load training data
perform adversarial attack
'''

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

configfile = 'config_files/config_adv.yml'
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


# In[4]:


data_loader = data_selector(cgf['DATASET']['name'], cgf['DATASET']['arguments'])
data, labels, diff = data_loader.load_data() 


# In[5]:


data = data[:3000,:,:]
labels = labels[:3000]

'''
data = np.load(cgf["DATASET"]["arguments"]["images"])
labels = np.load(cgf["DATASET"]["arguments"]["labels"])
'''


# In[6]:


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

original_data = cgf['DATASET']['arguments']['original_images']
adv_data = cgf['DATASET']['arguments']["adv_images"]
adv_labels = cgf['DATASET']['arguments']['adv_labels']
adv_diffs = cgf['DATASET']['arguments']['adv_diffs']

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
#model = mb.model_selector(model_name, input_shape, output_shape, model_arguments)

model = tf.keras.models.load_model(weights_path)

# Preprocessing
if model_name =='resnet':
    preprocessing = res_prep
    data = 255*data
    data = data - 122

    #data = tf.cast(data, dtype=tf.float32)
    #labels = tf.cast(data, dtype=tf.float32)
elif model_name == 'vgg16':
    preprocessing = vgg_prep
    data = 255*data
    data = data - 122
    #labels = np.reshape(labels,(-1))
    #data = tf.cast(data, dtype=tf.float32)
    #labels = tf.cast(data, dtype=tf.float32)
else:
    preprocessing = None 
    data = 255*data
    data = data - 122
    
model.compile(optimizer=optimizer,
              loss=loss_type,
              metrics = metric_list)

model.trainable = False


# In[ ]:


# Evaluate the model on training data (can also do on test)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on data")
if model_name =='resnet':
    eval_data = tf.cast(data, dtype=tf.float32)
    eval_labels = tf.cast(labels, dtype=tf.float32)
    eval_labels = np.reshape(eval_labels,(-1))

    results = model.evaluate(eval_data, eval_labels, batch_size=50)
elif model_name =='vgg16':
    eval_data = tf.cast(data, dtype=tf.float32)
    eval_labels = tf.cast(labels, dtype=tf.float32)
    eval_labels = np.reshape(eval_labels,(-1))
    results = model.evaluate(eval_data, eval_labels, batch_size=50)
else:
    results = model.evaluate(tf.convert_to_tensor(data), tf.convert_to_tensor(labels), batch_size=50)

print("Loss :{}, Accuracy: {}%.".format(results[0], results[1]))


# In[ ]:


data_loader = data_selector(cgf['DATASET']['name'], cgf['DATASET']['train_arguments'])

data_train, labels_train, diff_train = data_loader.load_data()

# Preprocessing
if model_name =='resnet':
    preprocessing = res_prep
    data_train = 255*data_train
    data_train = data_train-122
    data_train = tf.cast(data_train, dtype=tf.float32)
    labels_train = tf.cast(labels_train, dtype=tf.float32)
    labels_train = np.reshape(labels_train,(-1))
    #data = tf.cast(data, dtype=tf.float32)
    #labels = tf.cast(data, dtype=tf.float32)
elif model_name == 'vgg16':
    preprocessing = vgg_prep
    data_train = 255*data_train
    data_train = data_train - 122
    data_train = tf.cast(data_train, dtype=tf.float32)
    labels_train = tf.cast(labels_train, dtype=tf.float32)
    labels_train = np.reshape(labels_train,(-1))
    #data = tf.cast(data, dtype=tf.float32)
    #labels = tf.cast(data, dtype=tf.float32)
else:
    data_train = 255*data_train
    data_train = data_train - 122
    preprocessing = None 

# Evaluate the model on training data (can also do on test)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on data")
results = model.evaluate(tf.convert_to_tensor(data_train), tf.convert_to_tensor(labels_train), batch_size=50)
print("Loss :{}, Accuracy: {}%.".format(results[0], results[1]))


# ## Check whether at least one attack works

# In[ ]:


# Do adversarial attacks on training data (can also do on test)
attack_types = ['LinfPGD', 'LinfDeepFoolAttack', 'FGSM', 'L2PGD',  'PGD', 'L2DeepFoolAttack'] # 'FGM',
attack_type_success = dict((att, 0) for att in attack_types)
image_vulnerabilities = dict()
originals = []
adversarials = []
adv_lab = []
ans = []
diffs = []
success = 0
saved = False

plt.figure()

data_size = 3000

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

else:
    for i in range(data_size): 
        image_vulnerabilities['{}'.format(i)] = False
        if np.argmax(model.predict(tf.convert_to_tensor(data[i].reshape((1, 224, 224, 1))))) == int(labels[i]):
            
            # Only images that get correctly labeled in the first place
            colors = set(data[i].flatten())
            # print('The original shades are {}'.format(colors))
            colors = [colors.pop(),colors.pop(),colors.pop()]
            color_diff = [abs(colors[0]-colors[1]), abs(colors[0]-colors[2]), abs(colors[1]-colors[2])]
            lower = min(color_diff)

            epsilon = lower/4

            for attack_choice in attack_types:
                #_, clipped, _= attack_selector(attack, model, preprocessing, data[i], labels[i], epsilon)
                _, cand, _= attack_network(attack_choice, model, model_name, preprocessing, data[i], labels[i], epsilon)

                # print('The ones generated by the adversarial attack are {}'.format(set(cand.flatten())))


                if np.argmax(model.predict(cand)) !=  int(labels[i]):
                    
                    attack_type_success[attack_choice] += 1
                    image_vulnerabilities['{}'.format(i)] = True
                    
                    img_to_show = (np.reshape(data[i],(224,224)) +122)/255
                    cand_to_show = (np.reshape(cand,(224,224)) +122)/255
                
                    if saved == False:
                        
                        saved = True
                        
                        originals.append(img_to_show)
                        adversarials.append(cand_to_show)
                        adv_lab.append(labels[i])
                        diffs.append(diff[i])
                        ans.append(np.argmax(model.predict(cand)))
                    
            if image_vulnerabilities['{}'.format(i)]:
                
                success += 1
            
            if i%100==0:
                
                print(i)
        saved = False
                
                    
f = open("attack_type_success.txt","w")
f.write( str(attack_type_success) )
f.close()

g = open("image_vulnerabilities.txt","w")
g.write( str(image_vulnerabilities) )
g.close()
    
np.save(original_data, originals)
np.save(adv_data, adversarials)
np.save(adv_labels, adv_lab)
np.save(adv_diffs, diffs)

print('Adversarial susceptibility : {:.2f}%'.format(100*success/data_size))


# In[ ]:


# For FC3 381 and 629 in training become adversarial with LinfPGD
# For CNN4 38, 101, 313, 354, 562, 623, 629, 731, 839, 1013, 1071, 1091, 
# 1130, 1266, 1359, 1606, 1640, 1642, 1716, 1811, 1969, 2003, 2017, 2240, 2287, 2290, 2452, 2458, 2652, 2749, 2779
# 2947, 2955, 
attack_type_success


# In[ ]:


print('Adversarial susceptibility : {:.2f}%'.format(100*success/data_size))


# # Check on existing adv images

# In[ ]:





# In[ ]:


adv_diffs = cgf['DATASET']['arguments']['adv_diffs']
np.save(adv_diffs, diffs[:300])


# In[ ]:


for j in range(10):
    
    plt.figure(figsize=(20,10))

    plt.subplot(122)
    plt.matshow(originals[j], cmap = 'gray', fignum=False)
    plt.axis('off')
    plt.title('Adversarial image ({} attack) number of image: {}, label: {}, neural net predicts: {}'.format(attack, i+1, int(labels[i]), np.argmax(model.predict(cand))))

    plt.show()

for i in range(300):
    
                plt.figure(figsize=(20,10))

                plt.subplot(122)
                plt.matshow(adversarials[i], cmap = 'gray', fignum=False)
                plt.axis('off')
                plt.title('Adversarial image ({} attack) number of image: {}, label: {}, neural net predicts: {}'.format(attack, i+1, int(labels[i]), np.argmax(model.predict(cand))))

                plt.show()


# In[ ]:


plt.figure(figsize=(20,10))

plt.subplot(122)
plt.matshow(data_train[0], cmap = 'gray', fignum=False)
plt.axis('off')
plt.title('Adversarial image ({} attack) number of image: {}, label: {}, neural net predicts: {}'.format(attack, i+1, int(labels[i]), np.argmax(model.predict(cand))))

plt.show()


