import foolbox as fb
import tensorflow as tf


'''
This script is for implementing deep fool attacks on our stripes examples.

So far only one function is defined, that being FGSM. What it requires is 

model: takes in a tensorflow model
pre : preprocessing for the model. (Exists for Resnet and VGG16)
image : input
label: output
epsilon : the max distance (i think in l-inf norm) for which adv attacks are searched for
'''

def FGSM_attack(model, pre, image, label, epsilon):

    prep = dict()
    fmodel = fb.models.TensorFlowModel(model, bounds = (0,1) , device = None, preprocessing = prep)
    attack = fb.attacks.FGSM()

    # print('The original accuracy is: {}'.format(fb.utils.accuracy(fmodel, images, labels))
    
    image = image.reshape((1, 224, 224, 1))
    image = tf.convert_to_tensor(image)
    #image =tf.convert_to_tensor(image)
    #label = tf.convert_to_tensor([int(label)])
    
    label = int(label)

    if label == 1:
        label = 1 - 1e-8

    label = tf.convert_to_tensor([label], dtype=tf.float64)

    print(type(image))
    print(label)


    raw, clipped, is_adv = attack(fmodel, image, tf.convert_to_tensor(label), epsilons = epsilon)

    # robust_accuracy = 1 - is_adv.float32().mean(axis=-1)
    
    return raw, clipped, is_adv



    
