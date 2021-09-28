import foolbox as fb
import tensorflow as tf
from foolbox.attacks import *


'''
This script is for implementing deep fool attacks on our stripes examples.

So far only one function is defined, that being FGSM. What it requires is 

model: takes in a tensorflow model
pre : preprocessing for the model. (Exists for Resnet and VGG16)
image : input
label: output
epsilon : the max distance (i think in l-inf norm) for which adv attacks are searched for
'''

'''
def attack_selector(attack, model, pre, image, label, epsilon):
    
    if attack == 'FGSM':
        return FGSM_attack(model, pre, image, label, epsilon)
    if attack == 'FGM':
        return FGM_attack(model, pre, image, label, epsilon)
    if attack == 'NewtonFool':
        return NewtonFool_attack(model, pre, image, label, epsilon)
    
def FGSM_attack(model, pre, image, label, epsilon):

    prep = dict()
    fmodel = fb.models.TensorFlowModel(model, bounds = (0,1) , device = None, preprocessing = prep)
    attack = fb.attacks.FGSM()
    
    image = image.reshape((1, 224, 224, 1))
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor([int(label)])

    raw, clipped, is_adv = attack(fmodel, image, tf.convert_to_tensor(label), epsilons = epsilon)

    return raw, clipped, is_adv

def FGM_attack(model, pre, image, label, epsilon):

    prep = dict()
    fmodel = fb.models.TensorFlowModel(model, bounds = (0,1) , device = None, preprocessing = prep)
    attack = fb.attacks.FGM()
    
    image = image.reshape((1, 224, 224, 1))
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor([int(label)])

    raw, clipped, is_adv = attack(fmodel, image, tf.convert_to_tensor(label), epsilons = epsilon)

    return raw, clipped, is_adv

def NewtonFool_attack(model, pre, image, label, epsilon):

    prep = dict()
    fmodel = fb.models.TensorFlowModel(model, bounds = (0,1) , device = None, preprocessing = prep)
    attack = fb.attacks.NewtonFoolAttack()
    
    image = image.reshape((1, 224, 224, 1))
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor([int(label)])

    raw, clipped, is_adv = attack(fmodel, image, tf.convert_to_tensor(label), epsilons = epsilon)

    return raw, clipped, is_adv
'''

def attack_network(attack, model, model_name, pre, image, label, epsilon):
    
    prep = dict()
    if (model_name == 'resnet')^(model_name == 'vgg16'):
        fmodel = fb.models.TensorFlowModel(model, bounds = (-122,123) , device = None, preprocessing = prep)
    else:
        fmodel = fb.models.TensorFlowModel(model, bounds = (-122,123) , device = None, preprocessing = prep)
    attack = eval(attack+'()')
    
    image = image.reshape((1, 224, 224, 1))
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor([int(label)])

    raw, clipped, is_adv = attack(fmodel, image, tf.convert_to_tensor(label), epsilons = epsilon)#, epsilons = epsilon)

    return raw, clipped, is_adv
