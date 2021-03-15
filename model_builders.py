import tensorflow as tf
import numpy as np
from tensorflow import Tensor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv1D, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Add, BatchNormalization, ReLU
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

def model_selector(model_name, input_shape, output_shape, arguments):
    """ Select a model (network) based on `model_name` (str).
Arguments
---------
model_name (str): Name of the model loader
input_shape (list): List of integers specifying dimensions
output_shape (list): List of integers specifying dimensions
arguments (dict): Arguments to the model function

Returns
-------
Keras model
"""
    if model_name.lower() == "fc3":
        return build_model_fc3(input_shape, output_shape, arguments)
    elif model_name.lower() == "fc2":
        return build_model_fc2(input_shape, output_shape, arguments)
    elif model_name.lower() == "fc2_cheat":
        return build_model_fc2_cheat(input_shape, output_shape, arguments)
    elif model_name.lower() == "cnn2":
        return build_model_cnn2(input_shape, output_shape, arguments)
    elif model_name.lower() == "cnn3":
        return build_model_cnn3(input_shape, output_shape, arguments)
    elif model_name.lower() == "cnn4":
        return build_model_cnn4(input_shape, output_shape, arguments)
    elif model_name.lower() == "cnndrop":
        return build_model_cnndrop(input_shape, output_shape, arguments)
    elif model_name.lower() == "resnet":
        return build_model_resnet(output_shape, arguments, input_shape)
    elif model_name.lower() == "resnet_load":
        return build_model_resnet_load(output_shape, arguments, input_shape)
    elif model_name.lower() == "vgg16":
        return build_model_vgg16(output_shape, arguments, input_shape)
    else:
        print('Error: Could not find model with name %s' % (model_name))
        return None;

def build_model_resnet(output_shape, arguments, input_shape=(224, 224, 1)):
    
    # colored (224, 224, 3) 
    inputs = Input(shape=input_shape)
    final_act = arguments['final_act']

    z = ResNet50(input_tensor = inputs, weights = None, include_top=False)
    # weights = 'imagenet'

    for layer in z.layers:
        layer.trainable = False

    y = Flatten(name="flatten")(z.output)
    
    if final_act == 'none':
        output = Dense(output_shape)(y)
    else:
        output = Dense(output_shape, activation=final_act)(y) 

    model = Model(inputs, output)

    return model

def build_model_resnet_load(output_shape, arguments, input_shape=(224, 224, 3)):
    
    inputs = Input(shape=input_shape)
    final_act = arguments['final_act']

    z = ResNet50(input_tensor = inputs, weights = 'imagenet', include_top=False)

    for layer in z.layers:
        layer.trainable = True   
    
    y = Flatten(name="flatten")(z.output)
    
    if final_act == 'none':
        output = Dense(output_shape)(y)
    else:
        output = Dense(output_shape, activation=final_act)(y) 

    model = Model(inputs, output)

    return model


def relu_bn(inputs: Tensor) -> Tensor:
    
    relu = ReLu()(inputs)
    bn = BatchNormalization()(relu)
    
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int=3) -> Tensor:

    y = Conv2D(kernel_size = kernel_size,
            strides = (1 if not downsample else 2),
            filters=filters,
            padding="same")(x)

    y = relu_bn(y)
    y = Conv2d(kernel_size = kernel_size, 
                strides=1,
                filters=filters,
                padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                strides=2,
                filters=filters,
                padding="same")(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out

def build_model_vgg16(output_shape, arguments, input_shape=(224,224,1)):

    
    inputs = Input(shape=input_shape)
    final_act = arguments['final_act']

    z = VGG16(input_tensor = inputs, weights = None, include_top=False)

    for layer in z.layers:
        layer.trainable = False

    y = Flatten(name="flatten")(z.output)
    
    if final_act == 'none':
        output = Dense(output_shape)(y)
    else:
        output = Dense(output_shape, activation=final_act)(y) 

    model = Model(inputs, output)

    return model


def build_model_cnndrop(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Flatten())	
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 


    return model



def build_model_fc3(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=5, strides=(1,1), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(1,1), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(10))
    
    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 

    return model

def build_model_fc2(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
    bias = arguments['bias_initializer'] 

    model=Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(8,activation = act, use_bias=True)) #, bias_initializer= bias))#, activity_regularizer = regularizers.l1(0.01)))

    if final_act == 'none':
        model.add(Dense(output_shape, use_bias = True))
    else:
        model.add(Dense(output_shape, activation=final_act, use_bias=True)) 


    return model

def build_model_fc2_cheat(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
    bias = arguments['bias_initializer'] 
    weights_0 = np.matrix.transpose(np.concatenate(([np.ones(32*32)],np.zeros([31,32*32])), axis =0))
    bias_0 = np.zeros(32)
    bias_1 = np.ones(1)
    bias_1 = -96*bias_1
    weights_1 = np.zeros([31,1])
    weights_1 = np.concatenate((np.ones([1,1]),weights_1), axis = 0)

    model=Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(32,activation = act, weights =[weights_0,bias_0])) 

    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 


    return model

def build_model_cnn2(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=2, strides=(2,2), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(10))

    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 


    return model

def build_model_cnn3(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=2, strides=(2,2), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(20))

    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 


    return model

def build_model_cnn4(input_shape, output_shape, arguments):
    act = arguments['act'];
    final_act = arguments['final_act']
     
    model=Sequential()
    model.add(Conv2D(filters=24, kernel_size=2, strides=(2,2), padding='same', activation=act, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Conv2D(filters=48, kernel_size=2, strides=(2,2), padding='same',activation=act))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same', strides=None))
    model.add(Flatten())	
    model.add(Dense(20))

    if final_act == 'none':
        model.add(Dense(output_shape))
    else:
        model.add(Dense(output_shape, activation=final_act)) 


    return model
