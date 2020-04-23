import yaml
import numpy as np
import os
from os.path import join
import model_builders as mb
import tensorflow as tf


if __name__=="__main__":

    with open('experiment_range.yml') as ymlfile:
        cgf = yaml.load(ymlfile, Loader= yaml.SafeLoader)


    data_link = join("data","range_data.npy")
    label_link= join("data","range_label.npy")

    data = np.load(data_link)
    labels = np.load(label_link)

    os.environ['CUDA_VISIBLE_DEVICES']= "1" 

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            print(e)

    input_shape = (128,128,1)
    output_shape = (1)

    weights_path = "model/12/keras_model_files.h5"

    if cgf['MODEL']['name'].lower() == 'cnndrop':
        model = mb.build_model_cnndrop(input_shape, output_shape, cgf['MODEL']['arguments'])
    elif cgf['MODEL']['name'].lower() == 'fc3':
         model = mb.build_model_fc3(input_shape, output_shape, cgf['MODEL']['arguments'])       
    else:
        print('Error: model not found')

    model.load_weights(weights_path)

    optimizer = cgf['TRAIN']['optim']['type']
    loss_type = cgf['TRAIN']['loss']['type']
    metric_list = list(cgf['TRAIN']['metrics'].values())

    model.compile(optimizer = optimizer,
                  loss=loss_type,
                  metrics = metric_list)

    data = np.expand_dims(data,axis=3)

    results = model.predict(data)

    print(results)
