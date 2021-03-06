from __future__ import print_function

import os, cv2, sys
from re import S
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input
import numpy as np
import glob
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
set_session(tf.compat.v1.Session(config=config))

IMAGE_FILE_PATH_DISTORTED = ""         

path_to_weights = ""

IMAGE_SIZE = 299
INPUT_SIZE = 299

filename_results_labels_test = 'results.txt'

if os.path.exists(filename_results_labels_test):
    sys.exit("file exists")

focal_start = 40
focal_end = 500
classes_focal = list(np.arange(focal_start, focal_end+1, 10))
classes_distortion = list(np.arange(0, 61, 1) / 50.)

def get_paths(IMAGE_FILE_PATH_DISTORTED):

    paths_test = glob.glob(IMAGE_FILE_PATH_DISTORTED + "*.jpg")
    paths_test.sort()
    parameters = []
    labels_focal_test = []
    for path in paths_test:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        labels_focal_test.append(curr_parameter)
    labels_distortion_test = []
    for path in paths_test:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        labels_distortion_test.append(curr_parameter)

    c = list(zip(paths_test, labels_focal_test, labels_distortion_test))
    paths_test, labels_focal_test, labels_distortion_test = zip(*c)
    paths_test, labels_focal_test, labels_distortion_test = list(paths_test), list(labels_focal_test),  list(labels_distortion_test)
    labels_test = [list(a) for a in zip(labels_focal_test, labels_distortion_test)]

    return paths_test, labels_test

paths_test, labels_test = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_test), 'test samples')

y_true_focal = []
y_true_dist = []

y_pred_focal = []
y_pred_dist =  []

with tf.device('/gpu:0'):
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(1, activation='sigmoid', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(1, activation='sigmoid', name='output_distortion')(phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer._name = layer.name + "_phi"

    model = Model(main_input, [final_output_focal, final_output_distortion])
    model.load_weights(path_to_weights)

    print(len(paths_test))
    file = open(filename_results_labels_test, 'a')
    # file = open(filename_results_predict_test, 'a')
    for i, path in enumerate(paths_test):
        if i % 1000 == 0:
            print(i,' ',len(paths_test))
        image = cv2.imread(path)
        image = cv2.resize(image,(INPUT_SIZE,INPUT_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        image = np.expand_dims(image,0)

        image = preprocess_input(image) 

        # loop
        prediction_focal = model.predict(image)[0]
        prediction_dist = model.predict(image)[1]


        curr_focal_label = labels_test[i][0]
        y_true_focal.append(curr_focal_label)

        curr_focal_pred = (prediction_focal[0][0] * (focal_end+1. - focal_start*1.) + focal_start*1. ) * (IMAGE_SIZE*1.0) / (INPUT_SIZE*1.0)
        y_pred_focal.append(curr_focal_pred)


        curr_dist_label = labels_test[i][1]
        y_true_dist.append(curr_dist_label)

        curr_dist_pred = prediction_dist[0][0]*1.2
        y_pred_dist.append(curr_dist_pred)
        
        file.write(path + ' ' + str(curr_focal_pred) + ' '+ str(curr_dist_pred))
        file.write("\n")
    file.close()   