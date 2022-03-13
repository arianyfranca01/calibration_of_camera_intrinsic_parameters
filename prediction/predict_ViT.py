from __future__ import print_function

import os, cv2, sys
from re import S
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.imagenet_utils import preprocess_input
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

IMAGE_FILE_PATH_DISTORTED = "C:/afc8/dataset_continuous/test/"
path_to_weights = 'C:/afc8/DeepCalib-master/pesos/transformer_vit/weights_119_0.07.h5'

IMAGE_SIZE = 299
INPUT_SIZE = 299

filename_results = 'results_transformer_119.txt'

if os.path.exists(filename_results):
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

input_shape = (299, 299, 3)
#Configure the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 240
num_epochs = 10000
image_size = 300  # We'll resize input images to this size
patch_size = 50  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final

print(batch_size)

image_size = 300

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

#Compute the mean and the variance of the training data for normalization.

#Implement multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_regressor():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    #logits = layers.Dense(num_classes)(features)
    final_output_focal = layers.Dense(1, name='output_focal')(features)
    final_output_distortion = layers.Dense(1, name='output_distortion')(features)
    # Create the Keras model.
    model = keras.Model(inputs, [final_output_focal, final_output_distortion])
    return model

with tf.device('/gpu:0'):
    model = create_vit_regressor()
    model.load_weights(path_to_weights)

    print(len(paths_test))
    file = open(filename_results, 'a')
  
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