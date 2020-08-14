from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg

import tensorflow as tf

tf.enable_eager_execution()

model_used = "my_model-20200813-215626.h5"
model = load_model("models/" + model_used)

img = ImageDataGenerator(rescale=1. / 255)
test_gen = img.flow_from_directory(
    directory='./dataset_catdog/dataset/test',
    target_size=(150, 150),
    color_mode='rgb',
    shuffle=False
)


def incorrect_samples_index(prediction_model, test_generator):
    """
    returns indexes of all incorrectly classified images
    :param prediction_model: Our prediction
    :param test_generator: Our generator for our data
    :return: List of indexes
    """
    labels = test_generator.labels
    result = np.round(prediction_model.reshape(-1)).astype("int")
    index_list = []
    for i in range(test_gen.samples):
        if result[i] != labels[i]:
            index_list.append(i)
    return index_list


def plot_file_images(plt_object, filenames, rows, cols, base_index=None):
    fig, ax = plt_object.subplots(rows, cols)
    index_range = ax.shape[0] * ax.shape[1] + ax.shape[1]
    if base_index is None:
        base_index = random.randint(0, len(filenames) - index_range)
    elif base_index > len(filenames) - index_range:  # check that base_index+index_range exceeds list size
        base_index = len(filenames - index_range)
    fig.suptitle(f"Wrong predictions for range {base_index} to {base_index + index_range}")
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            img = mpimg.imread(filenames[base_index + (r * cols + c)])
            ax[r, c].imshow(img)
            ax[r, c].set_yticklabels([])
            ax[r, c].set_xticklabels([])


def plot_dataframe_images(plt_object, dataframe, rows, cols, base_index=None):
    """

    :param plt_object: plot object
    :param dataframe: Data frame with the images
    :param rows: number of rows in subsplot
    :param cols: number of columns in subplot
    :param base_index: base image index to start from. If nothing is given it chooses randomly.
    """
    fig, ax = plt_object.subplots(rows, cols)
    index_range = rows * cols + cols
    if base_index is None:
        base_index = random.randint(0, len(dataframe) - index_range)
    elif base_index > len(dataframe) - index_range:  # check that base_index+index_range exceeds list size
        base_index = len(dataframe - index_range)
    fig.suptitle(f"Wrong predictions for range {base_index} to {base_index + index_range}")
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            ax[r, c].imshow(dataframe[base_index + (r * cols + c)])
            ax[r, c].set_yticklabels([])
            ax[r, c].set_xticklabels([])


def get_sample_dataframe(test_generator):
    test_x = []
    for i in range(test_generator.__len__()):
        test_x.extend(
            test_generator.__getitem__(i)[0]
        )
    return test_x


### Setup code for running
# m = model.predict(test_gen)  # make a prediction

# x_test_dataframe = get_sample_dataframe(test_gen)  # get the images from test generator as dataframe
# x_test_filepaths = test_gen.filepaths  # get all image file paths
#
# wrong_predict_indexes = incorrect_samples_index(m, test_gen) # get indexes for wrongly classified images

# use indexes to make a list of images that have been augmented
# wrong_predicts_dataframe = [x_test_dataframe[i] for i in wrong_predict_indexes]

# plot examples
# plot_dataframe_images(plt, wrong_predicts_dataframe, 4, 8, base_index=0)
# plot_file_images(plt, x_test_filepaths, rows=4, cols=8, base_index=100)

# plt.show()

print(model.layers[0])

from keras import backend as K

input_img = model.inputs[0]
layer_dict = dict([(layer.name, layer) for layer in model.layers])


layer_name = 'conv2d'
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])