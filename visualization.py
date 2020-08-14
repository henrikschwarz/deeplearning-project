from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg

def incorrect_samples_index(prediction_model, test_generator):
    """
    returns indexes of all incorrectly classified images
    prediction_model : Our prediction
    test_generator : Our test generator for our data
    return : List of indexes
    """
    labels = test_generator.labels # Get generator labels
    result = np.round(prediction_model.reshape(-1)).astype("int") # get prediction labels
    index_list = [] # List to store indexes in
    for i in range(test_gen.samples):
        if result[i] != labels[i]: # If prediction doesn't match correct label
            index_list.append(i)   # add to list
    return index_list #


def plot_dataframe_images(plt_object, prediction_model, test_generator, rows, cols):
    """
    plt_object : matplotlib obbject
    prediction_model : Predicted classes
    test_generator : test data generator
    rows : rows in figure
    cols : columns in figure
    base_
    """
    fig, ax = plt_object.subplots(rows, cols, figsize(15,15)) # Create subplot
    wrong_indexes = incorrect_samples_index(m, test_gen) # Get indexes for wrong classifications
    dataframe = get_sample_dataframe(test_gen) # Get the images from test generator
    x_labels = np.round(prediction_model) # get predicted classes and round them
    index_range = rows * cols # Amount of pictures
    random_samples = random.sample(wrong_indexes, index_range) # Sample random data
    fig.suptitle(f"{index_range} wrong predictions") # Create figure title
    for r in range(ax.shape[0]): # loop rows
        for c in range(ax.shape[1]): # loop columns
            data_index = random_samples[(r * cols + c)] # index for incorrect classification
            ax[r, c].imshow(dataframe[data_index]) # image data from dataframe
            ax[r, c].set_yticklabels([]) # Remove ticks for aesthetics
            ax[r, c].set_xticklabels([])
            # Describe images
            ax[r, c].set_xlabel(f"Prediction {x_labels[data_index]}\n true label {test_gen.labels[data_index]}")


def plot_history_accuracy(plt_obj, model_history):
    # Plot accuracy and loss in a 2x1 subplot based on model performance every epoch
    accuracy = model_history.history["acc"]
    loss = model_history.history["loss"]
    val_accuracy = model_history.history["val_acc"]
    val_loss = model_history.history["val_loss"]
    fig, ax = plt_obj.subplots(2,1)
    ax[0].plot(accuracy, 'r', label='Accuracy')
    ax[0].plot(val_accuracy, 'b', label="Validation Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy for training data")
    ax[1].plot(loss, label="Loss")
    ax[1].plot(val_loss, label="Validation Loss")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss')
    ax[1].set_title("Loss and Validation loss")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.set_figheight(15)
    fig.set_figwidth(15)

def get_sample_dataframe(test_generator):
    # Get all images in a sequential list as a dataframe
    test_x = []
    for i in range(test_generator.__len__()):
        test_x.extend(
            test_generator.__getitem__(i)[0]
        )
    return test_x


def plot_filters(plt_obj, model, rows, cols, filter_index, image):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    X_train = image
    activations = activation_model.predict(X_train)
    activation = activations[filter_index]
    activation_index=0
    fig, ax = plt_obj.subplots(rows, cols, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
