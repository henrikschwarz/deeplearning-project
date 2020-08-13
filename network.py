# Names of Group 1 participants:
#   Isabella Junker Hacke, Gustav Nicolay Meilby Nobel, Henrik Schwarz,
#   Thobias Moldrup Sahi Aggerholm, Rasmus Boegeholt Vandkilde.

from silence_tensorflow import silence_tensorflow # Tensorflow muzzle
silence_tensorflow() #SSSHHHHHHH!!!!!!

# Task 2 - Creating the data generators
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import layers, Sequential, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "localhost:8888"])
url = tb.launch()


import datetime             # datetime module that can handle readable strings for TensorBoard.
# Create logging and tensorboard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Data generation
img = ImageDataGenerator(rescale=1./255) # Normalizes RGB values.
# Data augmentation techniques such as crop, pad, zoom, squeeze, rotate
#   and horizontal flip are commonly used to train large neural networks
train_img = ImageDataGenerator(rescale=1./255, zoom_range = 0.2, width_shift_range=0.1,height_shift_range=0.1, rotation_range=360, horizontal_flip=True, vertical_flip=True) #augments data
# Takes a path to a directory and generates batches of augmented data
trainGenerator = train_img.flow_from_directory(
    directory="./dataset_catdog/dataset/training",  # Defines the directory where the training data is located
    target_size=(150,150),                          # Down scales the images to 150px * 150px
    classes= ['cats', 'dogs'],                      # Identifyes the classes used
    color_mode="rgb",                               # Defines that the pictures are colored in the RGB colour space
    class_mode="binary",                            # Defines that the labels should be binary
    shuffle=True,                                   # Shuffles the images so they are nopt in a specific order
    batch_size= 32
    )

# Same as above but for val data
validationGenerator = img.flow_from_directory(
    directory="./dataset_catdog/dataset/validation",
    target_size=(150,150),
    classes= ['cats', 'dogs'],
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    batch_size= 32
    )

# Uses matplotlib to print viz of data fed to the model
image=trainGenerator.next()

print(image[1])
plt.subplot(221)
plt.imshow(image[0][0])
plt.subplot(222)
plt.imshow(image[0][1])
plt.subplot(223)
plt.imshow(image[0][2])
plt.subplot(224)
plt.imshow(image[0][3])
plt.show()

# Task 3 - Constructing the network
#   A Convolutional Neural Networks (CNN) is used in this binary image classification project.
#   First layer is 'Input'. A Matrix of pixel values in the shape of [WIDTH, HEIGHT, CHANNELS].
#   Second layer is convolutional. The purpose is to create a feature map from which we extract features using kernels,
#   starting with low-level reaching high-level features the deeper into the CNN we go.
#   Final output layer usually use the sigmoid activation function, so this is our starting point.

# Activate GPU
tf.config.set_soft_device_placement(True)

# Early stopping stops the model runtime when loss
#   and accuracy have diverged for a number of epochs
callback = callbacks.EarlyStopping(
    monitor='val_loss',             # monitor 'val_loss'
    patience=20                     # no. of diverging epochs with no improvement
    )

# Our CNN model
model= Sequential() # Init a sequential model from Keras

# The following is our best model so far
'''
# model.add(layers.Dense(64, activation='relu'))

model.add(layers.Conv2D(32,(3,3), activation='relu',input_shape=(150,150,3))) # 32 filters, kernel_size
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.add(layers.MaxPooling2D((2,2)))
'''
'''
model.add(layers.Dense(128,activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32,activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1, activation='sigmoid'))
'''

# Current model
# First conv layer
model.add(layers.Conv2D(
    filters=64,                 # no. of filters
    kernel_size=(4,4),          # kernel_size(x,x)
    activation='relu',          # Activation function used in layer
    input_shape=(150,150,3))    # image_size 150x150pixels x3 RGB colour
    )

# model.add(layers.Dense(64, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))                   # Pooling the input from .conv2d
model.add(layers.Conv2D(128,(3,3), activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Conv2D(256,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(512,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())                             # Vectorizes I/O

#
model.add(layers.Dense(32,activation='relu'))           # Fully connected layer (FFN) takes input from the former convolution
model.add(layers.Dense(1, activation='sigmoid'))        # Output layer for binary classification
model.summary()                                         # prints layers, output shape and params

# Compiling model
model.compile(
    optimizer='adam',           # utilizes the ADAM adaptive learning rate
    loss='binary_crossentropy', # the loss function
    metrics=['acc']             # Measures model accuracy
    )





# Fitting model
with tf.device("/device:GPU:0"): # Following piece of code is to run on GPU
    model.fit(
        trainGenerator,                  # Model to train
        validation_data=(validationGenerator),  # Data on which to evaluate the loss at end of epochs
        epochs=100,                             # no. of epochs
        verbose=1,                              # Amount of information during run. 1 is progress bar
        callbacks=[callback, tensorboard_callback]                    # list of callbacks to do in between epochs
        )

# Task 4 - Visualizing your results
# Below taken from: https://keras.io/examples/vision/image_classification_from_scratch/
'''
pred = model.prediction(validationGenerator[0])
pred_index = np.argmax(pred, axis = 1)
test_index = np.
'''
