from keras import layers
from keras import models

from vgg_model import vgg_conv

NUM_CLASSES = 3
WIDTH = 1024
DROPOUT = 0.5


def get_cars_model():
    # Create the model
    model = models.Sequential()
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(WIDTH, activation='relu'))
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model
