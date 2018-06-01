import os
import time

import pandas
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from car_model import get_cars_model
from vgg_model import IMAGE_SIZE

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    "data/validation",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

model = get_cars_model()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

save_model = ModelCheckpoint("checkpoints/weights_" + str(int(time.time())) + ".E{epoch:03d}-{val_loss:.3f}.hdf5",
                             monitor="val_loss", verbose=1, save_best_only=True, mode="auto")

# Train the model
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    callbacks=[save_model],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

# Save training results
pandas.DataFrame(hist.history).to_hdf(
    os.path.join("models", "history_{}.h5".format(str(time.time()))), "history")
