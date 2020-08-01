from __future__ import absolute_import, division, print_function, unicode_literals
import time
import datetime
import os 
import shutil
import json

from read_tfrecord import read_tf
from model import build_model

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import matplotlib.pyplot as plt

# Clear any logs from previous runs
if os.path.exists("log/"):
    shutil.rmtree("log/")

batch_size = 16
epochs = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 5

class Appvision_Training(object):

    def __init__(self):
        
        self.log = "logs/fit/"
        self.train_tfrec = "data/tfrecords/train.tfrecords"
        self.val_tfrec = "data/tfrecords/val.tfrecords"
        self.best_model_path = "models/checkpoint/best_model.ckpt"
        model_version = 1
        self.saved_model_path = os.path.join("models/saved_model",str(model_version))
        image_class = json.load(open('image_class_index.json'))
        self.labels = list(image_class.values())
        
    def visualize_sample(self, data_gen):

        sample_training_images, _ = next(data_gen)

        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip( sample_training_images[:5], axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


    def visualize_model_perform(self, history):

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def saved_model(self):


        
        # # Build the model
        model = build_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

        # # Load the best Model
        model.load_weights(self.best_model_path)
        

        tf.saved_model.save(
                model,
                self.saved_model_path
            )

        print("Converted to Saved Model")


    def load_model(self):

        new_model = tf.saved_model.load(self.saved_model_path)

        print(list(new_model.signatures.keys()))

        infer = new_model.signatures["serving_default"]
        print(infer.structured_outputs)

        return new_model, infer


    def evaluate_model(self, new_model, infer):
        # Data Loaders Training and Validation
        val_images, val_labels = read_tf(val_tfrec, NUM_CLASSES)

        # Evaluate the restored model
        # loss, acc = model.evaluate(val_images,  val_labels, verbose=2)
        # print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
        # print(val_images[0])
        for x in val_images:
            x = x.astype(np.float32)
            x = np.array([x])

            infer = new_model.signatures["serving_default"]
            label = infer(tf.constant(x))
            print(label)
            # labeling = infer(tf.constant(x))[new_model.outputs]
            # print(labeling)
            # decoded = labels[np.argmax(label)]
            # print("Result after saving and loading:\n", decoded)
            break
        # print(model.predict(val_images).shape)

    def load_data(self):

        # Loading the train dataset
        train_images, train_labels = read_tf(self.train_tfrec, NUM_CLASSES)

        print("<-------- Training TFRecords Loaded Successfully ------------> \n")

        # Loading the val dataset
        val_images, val_labels = read_tf(self.val_tfrec, NUM_CLASSES)

        print("<--------- Validation TFRecords Loaded Successfully ----------> \n")

        total_train = len(train_images)
        total_val = len(val_images)

        print("Total training images:", total_train)
        print("Total validation images:", total_val)


        # Create the data augumentor 
        train_image_generator = ImageDataGenerator(
                                                    rotation_range=20,
                                                    width_shift_range=0.2,
                                                    height_shift_range=0.2,
                                                    horizontal_flip=True,rescale=1./255) # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

        # Load the generator to arrays

        train_data_gen = train_image_generator.flow(train_images, train_labels, batch_size=batch_size,shuffle=True )

        validation_data_gen = validation_image_generator.flow(val_images, val_labels, batch_size=batch_size,shuffle=True )

        # Visualize some of the images
        self.visualize_sample(train_data_gen)


        return train_data_gen, validation_data_gen, total_train, total_val


    def run_training(self):

        # Data Loaders Training and Validation
        train_data_gen , val_data_gen, total_train, total_val = self.load_data()

        # Build the Model
        model = build_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
            
        # Tensorboard Integration
        log_dir= self.log + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Early Stopping 
        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)

        # Optmized Checkpoint
        model_path = self.best_model_path
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, mode='max', monitor='val_accuracy', verbose=1, save_weights_only=True, save_best_only=True)

        # Callbacks
        callbacks = [tensorboard_callback, earlystop_callback, checkpoint_callback]

        # Initate Training
        model.fit_generator(train_data_gen,steps_per_epoch=total_train // batch_size,
                                            epochs=epochs,validation_data=val_data_gen,
                                            validation_steps=total_val // batch_size,
                                            callbacks = callbacks
                                            )

        # # Visualize Model Performance
        # self.visualize_model_perform(history)

        print("Model Trained Successfully . The best model is available models/checkpoint/best_model.ckpt")



if __name__ == "__main__":
    
    app = Appvision_Training()
    
    app.run_training()
    
    app.saved_model()