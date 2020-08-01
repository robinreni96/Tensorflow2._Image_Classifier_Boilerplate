import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



def build_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES):
    
    # Model Structure 
    base_model = tf.keras.applications.ResNet50V2(input_shape=(IMG_HEIGHT,IMG_WIDTH,3),include_top=False,weights='imagenet')

    
    base_model.trainable = True
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    dense_layer = tf.keras.layers.Dense(512, activation='relu')
    
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")
    
    model_new = tf.keras.Sequential([base_model, global_average_layer, dense_layer, prediction_layer])
    
    # Model Hyperparameters
    base_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=['accuracy']
    
    model_new.compile(optimizer, loss, metrics ) 
    
    model_new.summary()
    
    print("<--------- Model Build Successful ------------> \n")
    
    return model_new
    