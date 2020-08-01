'''Generate TFRecord for the Image Files '''
from random import shuffle
import glob
import os
import cv2
import sys
import io
from PIL import Image
import numpy as np

import tensorflow as tf
shuffle_data = True  # shuffle the addresses before saving



def process_dir(source_dir):
    # Load the paths
    dir_paths = os.path.join(source_dir,"*/*.jpg")
    addrs = glob.glob(dir_paths)
    
    # Map the labels
    label = {'Bathroom':0, 'Bedroom':1, 'Dinning':2, 'Kitchen':3, 'Livingroom':4}
    map_label = lambda x: label[x.strip().split("/")[2]]
    labels = list(map(map_label, addrs))  # 0 = Cat, 1 = Dog

    # shuffle data
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)
    
    return addrs, labels


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    # img = cv2.imread(addr)
    # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    image_string = open(addr, 'rb').read()
    return image_string

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(addrs,labels,mode):
    
    tfrecord_dir = 'data/tfrecords/'
    
    if not os.path.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)
        
    filename = tfrecord_dir + '{}.tfrecords'.format(mode)  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(filename)
    for i in range(len(addrs)):

        # Load the image
        img = load_image(addrs[i])
        label = labels[i]
        # Create a feature
        feature = {'label': _int64_feature(label),
                'image_raw': _bytes_feature(img)}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()
    
    return filename

if __name__ == "__main__":
    train_dir = "data/train/"
    val_dir = "data/val/"
    
    # process train images
    train_paths, train_labels = process_dir(train_dir)
    f = write_tfrecord(train_paths, train_labels, "train")
    print("TFRecord Successful and saved in {}".format(f))
    
     # process val images
    val_paths, val_labels = process_dir(val_dir)
    f = write_tfrecord(val_paths, val_labels, "val")
    print("TFRecord Successful and saved in {}".format(f))
    
    