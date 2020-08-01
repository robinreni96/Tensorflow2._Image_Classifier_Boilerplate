import tensorflow as tf
import numpy as np

def _parse_image_function(example_proto):
    
     # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Parse the input tf.Example proto using the dictionary above.
    parsed_features =  tf.io.parse_single_example(example_proto, image_feature_description)
    
    parsed_features['image_raw'] = tf.io.decode_image(parsed_features['image_raw'])
    
    parsed_features['image_raw'] = tf.reshape(parsed_features['image_raw'], [224, 224, 3])
    
    return parsed_features
    

def read_tf(tfrecord_path, NUM_CLASSES):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    parsed_dataset = raw_image_dataset.map(_parse_image_function)
    
    image_raw = []
    label = []
    
    for image_features in parsed_dataset:
        temp_image = image_features['image_raw'].numpy()
        image_raw.append(temp_image)
        temp_label = tf.one_hot(image_features['label'], 5)
        label.append(temp_label)
    
    return np.array(image_raw), np.array(label)