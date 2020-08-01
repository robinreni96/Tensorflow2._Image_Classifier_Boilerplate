import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Prediction_Model(): 
    
    def __init__(self): 
        self.saved_model_path = "models/saved_model"
        self.labels = ['Bathroom','BedRoom','Dinning','Kitchen','LivingRoom'] 
        self.infer_model = self.load_model()
        
        
    def load_model(self):
        
        new_model = tf.saved_model.load(self.saved_model_path)
        
        infer_model = new_model.signatures["serving_default"]
        
        return new_model

    
    def predict(self, path):
        
        img = self.load_image(path)
        
        img = img.astype(np.float32)
        x = np.array([img])
        
        model = self.infer_model.signatures["serving_default"]
        
        predict_scores = model(tf.constant(x))
        
        confidence_score = np.max(predict_scores['dense_1'])
        
        decoded_label = self.labels[np.argmax(predict_scores)]
        
        return decoded_label, confidence_score
        

    def load_image(self, img_path): 
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=[224, 224])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x /= 255
        return x



if __name__ == "__main__": 
    path = "sample/sample_2.jpg"
    
    p = Prediction_Model()
    
    label, score = p.predict(path)
    
    print("Label : {}".format(label))
    print("Score : {}".format(score))


