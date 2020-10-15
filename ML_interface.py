from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import tensorflow as tf

class ML_interface():

    def __init__(self):
        pkl_file = open('rnn_models/covariance_scalar.pkl', 'r')
        self.output_scalar = pickle.load(pkl_file)
        pkl_file.close()

    def predictCovariance(self, model_name, measurements):
        model = tf.keras.models.load_model(model_name)
        RNN_input = measurements[np.newaxis,:] #Make compatible with RNN
        RNN_output = model.predict(RNN_input)[0]
        covariance_predictions = self.output_scalar.inverse_transform(RNN_output) #Undo Normalization
        return covariance_predictions

### EXAMPLE ###

"""
model_name = "rnn_models/pf_SLAM.h5"
measurements = np.ones((50, 360))
ML_interface = ML_interface()
output = ML_interface.predictCovariance(model_name, measurements)
"""