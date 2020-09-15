from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predictCovariance(model_name, RNN_input, output_scaler):
    model = load_model(model_name)
    RNN_input = MPC_Generated_Measurements[np.newaxis,:] #Make compatible with RNN
    RNN_output = model.predict(RNN_input)[0]
    covariance_predictions = output_scaler.inverse_transform(RNN_output) #Undo Normalization
    return covariance_predictions;