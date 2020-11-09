from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(123)  # for reproducibility

import collections
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler

input_scaler = MinMaxScaler(feature_range = (0.01, 0.99))
output_scaler = MinMaxScaler(feature_range = (0.01, 0.99))

###################################
#CREATE INPUTS AND OUTPUTS FROM DATA
###################################

# number of datasets
num_datasets = 6
# number of data to use per dataset (ensure each dataset has equal or more data points than this value)
#num_dataToUse = 2445
# number of timesteps per sample.
num_timesteps = 10
# number of epochs used for training
EPOCHS = 30000
# indicate whether dataset is from lidar scans or rgbd, default is lidar
lidar = True

if lidar:
    num_features = 363
else:
    num_features = 15


train_inputs = np.zeros((1, num_features))
train_outputs = np.zeros((1, 4))

for i in range(0, num_datasets):
    data_name = 'data_collection/dataset' + str(i + 1) + '.csv'
    dataset = pd.read_csv(data_name, header=None)
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset = dataset.dropna()
    dataset = dataset.values

    print("Concatenating dataset #{} ".format(i+1))
    if lidar:
        for j in range(0, dataset.shape[0]):
            train_inputs = np.vstack((train_inputs, dataset[j, 0:num_features]))
            train_outputs = np.vstack((train_outputs, dataset[j, -4:]))
    else:
        for j in range(0, dataset.shape[0]):
            train_inputs = np.vstack((train_inputs, dataset[j, 0:num_features]))
            output = np.hstack((dataset[j, -9], dataset[j, -8], dataset[j, -6], dataset[j, -5]))
            train_outputs = np.vstack((train_outputs, output))


"""
# all data is concatenated into a 3D vector, based on how many datasets are currently in the data_collection folder
train_inputs = np.zeros((num_datasets, num_dataToUse, num_features))
train_outputs = np.zeros((num_datasets, num_dataToUse, 4))
for i in range(0, num_datasets):
    data_name = 'data_collection/dataset' + str(i+1) + '.csv'
    dataset = pd.read_csv(data_name, header=None)
    dataset = dataset.values

    for j in range(0, dataset.shape[1]-4):
        train_inputs[i, :, j] = dataset[0:num_dataToUse, j]
    ind = 0

    for z in range(dataset.shape[1]-4, dataset.shape[1]):
        train_outputs[i,:,ind] = dataset[0:num_dataToUse, z]
        ind+=1
"""

# datasets are converted into a single dataset for scalar transformation
#train_inputs = train_inputs.reshape((-1, num_features))
#train_outputs = train_outputs.reshape((-1, 4))
train_inputs  = input_scaler.fit_transform(train_inputs)
train_outputs = output_scaler.fit_transform(train_outputs)

# new dataset is created, which uses the previous datasets and splits it into several datasets or samples, where each
# sample represents the number of timesteps to be predicted/trained on at a time. Ex, if the MPC prediction horizon is
# 16, then each sample will have 16 timesteps.
num_samples = int(np.floor(train_inputs.shape[0]/num_timesteps))
train_inputsFinal = np.zeros((num_samples, num_timesteps, num_features))
train_outputsFinal = np.zeros((num_samples, num_timesteps, 4))
index = 0
for i in range(0, num_samples):
    train_inputsFinal[i, :, :] = train_inputs[index:index+num_timesteps, :]
    train_outputsFinal[i, :, :] = train_outputs[index:index+num_timesteps, :]
    index += num_timesteps

##Save Scaler##
from pickle import dump
dump(output_scaler, open('rnn_models/covariance_scalar.pkl', 'wb'))
print("Data Successfully Concatenated.")

###################################
####TRAIN THE RNN####
###################################

ACTIVATION_1 = 'relu'
model = tf.keras.Sequential()
model.add(layers.SimpleRNN(256, input_shape=(num_timesteps, num_features), activation=ACTIVATION_1, return_sequences=True))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation=ACTIVATION_1))
#model.add(layers.Reshape((100, 256)))
model.add(layers.SimpleRNN(128, input_shape=(None, None), activation=ACTIVATION_1, return_sequences=True))
#model.add(layers.Dropout(0.1))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(32, activation=ACTIVATION_1))
#model.add(layers.Reshape((100, 32)))
model.add(layers.SimpleRNN(32, activation=ACTIVATION_1, return_sequences=True))
model.add(layers.SimpleRNN(16, activation=ACTIVATION_1, return_sequences=True))
#model.add(layers.SimpleRNN(8, activation=ACTIVATION_1, return_sequences=True))
model.add(layers.SimpleRNN(4, activation=ACTIVATION_1, return_sequences=True))

# try either sgd or adamax
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['accuracy'])
model.summary()
# batch size = the number of samples? (samples, timesteps, features)
history = model.fit(train_inputsFinal, train_outputsFinal, batch_size=32, epochs=EPOCHS, verbose=2, shuffle=False)

if lidar:
    model_name = "rnn_models/pf_SLAM.h5"
else:
    model_name = "rnn_models/vio_SLAM.h5"

model.save(model_name)

###################################
#####PLOT LOSS######
###################################
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Trained RNN: Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Test'], loc='upper left')
plt.show()

###################################
####SHOW RESULTS####
###################################

####### THIS CODE IS USED TO MAKE PREDICTIONS #######
model = tf.keras.models.load_model(model_name)

RNN_output = np.zeros((num_samples, num_timesteps, 4))
for i in range(0, num_samples):
    Measurements = train_inputsFinal[i, :, :]
    RNN_input = Measurements[np.newaxis,:]
    RNN_output_predict = model.predict(RNN_input)[0]
    RNN_output[i,:,:] = RNN_output_predict

RNN_output = RNN_output.reshape((-1,4))
RNN_output = output_scaler.inverse_transform(RNN_output)

###### Demonstrate Functionality ######
truth_output = train_outputsFinal
truth_output = truth_output.reshape((-1,4))
truth_output = output_scaler.inverse_transform(truth_output)

plt.plot(truth_output[:,0])
plt.plot(RNN_output[:,0])
plt.legend(["Truth", "Prediction"])
plt.title("Truth vs Prediction Covariances, xx")
plt.ylabel('covariance (m^2)')
plt.xlabel('timestep')
plt.show()

plt.plot(truth_output[:,1])
plt.plot(RNN_output[:,1])
plt.legend(["Truth", "Prediction"])
plt.ylabel('covariance (m^2)')
plt.xlabel('timestep')
plt.title("Truth vs Prediction Covariances, xy")
plt.show()

plt.plot(truth_output[:,2])
plt.plot(RNN_output[:,2])
plt.ylabel('covariance (m^2)')
plt.xlabel('timestep')
plt.legend(["Truth", "Prediction"])
plt.title("Truth vs Prediction Covariances, yx")
plt.show()

plt.plot(truth_output[:,3])
plt.plot(RNN_output[:,3])
plt.ylabel('covariance (m^2)')
plt.xlabel('timestep')
plt.legend(["Truth", "Prediction"])
plt.title("Truth vs Prediction Covariances, yy")
plt.show()

# save the different between truth and prediction for evaluation of constant difference (if it exists)
exx = truth_output[:,0]-RNN_output[:,0]
exy = truth_output[:,1]-RNN_output[:,1]
eyx = truth_output[:,2]-RNN_output[:,2]
eyy = truth_output[:,3]-RNN_output[:,3]

error_dataset = np.vstack((exx, exy, eyx, eyy)).reshape(exx.shape[0], 4)
np.savetxt("ErrorDataset_rnn_pf.csv", error_dataset, delimiter=",", header="xx, xy, yx, yy")
