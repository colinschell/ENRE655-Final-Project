#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras.utils as np_utils


#%%
dataset = np.load('CMAPPS_FD001_nonsc.npz')
X_train_pre_img = dataset['x_train'] 
X_test_pre_img = dataset['x_test']

sc =  MinMaxScaler() # Tip: if your data set contains negative values use feature_range=(-1,1). Otherwise use feature_range=(0,1)
X_train_pre_img = sc.fit_transform(X_train_pre_img) # Always fit the scaler to the training dataset. For the test dataset, just transform it.
X_test_pre_img = sc.transform(X_test_pre_img) 

x_train = np.reshape(X_train_pre_img,[-1,30,14])
x_test = np.reshape(X_test_pre_img,[-1,30,14])

y_train = dataset['y_train'] 
y_test = dataset['y_test']

#hyperparameters
batch_size = [128, 256]
learning_rate = [0.001, 0.003]
nodes = [8, 16, 32, 64]
layers = [1, 2]
optimizer_types = ['adam', 'nadam']
epochs=40

LSTM_test_RMSEs =  [[[[[0 for x in batch_size] for x in nodes] for x in learning_rate] for x in layers] for x in optimizer_types]
LSTM_train_RMSEs = LSTM_test_RMSEs

test_RMSEs = np.empty([2,1])
train_RMSEs = np.empty([2,1])

#%%
counter=0
for b,batch in enumerate(batch_size): 
  for r,rate in enumerate(learning_rate): 
    for n,node in enumerate(nodes): 
      for l,layer in enumerate(layers): 
        for o,optimizer in enumerate(optimizer_types):
          for i in [0,1]: #model averaging for loop
              tf.keras.backend.clear_session()

              if layer == 1:
                model = tf.keras.models.Sequential([
                  tf.keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]

                  tf.keras.layers.LSTM(node, return_sequences=True),
              
                  tf.keras.layers.Flatten(), #uncomment in case return_sequences = True
                  tf.keras.layers.Dense(units=1)
              ])
                
              elif layer == 2:
                model = tf.keras.models.Sequential([
                  tf.keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]

                  tf.keras.layers.LSTM(node, return_sequences=True),
                  tf.keras.layers.LSTM(node, return_sequences=True),
              
                  tf.keras.layers.Flatten(), #uncomment in case return_sequences = True
                  tf.keras.layers.Dense(units=1)
              ])

              if optimizer == 'adam':
                model.compile(
                      optimizer=tf.keras.optimizers.Adam(rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                          )
                
              elif optimizer == 'nadam':
                model.compile(
                      optimizer=tf.keras.optimizers.Nadam(rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                          )

              model_history = model.fit(x_train, 
                                    y_train, 
                                    batch_size = batch, 
                                    epochs = epochs, 
                                    validation_split = 0.15, 
                                    verbose=0)

              # Predicting the Test set results. 
              # CLassification/Diagnostics
              y_pred_train = model.predict(x_train).reshape(-1,1)
              y_pred_test = model.predict(x_test).reshape(-1,1)

              rmse_train = np.sqrt(np.mean(np.square(y_pred_train - y_train.reshape(-1,1))))
              rmse_test = np.sqrt(np.mean(np.square(y_pred_test - y_test.reshape(-1,1))))

              test_RMSEs[i]=rmse_test
              train_RMSEs[i]=rmse_train

          #print('batch = ', batch, '| node = ', node, '| rate = ', rate, '| layers = ', layer, '| optimizer = ', optimizer) 

          LSTM_test_RMSEs[o][l][r][n][b] = np.average(test_RMSEs)
          LSTM_train_RMSEs[o][l][r][n][b] = np.average(train_RMSEs)

          counter+=1
          print('Finished run # ', counter, ' of ', len(batch_size)*len(learning_rate)*len(nodes)*len(layers)*len(optimizer_types), 
                '| This runs Test RMSE = ', LSTM_test_RMSEs[o][l][r][n][b])

#%%
test=LSTM_test_RMSEs
train=LSTM_train_RMSEs
test=list(np.concatenate(test).flat)
train=list(np.concatenate(train).flat)
testdf = pd.DataFrame(test)
testdf.to_csv('LSTMtestRMSEs.csv')
traindf = pd.DataFrame(train)
traindf.to_csv('LSTMtrainRMSEs.csv')
# %%
