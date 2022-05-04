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

test_RMSEs = np.empty([5,1])
train_RMSEs = np.empty([5,1])
filter = 20
dim = 8

#%%
for i in [0,1,2,3,4]:
#First we clean the session to make sure that our computer will only be running one ANN
    tf.keras.backend.clear_session()

# Initialize the Artificial Neural Network (ANN) and build the Graph.
    model = tf.keras.models.Sequential([
                        tf.keras.layers.Input(shape=(x_train.shape[1],x_train.shape[2])),# Shape [batch, time, features] => [batch, time, lstm_units]
                        tf.keras.layers.Reshape((x_train.shape[1],x_train.shape[2],1)),
                        #tf.keras.layers.Conv2D(16, (1,x_train.shape[2]), activation='relu'), #(feature maps, kernel size, activation function)
                        tf.keras.layers.Conv2D(filter, (1,dim), padding="valid", activation='relu'), #(feature maps, kernel size, activation function)
                        
                        tf.keras.layers.Reshape((30,filter*(x_train.shape[2]-dim+1))),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        #tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.LSTM(64, return_sequences=True),
                        #tf.keras.layers.Dropout(0.2),


                        tf.keras.layers.Flatten(), 
                        tf.keras.layers.Dense(units=1)
                    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.003),
            loss=tf.keras.losses.MeanSquaredError(),
                )

    model_history = model.fit(x_train, 
                          y_train, 
                          batch_size = 128, 
                          epochs = 70, 
                          validation_split = 0.15, 
                          verbose=2)


    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Loss Plot for Model Run #{}'.format(i+1))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Predicting the Test set results. 
    # CLassification/Diagnostics
    y_pred_train = model.predict(x_train).reshape(-1,1)
    y_pred_test = model.predict(x_test).reshape(-1,1)

    rmse_train = np.sqrt(np.mean(np.square(y_pred_train - y_train.reshape(-1,1))))
    rmse_test = np.sqrt(np.mean(np.square(y_pred_test - y_test.reshape(-1,1))))

    test_RMSEs[i]=rmse_test
    train_RMSEs[i]=rmse_train
    print(rmse_train)
    print(rmse_test)
    
    plt.figure(figsize=(7,7))
    plt.scatter(y_test,y_pred_test)
    plt.plot([0,1.01*y_test.max()],[0,1.1*y_test.max()], 'r')
    plt.title('Model Predictions for Model Run #{}'.format(i+1))
    plt.show()

    

print('Train RMSE:',np.average(train_RMSEs))
print('Test RMSE:',np.average(test_RMSEs))

print('Train RMSE Standard Deviation:',np.std(train_RMSEs))
print('Test RMSE Standard Deviation:',np.std(test_RMSEs))


# %%
