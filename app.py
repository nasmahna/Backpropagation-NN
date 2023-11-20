from pickle import NONE
#Import Library

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf
import random
import flask
import os

from math import exp

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score

from tensorflow import keras
from keras import layers
from keras.layers import Dense

from flask import Flask, render_template, jsonify, request


app = Flask(__name__)
app.secret_key = 'bpnn'
model = ''

#Data understanding, global variabel intialization\
train_ratio = 0.8
epochs = 100
batch_size = 50
lr = 0.0001
atr_amounts = 5
neurons = 5
activation = 'sigmoid'


# len_traindata = None
# saveloc = None
# status_training_done = None


model = None 
# valid = None
# mape = None
# accuracy = None

@app.route('/',methods = ['GET'])
def index():
     return render_template('index.html')

#Upload Data
@app.route('/input', methods = ['GET', 'POST'])
def input():
    global loc
    if request.method =='POST':
        uploaded = request.files['uploaded']
        loc = os.path.join("upload", str(request.files['uploaded'].name) + '.csv')
        uploaded.save(loc)
        return render_template('index.html')
    return render_template('input.html')

# df = pd.read_csv(loc)

@app.route('/lihat', methods = ['GET'])
def lihat():
    global loc
    df = pd.read_csv(loc)
    timeseries_df = create_dataset(df["Data"].values, 6)
    return render_template('lihat.html', df=df, timeseries_df=timeseries_df)

# def read_data():
#     df = pd.read_csv('/Users/macbookair/Documents/Proyek Informatika /JST-Backpropagation-Backup/datasets/dataipm.csv')
#     df.head(10)

# def set_len_traindata(df):
#   global len_traindata
#   len_traindata = math.ceil(len(df) * .8) #.8 is a train ratio

# def set_saveloc(data_dir):
#   global saveloc
#   saveloc = data_dir

# def set_training_status():
#   global status_training_done
#   status_training_done = True

# convert an array of values into a dataset matrix
def create_dataset(arr, look_back=1):
    data = []
    for e in range(len(arr)):
        lookback_data = arr[e:look_back+e]
        if len(lookback_data) == look_back:
            data.append(arr[e:look_back+e])
    columns = ["X"+str(i) for i in range(1,look_back)]
    columns.append("Target")
    return pd.DataFrame(data, columns=columns)

@app.route('/grafik', methods = ['GET'])
def grafik():
    global loc
    df = pd.read_csv(loc)
    df.set_index("Tahun", inplace=True)
    plt.plot(df)
    plt.title("Data IPM")
    plt.xlabel("Tahun")
    plt.ylabel("Data")
    plt.savefig('static/assets/plot-data.png')
    return render_template('grafik.html') 

@app.route('/scalling', methods = ['GET'])
def scalling():
    global loc
    #Normalisasi dengan MinMax
    df = pd.read_csv(loc)
    timeseries_df = create_dataset(df["Data"].values, 6)
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_scaled = minmax_scaler.fit_transform(np.array(timeseries_df))
    return render_template('scalling.html', minmax_scaled=minmax_scaled)

def prep_data(atr_amounts):
    global loc
    global train_ratio
    df = pd.read_csv(loc)
    #Preparing Train Data
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_scaler = minmax_scaler.fit(df)
    minmax_scaled = minmax_scaler.transform(df)

    #Length of Train Data
    len_datatrain= math.ceil(len(df)* train_ratio)

    #Train data to be scaled process
    train_data = minmax_scaled[0:len_datatrain, :]

    #Splitting for training data
    X_train = [] #1D list
    y_train = []
    for i in range(atr_amounts, len(train_data)):
        X_train.append(train_data[i - atr_amounts: i,0])
        y_train.append(train_data[i, 0])

    #Coverting x & y train to be arrays in numpy
    X_train, y_train = np.array(X_train), np.array(y_train)

    #Scaling data for backprop
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    #Prep Test Data
    test_data = minmax_scaled[len_datatrain - atr_amounts:, :]

    #Splitting for Data test
    X_test = []
    y_test = df.iloc[len_datatrain:, :]
    for i in range(atr_amounts, len(test_data)):
        X_test.append(test_data[i-atr_amounts:i,0])

    #Converting X_test to be array in numpy
    X_test = np.array(X_test)

    #Scaling data
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    return X_train, y_train, X_test, y_test, minmax_scaler

def mse(actual, predicted):
    actual = np.array(actual, dtype="float64")
    predicted = np.array(predicted, dtype="float64")
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()

def mape(actual, pred): 
    actual = np.array(actual, dtype="float64")
    pred = np.array(pred, dtype="float64")
    return np.mean(np.abs((actual - pred) / actual)) * 100

def training(X_train, y_train, neurons, atr_amounts, activation):
    random.seed(10)
    tf.random.set_seed(10)

    #kernel initializer digunakan untuk inisialisasi bobot
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(neurons, input_shape=(atr_amounts,), activation=activation, kernel_initializer=tf.keras.initializers.HeUniform(seed=10)),
            tf.keras.layers.Dense(1)])

    #Compiling Models
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=["MAE"]) #matriks to evaluate model 

    #Training Model
    history = model.fit(X_train,
                        y_train,
                        verbose=1,
                        epochs=100,
                        batch_size=32)
    return (model, history)

@app.route('/latih', methods = ['GET'])
def latih():
    global loc
    global df
    global atr_amounts
    global activation
    global train_ratio
    global neurons
    global model
    df = pd.read_csv(loc)
    X_train, y_train, X_test, y_test, minmax_scaler = prep_data(atr_amounts)
    model, history = training(X_train, y_train, neurons, atr_amounts, activation)
    return render_template('latih.html')

def testing(X_train, X_test, y_test, minmax_scaler):
    global df
    global model

    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    minmax_scaler = minmax_scaler.fit(df)

    pred = model.predict(X_test)
    pred = minmax_scaler.inverse_transform(pred)

    res_mse = []
    for i,j in zip(np.array(y_test), pred):
        mse = mean_squared_error(i,j)
        res_mse.append(mse)

    len_datatrain= math.ceil(len(df)* train_ratio)

    #Plotting Prediction Values
    train = df[:len_datatrain]
    valid = df[len_datatrain:]
    valid["Prediction"] = pred
    valid['MSE'] = res_mse 

    #Graph Predict
    plt.figure(figsize=(15,5), facecolor='#464745')
    plt.xlabel('Tahun', fontsize=15)
    ax = plt.axes()
    ax.set_facecolor('#464745')
    plt.ylabel('Data', fontsize=15)
    plt.plot(train['Data'])
    plt.plot(valid[['Data', 'Prediction']])
    plt.legend(['Train', 'Test', 'Prediction'], loc='lower left')
    plt.rcParams.update({
        'text.color': '#464745',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white'
        })
    plt.savefig('static/assets/res_tmp.png')

    #MAPE and Accuracy
    mape = mean_absolute_percentage_error(y_test, pred)   #validation
    acc = round(100 - mape, 3)

    updated_data = []
    for i in range(0, len(X_test[-1])):
        if i == 0:
            pass
        else:
            updated_data.append(X_test[-1][i])
    updated_data.append([0])
    updated_data = np.array(updated_data)

    X_test = np.append(X_test, [updated_data], axis=0)
    X_test = np.asarray(X_test).astype(np.float32)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    updated_pred = model.predict(X_test)
    updated_pred = minmax_scaler.inverse_transform(updated_pred) 

    #Reading the result data after upgrading process on prediction using data testing
    rest_predict = updated_pred[-1]
    return mape, acc, rest_predict


@app.route('/uji', methods = ['GET'])
def uji():
    global loc
    global df
    global atr_amounts
    global activation
    global train_ratio
    global neurons
    global model
    df = pd.read_csv(loc)
    X_train, y_train, X_test, y_test,minmax_scaler = prep_data(atr_amounts)
    
    model, history = training(X_train, y_train, neurons, atr_amounts, activation)
    mape, acc, rest_predict = testing(X_train, X_test, y_test, minmax_scaler)
    return render_template('uji.html', mape=mape, acc=acc, rest_predict=rest_predict)

app.run(host='127.0.0.1', debug=True)