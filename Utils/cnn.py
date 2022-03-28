import time
import json
import logging as log
import sys
import datetime

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt
# import keras
import tensorflow as tf
print(tf.__version__)

# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

np.random.seed(0)
tf.random.set_seed(0)

def gen_net(model_name, weights, freeze, not_train, top = False):
    '''
    :param model_name: vgg, vgg19, inception
    :param weights: True/False
    :param pretrain: True/False
    :return:
    '''
    
    if (model_name == "vgg"):
        if (weights == True):
            model = VGG16(
                input_shape=(128,128,3), 
                weights = 'imagenet', 
                include_top = False)
        else:
            model = VGG16(
                input_shape=(128,128,3), 
                weights = None, 
                include_top = False)
            
    if (model_name == "vgg19"):
        if (weights == True):
            model = VGG19(
                input_shape=(128,128,3), 
                weights = 'imagenet', 
                include_top = False)
        else:
            model = VGG19(
                input_shape=(128,128,3), 
                weights = None, 
                include_top = False)
    
    if (model_name == "inception"):
        if (weights == True):
            model = InceptionV3(
                input_shape=(128,128,3), 
                weights = 'imagenet', 
                include_top = False)
        else:
            model = InceptionV3(
                input_shape=(128,128,3),
                weights = None, 
                include_top = False)

    # This section has to be modified in order a pre defined number of layers
    if (freeze):
      if (top):
        for layer in model.layers[:not_train]:
            layer.trainable = False
        else:
          for layer in model.layers[not_train:]:
            layer.trainable = False

    # Adding Last layers
    base_outputs = model.output
    x = Flatten()(base_outputs)
    x = Dense(1)(x)

    new_model = Model(inputs = model.input, outputs = x) 
    new_model.summary()

    # Make sure you have frozen the correct layers
    for i, layer in enumerate(new_model.layers):
        print(i, layer.name, layer.trainable)

    return new_model


class network_fit(object):
    '''
    Class for network managing. 
    Training, validation and testing methods are implemented here.
    '''

    def __init__(self, train_samples, label_array_train, test_samples, label_array_test,
                 model_path, model_name, weights, freeze, not_train, verbose=1, top = False):
        '''
        Constructor
        Generate a CNN and train
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.train_samples = train_samples
        self.label_array_train = label_array_train
        self.test_samples = test_samples
        self.label_array_test = label_array_test
        self.model_path = model_path
        self.verbose = verbose
        self.top = top

        # self.mlps = gen_net(self.train_samples.shape[1], self.n_hidden1, self.n_hidden2)
        self.mlps = gen_net(model_name, weights, freeze, not_train, top)



    def train_net(self, epochs, batch_size, lr= 1e-05, plotting=True):
        '''
        Training workflow

        :param epochs:
        :param batch_size:
        :param lr:
        :return: trained net and stop epoch
        '''
        print("Initializing network...")
        # compile the model
        rp = optimizers.RMSprop(learning_rate=lr, rho=0.9, centered=True)
        adm = optimizers.Adam(learning_rate=lr, epsilon=1)
        sgd_m = optimizers.SGD(learning_rate=lr)

        keras_rmse = tf.keras.metrics.RootMeanSquaredError()
        self.mlps.compile(loss='mean_squared_error', optimizer=adm, metrics=[keras_rmse, 'mae'])
        
        log_dir = current_dir + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        

        # Train the model
        history = self.mlps.fit(self.train_samples, self.label_array_train, epochs=epochs, batch_size=batch_size,
                                validation_split=0.2, verbose=self.verbose,
                                callbacks=[
                                    tensorboard_callback,
                                    EarlyStopping(monitor='val_root_mean_squared_error', 
                                                  min_delta=0.1, 
                                                  patience=20, 
                                                  verbose=self.verbose, 
                                                  mode='min'),
                                    ModelCheckpoint(self.model_path, 
                                                    monitor='val_root_mean_squared_error', 
                                                    save_best_only=True, 
                                                    mode='min', 
                                                    verbose=self.verbose)])
       

        val_rmse_k = history.history['val_root_mean_squared_error']
        val_rmse_min = min(val_rmse_k)
        min_val_rmse_idx = val_rmse_k.index(min(val_rmse_k))
        stop_epoch = min_val_rmse_idx +1
        val_rmse_min = round(val_rmse_min, 4)
        print ("val_rmse_min: ", val_rmse_min)

        trained_net = self.mlps

        ## Plot training & validation loss about epochs
        if plotting == True:
            # summarize history for Loss
            fig_acc = plt.figure(figsize=(10, 10))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.ylim(0, 2000)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()


        return trained_net, stop_epoch



    def test_net(self, trained_net=None, best_model=True, plotting=True):
        '''
        Evalute the trained network on test set
        :param trained_net:
        :param best_model:
        :param plotting:
        :return: RMSE and Score
        '''
        # Load the trained model
        if best_model:
            estimator = load_model(self.model_path)
        else:
            estimator = load_model(trained_net)

        # predict the RUL
        y_pred_test = estimator.predict(self.test_samples)
        y_true_test = self.label_array_test # ground truth of test samples

        pd.set_option('display.max_rows', 1000)
        test_print = pd.DataFrame()
        test_print['y_pred'] = y_pred_test.flatten()
        test_print['y_truth'] = y_true_test.flatten()
        test_print['diff'] = abs(y_pred_test.flatten() - y_true_test.flatten())
        test_print['diff(ratio)'] = abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()
        test_print['diff(%)'] = (abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()) * 100

        y_predicted = test_print['y_pred']
        y_actual = test_print['y_truth']
        rms = sqrt(mean_squared_error(y_actual, y_predicted)) # RMSE metric
        test_print['rmse'] = rms
        print(test_print)


        # Score metric
        h_array = y_predicted - y_actual
        s_array = np.zeros(len(h_array))
        for j, h_j in enumerate(h_array):
            if h_j < 0:
                s_array[j] = math.exp(-(h_j / 13)) - 1

            else:
                s_array[j] = math.exp(h_j / 10) - 1
        score = np.sum(s_array)

        # Plot the results of RUL prediction
        if plotting == True:
            fig_verify = plt.figure(figsize=(12, 6))
            plt.plot(y_pred_test, color="blue")
            plt.plot(y_true_test, color="green")
            plt.title('prediction')
            plt.ylabel('value')
            plt.xlabel('row')
            plt.legend(['predicted', 'actual data'], loc='upper left')
            plt.show()

        return rms, score