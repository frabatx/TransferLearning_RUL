# -*- coding: utf-8 -*-

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from itertools import repeat
from scipy.stats import randint, expon, uniform

import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import cv2
import io
from PIL import Image
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from sklearn import preprocessing


# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# random seed predictable
seed = 0
random.seed(seed)
np.random.seed(seed)

# Path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
## Dataset path
train_FD001_path = current_dir +'/cmapss/train_FD001.csv'
test_FD001_path = current_dir +'/cmapss/test_FD001.csv'
RUL_FD001_path = current_dir+'/cmapss/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

train_FD002_path = current_dir +'/cmapss/train_FD002.csv'
test_FD002_path = current_dir +'/cmapss/test_FD002.csv'
RUL_FD002_path = current_dir +'/cmapss/RUL_FD002.txt'
FD002_path = [train_FD002_path, test_FD002_path, RUL_FD002_path]

train_FD003_path = current_dir +'/cmapss/train_FD003.csv'
test_FD003_path = current_dir +'/cmapss/test_FD003.csv'
RUL_FD003_path = current_dir +'/cmapss/RUL_FD003.txt'
FD003_path = [train_FD003_path, test_FD003_path, RUL_FD003_path]

train_FD004_path =current_dir +'/cmapss/train_FD004.csv'
test_FD004_path = current_dir +'/cmapss/test_FD004.csv'
RUL_FD004_path = current_dir +'/cmapss/RUL_FD004.txt'
FD004_path = [train_FD004_path, test_FD004_path, RUL_FD004_path]

## Read csv file to pandas dataframe
FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
dp_str = ["none", "FD001", "FD002", "FD003", "FD004"]

## Setting column names
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
col_rul = ['RUL_truth']
# Sensors not to be considered (those that do not disclose any pattern in their ts)
sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']


def gen_sequence(id_df, seq_length, seq_cols):
    """ 
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones 
    """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.

    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """ 
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones 
    """
    # For one id I put all the labels in a single matrix.
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

## Concatenate Time Windows
def change_format_rect(X_rp):
    '''
    Takes 14 Time Window sensors array and reshapes into a RECTANGULAR array(224x64)
    '''
    temp_row = np.array([]).reshape(0,len(X_rp[0]))
    temp_array = np.array([]).reshape(len(X_rp[0])*7,0)
    for i in range(len(X_rp)):
        x = X_rp[i]
        temp_row = np.vstack([temp_row,x]) 
        if (i+1)%7 == 0:
            temp_array = np.hstack([temp_array, temp_row])
            temp_row = np.array([]).reshape(0,len(X_rp[0]))  
    new_train0 = temp_array
    plt.figure(figsize=(10, 10))
    plt.imshow(temp_array, origin='lower')
    plt.show()
    # 4- Extract RGB channels with cv2.split(img)
    ## To make a figure without the frame
    my_dpi = 96
    fig = plt.figure(figsize=(64/my_dpi, 224/my_dpi), dpi=my_dpi, frameon=False)
    ## To make the content fill the whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ## Drawing image
    ax.imshow(new_train0)
    fig.savefig('combinedrect.png', dpi = my_dpi)
    img = cv2.imread('combinedrect.png')
    plt.close(fig)
    return  np.asarray(cv2.split(img))
def change_format(X_rp):
    '''
    Takes 14 Time Window sensors array and reshapes into a SQUARE array(128x128)
    '''
    dummy_array = np.zeros((64,32))
    temp_row = np.array([]).reshape(0,len(X_rp[0]))
    temp_array = np.array([]).reshape(len(X_rp[0])*4,0)
    for i in range(len(X_rp)):
    # print ("temp_row.shape", temp_row.shape)
    # print ("temp_array.shape", temp_array.shape)        
        x = X_rp[i]
        # temp_row = np.concatenate((temp_row,x), axis=0)
        temp_row = np.vstack([temp_row,x])
        if (i+1)%4 == 0:
            temp_array = np.hstack([temp_array, temp_row])
            temp_row = np.array([]).reshape(0,len(X_rp[0]))
        if (i+1)==len(X_rp):
            temp_row = np.vstack([temp_row,dummy_array])
            temp_array = np.hstack([temp_array, temp_row])
            temp_row = np.array([]).reshape(0,len(X_rp[0]))   
        
        #Remove comment to show the concatenated image      
        #plt.figure(figsize=(10, 10))
        #plt.imshow(temp_array)
        #plt.show()
    new_train0 = temp_array
    # 4- Extract RGB channels with cv2.split(img)
    ## To make a figure without the frame
    my_dpi = 96
    array_size = 128
    fig = plt.figure(figsize=(array_size/my_dpi, array_size/my_dpi), dpi=my_dpi, frameon=False)
    ## To make the content fill the whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ## Drawing image
    ax.imshow(new_train0)
    fig.savefig('combined.png', dpi = my_dpi)
    img = cv2.imread('combined.png')
    plt.close(fig)
    return  np.asarray(cv2.split(img))
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
def format_samples(train_samples, test_samples, shape):
    '''
    Format training and test sets (n_samples, image_size, channels)
    '''
    start = time.time()
    if shape == 1:
        resh_train = np.asarray([change_format(rp) for rp in train_samples])
    if shape == 2:
        resh_train = np.asarray([change_format_rect(rp) for rp in train_samples])
    end = time.time()
    print("Reshape train time: ")
    timer(start,end)
    start = time.time()
    if shape == 1:
        resh_test = np.asarray([change_format_rect(rp) for rp in test_samples])
    if shape == 2:
        resh_test = np.asarray([change_format_rect(rp) for rp in test_samples])
    end = time.time()
    print("Reshape test time: ")
    timer(start,end)

    return resh_train, resh_test

class input_gen(object):
    '''
    class for data preparation (rps generator)
    '''

    def __init__(self, data_path_list, sequence_length, sensor_drop, filter_number, piecewise_lin_ref=125, preproc=False, visualize=True, is_filter = False):
        '''
        :param data_path_list: python list of four sub-dataset
        :param sequence_length: legnth of sequence (sliced time series)
        :param sensor_drop: sensors not to be considered
        :param piecewise_lin_ref: max rul value (if real rul value is larger than piecewise_lin_ref,
        then the rul value is piecewise_lin_ref)
        :param preproc: preprocessing
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.data_path_list = data_path_list
        self.sequence_length = sequence_length
        self.sensor_drop = sensor_drop
        self.preproc = preproc
        self.piecewise_lin_ref = piecewise_lin_ref
        self.visualize = visualize
        self.filter_number = filter_number


        ## Assign columns name
        cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
        cols += ['sensor_{0:02d}'.format(s + 1) for s in range(21)]
        col_rul = ['RUL_truth']

        train_FD = pd.read_csv(self.data_path_list[0], sep=' ', header=None,
                               names=cols, index_col=False)
        test_FD = pd.read_csv(self.data_path_list[1], sep=' ', header=None,
                              names=cols, index_col=False)
        RUL_FD = pd.read_csv(self.data_path_list[2], sep=' ', header=None,
                             names=col_rul, index_col=False)

        print(f"Train shape: {train_FD.shape}")
        if (is_filter):
          random_units = random.sample(range(1, 100), filter_number)
          train_FD = train_FD[train_FD['unit_nr'].isin(random_units)]
          # test_FD = test_FD[test_FD['unit_nr'].isin(random_units)]
          RUL_FD[RUL_FD.index.isin(set(test_FD['unit_nr']))]
          train_FD = train_FD[train_FD['unit_nr']<=filter_number]
          # test_FD = test_FD[test_FD['unit_nr']<=filter_number]
          # RUL_FD = RUL_FD.head(filter_number)


        ## Calculate RUL and append to train data
        # get the time of the last available measurement for each unit
        mapper = {}
        for unit_nr in train_FD['unit_nr'].unique():
            mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()

        # calculate RUL = time.max() - time_now for each unit
        train_FD['RUL'] = train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles']
        # piecewise linear for RUL labels
        train_FD['RUL'].loc[(train_FD['RUL'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref

        # Cut max RUL ground truth
        RUL_FD['RUL_truth'].loc[(RUL_FD['RUL_truth'] > self.piecewise_lin_ref)] = self.piecewise_lin_ref

        ## Excluse columns which only have NaN as value
        cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
        cols_const = [col for col in train_FD.columns if len(train_FD[col].unique()) <= 2]

        ## Drop exclusive columns
        # train_FD = train_FD.drop(columns=cols_const + cols_nan)
        # test_FD = test_FD.drop(columns=cols_const + cols_nan)

        train_FD = train_FD.drop(columns=cols_const + cols_nan + sensor_drop)

        test_FD = test_FD.drop(columns=cols_const + cols_nan + sensor_drop)


        if self.preproc == True:
            ## preprocessing(normailization for the neural networks)
            min_max_scaler = preprocessing.MinMaxScaler()
            # for the training set
            # train_FD['cycles_norm'] = train_FD['cycles']
            cols_normalize = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])

            norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_FD[cols_normalize]),
                                         columns=cols_normalize,
                                         index=train_FD.index)
            join_df = train_FD[train_FD.columns.difference(cols_normalize)].join(norm_train_df)
            train_FD = join_df.reindex(columns=train_FD.columns)

            # for the test set
            # test_FD['cycles_norm'] = test_FD['cycles']
            cols_normalize_test = test_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2'])
            # print ("cols_normalize_test", cols_normalize_test)
            norm_test_df = pd.DataFrame(min_max_scaler.transform(test_FD[cols_normalize_test]), columns=cols_normalize_test,
                                        index=test_FD.index)
            test_join_df = test_FD[test_FD.columns.difference(cols_normalize_test)].join(norm_test_df)
            test_FD = test_join_df.reindex(columns=test_FD.columns)
            test_FD = test_FD.reset_index(drop=True)
        else:
            # print ("No preprocessing")
            pass

        # Specify the columns to be used
        sequence_cols_train = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])
        sequence_cols_test = test_FD.columns.difference(['unit_nr', 'os_1', 'os_2', 'cycles'])



        ## generator for the sequences
        # transform each id of the train dataset in a sequence
        seq_gen = (list(gen_sequence(train_FD[train_FD['unit_nr'] == id], self.sequence_length, sequence_cols_train))
                   for id in train_FD['unit_nr'].unique())

        # generate sequences and convert to numpy array in training set
        seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
        self.seq_array_train = seq_array_train.transpose(0, 2, 1) # shape = (samples, sensors, sequences)
        print("seq_array_train.shape", self.seq_array_train.shape)

        # generate label of training samples
        label_gen = [gen_labels(train_FD[train_FD['unit_nr'] == id], self.sequence_length, ['RUL'])
                     for id in train_FD['unit_nr'].unique()]
        self.label_array_train = np.concatenate(label_gen).astype(np.float32)

        # generate sequences and convert to numpy array in test set (only the last sequence for each engine in test set)
        seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-self.sequence_length:]
                               for id in test_FD['unit_nr'].unique() if
                               len(test_FD[test_FD['unit_nr'] == id]) >= self.sequence_length]

        seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
        self.seq_array_test_last = seq_array_test_last.transpose(0, 2, 1) # shape = (samples, sensors, sequences)
        print("seq_array_test_last.shape", self.seq_array_test_last.shape)

        # generate label of test samples
        y_mask = [len(test_FD[test_FD['unit_nr'] == id]) >= self.sequence_length for id in test_FD['unit_nr'].unique()]
        label_array_test_last = RUL_FD['RUL_truth'][y_mask].values
        self.label_array_test = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)


        ## Visualize Run-2-failure TS of the first engine in the training set.(Please deactivate after understanding)
        if self.visualize == True:
            # R2F TS of the first engine
            pd.DataFrame(train_FD[train_FD['unit_nr'] == 1][sequence_cols_train].values,
                             columns=sequence_cols_train).plot(subplots=True, figsize=(15, 15))

            # The last sequences sliced from each TS (of the first engine)
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            colors = colors + colors + colors

            seq_gen = (
            list(gen_sequence(train_FD[train_FD['unit_nr'] == id], self.sequence_length, sequence_cols_train))
            for id in train_FD['unit_nr'].unique())

            seq_list_engine = list(seq_gen)
            seq_engine_1_array = np.asarray(seq_list_engine[0])

            last_seq_engine_1_array = seq_engine_1_array[-1, :, :]
            fig_ts = plt.figure(figsize=(15, 15))
            for s in range(last_seq_engine_1_array.shape[1]):
                seq_s = last_seq_engine_1_array[:, s]
                # plt.subplot(last_seq_engine_1_array.shape[1],(s//4) + 1, (s%4)+1)
                plt.subplot(4, 4, s + 1)
                plt.plot(seq_s, "y", label=sequence_cols_train[s], color=colors[s])
                plt.legend()

            plt.xlabel("time(cycles)")
            plt.show()

    # Filter the number of data
    def filter(df, num):
      return df[df['unit_nr']<=num]

    def rps(self, thres_type=None, thres_percentage=50, flatten=False, visualize=True):
        '''
        generate RPs from sequences
        :param thres_type:  ‘point’, ‘distance’ or None (default = None)
        :param thres_percentage:
        :param flatten:
        :param visualize: visualize generated RPs (first training sample)
        :return: PRs (samples for NNs and their label)
        '''

        # Recurrence plot transformation for training samples
        rp_train = RecurrencePlot(threshold=thres_type, percentage=thres_percentage,flatten=flatten)

        rp_list = []
        for idx in range(self.seq_array_train.shape[0]):
            temp_mts = self.seq_array_train[idx]
            # print (temp_mts.shape)
            X_rp_temp = rp_train.fit_transform(temp_mts)
            # print (X_rp_temp.shape)
            rp_list.append(X_rp_temp)

        rp_train_samples = np.stack(rp_list, axis=0)

        # Recurrence plot transformation for test samples
        rp_test = RecurrencePlot(threshold=thres_type, percentage=thres_percentage, flatten=flatten)
        rp_list = []
        for idx in range(self.seq_array_test_last.shape[0]):
            temp_mts = self.seq_array_test_last[idx]
            # print (temp_mts.shape)
            X_rp_temp = rp_test.fit_transform(temp_mts)
            # print (X_rp_temp.shape)
            rp_list.append(X_rp_temp)
        rp_test_samples = np.stack(rp_list, axis=0)

        label_array_train = self.label_array_train
        label_array_test = self.label_array_test

        # Visualize RPs of the last sequences sliced from each TS (of the first engine)
        if visualize == True:
            X_rp = rp_train_samples[-1]
            plt.figure(figsize=(15, 15))
            for s in range(len(X_rp)):
                # plt.subplot(last_seq_engine_1_array.shape[1],(s//4) + 1, (s%4)+1)
                plt.subplot(4, 4, s + 1)
                if flatten == True:
                    img = np.atleast_2d(X_rp[s])
                    plt.imshow(img, extent=(0, img.shape[1], 0, round(img.shape[1]/9)))
                else:
                    plt.imshow(X_rp[s], origin='lower')
                # plt.legend()
            plt.show()

        return  rp_train_samples, label_array_train, rp_test_samples, label_array_test


def main():

    parser = argparse.ArgumentParser(description='RPs creator')
    parser.add_argument('-i', type=int, help='Input sources', required=True)
    parser.add_argument('-l', type=int, default=32, help='sequence length')
    parser.add_argument('--method', type=str, default='rps', help='data representation: rps', required=False)
    parser.add_argument('--preproc', type=str, default='yes', help='MinMax Scaler activation', required=False)
    parser.add_argument('--thres_type', type=str, default=None, required=False,
                        help='threshold type for RPs: distance or point ')
    parser.add_argument('--thres_value', type=int, default=50, required=False,
                        help='percentage of maximum distance or black points for threshold')
    parser.add_argument('--flatten', type=str, default='no', help='flatten rps array.')
    parser.add_argument('--visualize', type=str, default='yes', help='visualize rps.')
    parser.add_argument('--filter_num', type=int, default=100, required=False, help='Is the percentage of training set returned by the algorithm')
    parser.add_argument('--verbose', type=int, default=2, required=False, help='Verbose TF training')
    parser.add_argument('--shape', type=int, default=1, required=False, help='The shape of the final Image. 1 is a squared shape, 2 for rectangular shape.')

    args = parser.parse_args()


    # Architecture preferences
    dp = FD_path[args.i]
    subdataset = dp_str[args.i]
    sequence_length = args.l
    thres_type = args.thres_type
    thres_value = args.thres_value
    method = args.method
    filter_number = args.filter_num
    flatten = args.flatten
    shape = args.shape
    
    if flatten == 'yes':
        flatten = True
    elif flatten == 'no':
        flatten = False

    visualize = args.visualize
    if visualize == 'yes':
        visualize = True
    elif visualize == 'no':
        visualize = False
    
    preproc = args.preproc
    if preproc == 'yes':
        preproc = True
    elif preproc == 'no':
        preproc = False

    # Start pre-processing
    start = time.time()
    print("Dataset: ", subdataset)
    print("Seq_len: ", sequence_length)

    data_class = input_gen(data_path_list=dp, 
                        sequence_length=sequence_length,
                        preproc = preproc, 
                        sensor_drop= sensor_drop, 
                        filter_number = filter_number,
                        is_filter = (filter_number<100),
                        visualize=visualize)
    if method == 'rps':
        train_samples, label_array_train, test_samples, label_array_test = data_class.rps(
            thres_type=thres_type,
            thres_percentage=thres_value,
            flatten=flatten,
            visualize=visualize)

    elif method == 'jrp': # TODO
        pass

    print (f"Change format of {filter_number}% of the original dataset")
    train_samples, test_samples = format_samples(train_samples, test_samples, shape)

    print ("train_samples.shape: ", train_samples.shape) # shape = (samples, channels, height, width)
    print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
    print ("test_samples.shape: ", test_samples.shape) # shape = (samples, channels, height, width)
    print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

    train_samples = np.transpose(train_samples, (0,2,3,1))
    test_samples = np.transpose(test_samples, (0,2,3,1))

    print ("train_samples.shape: ", train_samples.shape) # shape = (samples, height, width, channels)
    print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
    print ("test_samples.shape: ", test_samples.shape) # shape = (samples, height, width, channels)
    print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)
    
    np.save(current_dir + f'/preprocess/preprocess_{str(filter_number)}/train_samples.npy', train_samples)
    np.save(current_dir + f'/preprocess/preprocess_{str(filter_number)}/test_samples.npy', test_samples)
    np.save(current_dir + f'/preprocess/preprocess_{str(filter_number)}/label_array_train.npy', label_array_train)
    np.save(current_dir + f'/preprocess/preprocess_{str(filter_number)}/label_array_test.npy', label_array_test)

    end = time.time()
    print("Preprocessing time: ", timer(start, end))

if __name__ == '__main__':
    main()