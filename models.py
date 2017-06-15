import os,sys
import glob
import subprocess
import numpy as np
import pandas as pd
import natsort
import h5py
import cPickle as pickle
import tempfile
import re
import random
import pyDNase
from pyfasta import Fasta

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import minmax_scale
from keras.preprocessing import sequence
from keras.optimizers import RMSprop,Adadelta
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D,Conv2D,Convolution1D,Convolution2D, MaxPooling1D, MaxPooling2D
from keras.regularizers import l2, l1, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,ELU
from keras.constraints import maxnorm, nonneg,unitnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.utils.io_utils import HDF5Matrix
from keras.layers.noise import GaussianDropout
from keras.layers import Merge
from keras.layers.local import LocallyConnected1D
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv1D,Conv2D,Convolution1D,Convolution2D, MaxPooling1D, MaxPooling2D
from keras.layers.pooling import GlobalMaxPooling2D,GlobalMaxPooling1D
from keras.optimizers import Nadam
from keras.initializers import Constant
from keras.optimizers import Adadelta
import keras.layers
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.local import LocallyConnected1D
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from keras.layers.embeddings import Embedding

from . import data_prep as DP

from CNN_Models import cnn_helpers2 as CH

def make_CENNTIPEDE_model(data):
    ada = Adadelta()
    input0 = Input(shape=(data['train_data_X'].shape[1],))
    lay1 = Dense(500,activation='relu',name='HL1',use_bias=False,activity_regularizer=regularizers.l1(10e-5))(input0)
    lay2 = Dense(100,activation='relu',name='HL2',use_bias=False,activity_regularizer=regularizers.l1(10e-5))(lay1)
    # do1 = Dropout(0.25)(lay2)
    # lay3 = Dense(500,activation='relu',name='HL3',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(lay2)
    # do2 = Dropout(0.25)(lay2)
    predictions = Dense(1, activation='sigmoid')(lay2)

    model = Model(inputs=input0, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                    optimizer=ada,
                    metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def make_CENNTIPEDE_Embedding_model(data):
    ada = Adadelta()
    input0 = Input(shape=(data['train_data_X'].shape[1],))
    emb = Embedding(2,2,input_length=data['train_data_X'].shape[1])(input0)
    # fl = Flatten()(emb)
    conv1 = Conv1D(5, data['train_data_X'].shape[1],activation='relu')(emb)
    # lay1 = Dense(100,activation='relu',name='HL1',use_bias=False,activity_regularizer=regularizers.l1(10e-5))(fl)
    fl = Flatten()(conv1)
    lay2 = Dense(100,activation='relu',name='HL2',use_bias=False,activity_regularizer=regularizers.l1(10e-5))(fl)
    # do1 = Dropout(0.25)(lay2)
    # lay3 = Dense(500,activation='relu',name='HL3',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(lay2)
    # do2 = Dropout(0.25)(lay2)
    # predictions = Dense(1, activation='sigmoid')(lay2)
    # fl = Flatten()(lay2)
    predictions = Dense(1, activation='sigmoid')(lay2)

    # model = Model(inputs=input0, outputs=predictions)
    model = Model(inputs=input0, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                    optimizer=ada,
                    metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def make_CENNTIPEDE_AE_model(data):
    ada = Adadelta()
    input0 = Input(shape=(data['train_data_X'].shape[1],))
    encoded = Dense(500,activation='relu',name='e1',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(input0)
    encoded = Dense(250,activation='relu',name='e2',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(encoded)
    # encoded = Dense(250,activation='relu',name='e2',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(encoded)
    # encoded = Dense(100,activation='relu',name='e3',use_bias=True)(encoded)
    # encoded = Dense(50,activation='relu',name='e4',use_bias=True)(encoded)
    # encoded = Dense(5,activation='relu',name='e5',use_bias=True)(encoded)

    # decoded = Dense(100,activation='relu',name='d1',use_bias=True)(encoded)
    # decoded = Dense(250,activation='relu',name='d2',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(decoded)
    decoded = Dense(500,activation='relu',name='d1',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dense(data['train_data_X'].shape[1],activation='relu',name='d2',use_bias=True,activity_regularizer=regularizers.l1(10e-5))(decoded)

    # predictions = Dense(1, activation='sigmoid')(do2)

    model = Model(inputs=input0, outputs=decoded)
    model.compile(loss='binary_crossentropy',
                    optimizer=ada,
                    metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def make_CENNTIPEDE_RNN_model(data):
    ada = Adadelta()
    input0 = Input(shape=(data['train_data_X'].shape[1],))
    gru = GRU(2)(input0)
    hl = Dense(50)(gru)
    pred = Dense(1, activation='sigmoid')(hl)
    model = Model(inputs=input0, outputs=pred)
    model.compile(loss='binary_crossentropy',
                    optimizer=ada,
                    metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def fit_CENNTIPEDE_model(model, data):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=5,verbose=0)
    model.fit(data['train_data_X'], data['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = (data['val_data_X'],data['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def fit_CENNTIPEDE_AE_model(model, data):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=5,verbose=0)
    model.fit(data['train_data_X'], data['train_data_X'],
        epochs=30,
        # batch_size=256,
        shuffle=True,
        validation_data = (data['val_data_X'],data['val_data_X']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def CENNTIPEDE_Effect_SNP_model(data1,data2):
    ada = Adadelta()

    #CENN side Footprints
    input0 = Input(shape=(data1['train_data_X'].shape[1],))
    lay1a = Dense(1000,activation='relu',name='HL1a',use_bias=True)(input0)

    #CENN side Effect SNPS
    input1 = Input(shape=(data2['train_data_X'].shape[1],))
    lay1b = Dense(1000,activation='relu',name='HL1b',use_bias=True)(input1)

    m = keras.layers.add([lay1a,lay1b])

    mHL = Dense(100,activation='relu',name='mHL')(m)
    predictions = Dense(1, activation='sigmoid')(mHL)
    # predictions = Dense(1, activation='sigmoid')(m)
    model = Model(inputs=[input0,input1], outputs=predictions)
    model.compile(loss='binary_crossentropy',optimizer=ada,metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def CENNTIPEDE_CNNtipede_model(data1,data2,data3):
    ada = Adadelta()

    #CENN side Footprints
    input0 = Input(shape=(data1['train_data_X'].shape[1],))
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    lay1a = Dense(1000,activation='relu',name='HL1a',use_bias=True)(input0)

    #CENN side Effect SNPS
    input1 = Input(shape=(data2['train_data_X'].shape[1],))
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    lay1b = Dense(1000,activation='relu',name='HL1b',use_bias=True)(input1)

    #CNN side
    input2 = Input(shape=(300,4), name='dna_seqs0')
    # num_filt_0 = data1['train_data_X'].shape[1]
    num_filt_0 = 1000
    conv = Conv1D(filters=num_filt_0,kernel_size=20,padding='same',activation='relu',name="CNN_conv",trainable=True,use_bias=True)(input2)
    pool = MaxPooling1D(pool_size=300,name='WX_max')(conv)
    f0 = Flatten(name='flatten')(pool)

    # m = keras.layers.concatenate([lay1a,lay1b,f0])
    m = keras.layers.add([lay1a,lay1b,f0])

    # mHL = Dense(100,activation='relu',name='mHL')(m)
    # predictions = Dense(1, activation='sigmoid')(mHL)
    predictions = Dense(1, activation='sigmoid')(m)
    model = Model(inputs=[input0,input1,input2], outputs=predictions)
    model.compile(loss='binary_crossentropy',optimizer=ada,metrics=["binary_accuracy","mean_absolute_error"])
    return(model)

def CENNTIPEDE_CNNtipede_pwm_model(data1,data2,data3,kernel_size_seq=22):
    conv_weights,num_filts_seq,good_pwms_idx = make_pwm_conv_filters(kernel_size_seq,rev_comp=False)

    ada = Adadelta()

    #CENN side Footprints
    # input0 = Input(shape=(data1['train_data_X'].shape[1],),name='Footprints')
    input0 = Input(shape=(num_filts_seq,),name='Footprints')
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    # lay1a = Dense(num_filts_seq,activation='relu',name='HL1a',use_bias=True)(input0)

    #CENN side Effect SNPS
    # input1 = Input(shape=(data2['train_data_X'].shape[1],),name='Effect SNPs')
    input1 = Input(shape=(num_filts_seq,),name='Effect SNPs')
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    # lay1b = Dense(num_filts_seq,activation='relu',name='HL1b',use_bias=True)(input1)

    #CNN side
    input2 = Input(shape=(300,4), name='Sequence')
    conv = Conv1D(filters=num_filts_seq,kernel_size=kernel_size_seq,padding='same',activation=None,name="CNN_conv",trainable=False,use_bias=False)(input2)
    pool = MaxPooling1D(pool_size=300,name='WX_max')(conv)
    f0 = Flatten(name='flatten_CNN')(pool)

    # m = keras.layers.concatenate([lay1a,lay1b,f0])
    # m = keras.layers.concatenate([input0,input1,f0])
    m = keras.layers.multiply([input0,input1,f0])

    mHL = Dense(100,activation='relu',name='mHL')(m)
    # mr = Reshape((num_filts_seq,3))(m)
    # mrf = Flatten(name='flatten_merge')(mr)
    predictions = Dense(1, activation='sigmoid')(mHL)
    model = Model(inputs=[input0,input1,input2], outputs=predictions)
    model.compile(loss='binary_crossentropy',optimizer=ada,metrics=["binary_accuracy","mean_absolute_error"])

    ## Seed the filters with PWM
    W0 = model.get_layer('CNN_conv').get_weights()
    W0[0][:,:,:num_filts_seq] = conv_weights
    W0[0] = W0[0][:,:,:num_filts_seq]
    model.get_layer('CNN_conv').set_weights([W0[0]])

    return(model,good_pwms_idx)

def CENNTIPEDE_CNNtipede_pwm_model_proto(data1,data2,data3,kernel_size_seq=22):
    conv_weights,num_filts_seq,good_pwms_idx = make_pwm_conv_filters(kernel_size_seq,width=8,rev_comp=False)

    ada = Adadelta()

    #CENN side Footprints
    # input0 = Input(shape=(data1['train_data_X'].shape[1],),name='Footprints')
    input0 = Input(shape=(num_filts_seq,),name='Footprints')
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    # lay1a = Dense(num_filts_seq,activation='relu',name='HL1a',use_bias=True)(input0)

    #CENN side Effect SNPS
    # input1 = Input(shape=(data2['train_data_X'].shape[1],),name='Effect SNPs')
    input1 = Input(shape=(num_filts_seq,),name='Effect SNPs')
    # lay1 = Dense(data1['train_data_X'].shape[1],activation='relu',name='HL1',use_bias=True)(input0)
    # lay1b = Dense(num_filts_seq,activation='relu',name='HL1b',use_bias=True)(input1)

    #CNN side
    input2 = Input(shape=(300,8), name='Sequence')
    conv = Conv1D(filters=num_filts_seq,kernel_size=kernel_size_seq,padding='same',activation=None,name="CNN_conv",trainable=True,use_bias=True)(input2)
    pool = MaxPooling1D(pool_size=300,name='WX_max')(conv)
    f0 = Flatten(name='flatten_CNN')(pool)

    # m = keras.layers.concatenate([lay1a,lay1b,f0])
    # m = keras.layers.concatenate([input0,input1,f0])
    m = keras.layers.multiply([input0,input1,f0])

    mHL = Dense(500,activation='relu',name='mHL')(m)
    mHLD = Dropout(0.25)(mHL)
    # mr = Reshape((num_filts_seq,3))(m)
    # mrf = Flatten(name='flatten_merge')(mr)
    predictions = Dense(1, activation='sigmoid')(mHLD)
    model = Model(inputs=[input0,input1,input2], outputs=predictions)
    model.compile(loss='binary_crossentropy',optimizer=ada,metrics=["binary_accuracy","mean_absolute_error"])

    ## Seed the filters with PWM
    W0 = model.get_layer('CNN_conv').get_weights()
    W0[0][:,:,:num_filts_seq] = conv_weights
    W0[0] = W0[0][:,:,:num_filts_seq]
    # model.get_layer('CNN_conv').set_weights([W0[0]])
    model.get_layer('CNN_conv').set_weights(W0)

    return(model,good_pwms_idx)

def fit_CENNTIPEDE_CNNtipede_pwm_model_proto(model, data1, data2, data3,good_pwms_idx):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=5,verbose=0)
    model.fit([data1['train_data_X'][:,good_pwms_idx],data2['train_data_X'][:,good_pwms_idx],data3['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'][:,good_pwms_idx],data2['val_data_X'][:,good_pwms_idx],data3['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)
