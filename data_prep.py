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

# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import maxabs_scale
# from sklearn.preprocessing import minmax_scale
# from keras.preprocessing import sequence
# from keras.optimizers import RMSprop,Adadelta
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.models import load_model
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv1D,Conv2D,Convolution1D,Convolution2D, MaxPooling1D, MaxPooling2D
# from keras.regularizers import l2, l1, l1_l2
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU,ELU
# from keras.constraints import maxnorm, nonneg,unitnorm
# from keras.layers.recurrent import LSTM, GRU
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras import backend as K
# from keras.utils.io_utils import HDF5Matrix
# from keras.layers.noise import GaussianDropout
# from keras.layers import Merge
# from keras.layers.local import LocallyConnected1D
# from keras.layers.convolutional import ZeroPadding1D
# from keras.layers.core import Reshape
# from keras.layers.convolutional import Conv1D,Conv2D,Convolution1D,Convolution2D, MaxPooling1D, MaxPooling2D
# from keras.layers.pooling import GlobalMaxPooling2D,GlobalMaxPooling1D
# from keras.optimizers import Nadam
# from keras.initializers import Constant
# from keras.optimizers import Adadelta
# import keras.layers
# from keras.layers.convolutional import Conv1D# from keras.layers.merge import Co# from keras.layers.local import LocallyConnected1D
# from keras.models import Model
# from keras.layers import Dense, Input
# from keras import regularizers

from CNN_Models import cnn_helpers2 as CH
from collections import OrderedDict
from scipy import sparse

def load_compendium_footprints():
    foot = sparse.load_npz('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_footprint_sparse.npz')
    foot = foot.toarray()
    return(foot)

def load_compendium_ES():
    es = sparse.load_npz('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_es_sparse.npz')
    es = es.toarray()
    return(es)

def make_all_compendium_ES():
    compendium_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed'
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    compendium_df = pd.read_csv(compendium_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,effect_snp in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp).split(".")[0]
        cmd_str = "less {1} | sed -E '/#1/d' | bedtools intersect -a {0} -b stdin -c | cut -f4".format(compendium_bed_file,effect_snp)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        try:
            compendium_df[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            compendium_df[pwm_name] = 0

        if idx % 100 == 0:
            compendium_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.h5','df',mode='w')

    compendium_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.h5','df',mode='w')

def make_all_compendium_footprints():
    compendium_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed'
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    compendium_df = pd.read_csv(compendium_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]

        cmd_str = "bedtools slop -g /wsu/home/groups/piquelab/footprint.fungei/byTreatAllRepsMerged/centipede/x8bFiles/new.chromSizes.txt -l 0 -r 1 -i {1} | bedtools intersect -a {0} -b stdin -c | cut -f4".format(compendium_bed_file,pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        compendium_df[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            compendium_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_footprints.h5','df',mode='w')

    compendium_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_footprints.h5','df',mode='w')

def load_cindy_mpra_footprints():
    df = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_footprint_single_snp.h5','df')
    X=np.array(df.iloc[:,3:])
    X[X>1] = 1
    return(X)

def load_cindy_mpra_ES():
    df = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_ES_single_snp.h5','df')
    X=np.array(df.iloc[:,3:])
    X[X>1] = 1
    return(X)


def load_tewhey_circuit_footprints():
    df = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_footprint_single_snp.h5','df')
    X=np.array(df.iloc[:,3:])
    X[X>1] = 1
    return(X)

def load_tewhey_circuit_ES():
    df = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_effect_single_snp.h5','df')
    X=np.array(df.iloc[:,3:])
    X[X>1] = 1
    return(X)

def make_cindy_mpra_test_data_ES():
    # mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_mpra_pos.bed'
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_mpra_300_pos.bed'
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,effect_snp in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp).split(".")[0]
        cmd_str = "less {1} | sed -E '/#1/d' | bedtools intersect -a {0} -b stdin -c | cut -f4".format(mpra_bed_file,effect_snp)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        try:
            mpra_df[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            mpra_df[pwm_name] = 0

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_ES_single_snp.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_ES_single_snp.h5','df',mode='w')

def make_cindy_mpra_test_data_single_snp():
    # mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_mpra_pos.bed'
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_mpra_300_pos.bed'
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]

        cmd_str = "bedtools slop -g /wsu/home/groups/piquelab/footprint.fungei/byTreatAllRepsMerged/centipede/x8bFiles/new.chromSizes.txt -l 0 -r 1 -i {1} | bedtools intersect -a {0} -b stdin -c | cut -f4".format(mpra_bed_file,pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        mpra_df[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_footprint_single_snp.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/cindy_dsqtl_footprint_single_snp.h5','df',mode='w')

def make_tewhey_test_data_ES():
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_300_windows.tab'
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,effect_snp in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp).split(".")[0]
        cmd_str = "less {1} | sed -E '/#1/d' | bedtools intersect -a {0} -b stdin -c | cut -f4".format(mpra_bed_file,effect_snp)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        try:
            mpra_df[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            mpra_df[pwm_name] = 0

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_effect_snp_window.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_effect_snp_window.h5','df',mode='w')

def make_tewhey_test_data_ES_single_snp():
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_300_pos.bed'
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,effect_snp in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp).split(".")[0]
        cmd_str = "less {1} | sed -E '/#1/d' | bedtools intersect -a {0} -b stdin -c | cut -f4".format(mpra_bed_file,effect_snp)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        try:
            mpra_df[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            mpra_df[pwm_name] = 0

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_effect_single_snp.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_effect_single_snp.h5','df',mode='w')

def make_tewhey_test_data_window():
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_300_windows.tab'
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools intersect -a {0} -b {1} -c | cut -f4".format(mpra_bed_file,pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        mpra_df[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_footprint_window.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_footprint_window.h5','df',mode='w')

def make_tewhey_test_data_single_snp():
    mpra_bed_file = '/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_300_pos.bed'
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    mpra_df = pd.read_csv(mpra_bed_file,delim_whitespace=True,header=None,names=["chr","start","stop"])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools slop -g /wsu/home/groups/piquelab/footprint.fungei/byTreatAllRepsMerged/centipede/x8bFiles/new.chromSizes.txt -l 0 -r 1 -i {1} | bedtools intersect -a {0} -b stdin -c | cut -f4".format(mpra_bed_file,pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        mpra_df[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_footprint_single_snp.h5','df',mode='w')

    mpra_df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/mpra/mpra_footprint_single_snp.h5','df',mode='w')

def make_matrix():

    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    df = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed -b {0} -c | cut -f4".format(pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        df[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/data_mat.h5','df',mode='w')

    df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/data_mat.h5','df',mode='w')

def make_effect_snp_vector():

    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    df = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])
    # df['effect_snp'] = 0
    effect_snp_arr = np.zeros((df.shape[0]))
    for idx,effect_snp_file in enumerate(effect_snps):
        # pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        cmd_str = "less {0} | sed -E '/#1/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed -b stdin -c | cut -f4".format(effect_snp_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = np.array(raw_out.split('\n')[:-1],dtype=np.int)
        effect_snp_arr[np.array(out_list) > 0] = 1

        # if idx % 100 == 0:
            # df.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mat.h5','df',mode='w')
    df_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/data_mat.h5','df')
    df_mat['effect_snp'] = effect_snp_arr
    df_mat.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mat.h5','df',mode='w')

def make_effect_snp_mask_matrix():

    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    df_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])
    df_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])

    for idx,effect_snp_file in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        try:
            cmd_str_pos = "less {0} | sed -E '/#1/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed -b stdin -c | cut -f4".format(effect_snp_file)
            raw_out_pos = subprocess.check_output(cmd_str_pos,shell=True)
            out_list_pos = np.array(raw_out_pos.split('\n')[:-1],dtype=np.int)
            out_list_pos[out_list_pos > 0] = 1
            df_pos[pwm_name] = out_list_pos
        except:
            df_pos[pwm_name] = 0

        try:
            cmd_str_neg = "less {0} | sed -E '/#1/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed -b stdin -c | cut -f4".format(effect_snp_file)
            raw_out_neg = subprocess.check_output(cmd_str_neg,shell=True)
            out_list_neg = np.array(raw_out_neg.split('\n')[:-1],dtype=np.int)
            out_list_neg[out_list_neg > 0] = 1
            df_neg[pwm_name] = out_list_neg
        except:
            df_neg[pwm_name] = 0

        if idx % 100 == 0:
            df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mask_mat_pos.h5','df',mode='w')
            df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mask_mat_neg.h5','df',mode='w')
    df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mask_mat_pos.h5','df',mode='w')
    df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mask_mat_neg.h5','df',mode='w')

def make_effect_snp_mask_matrix_2():
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    pos = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_pos_peak.bed'
    neg = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_neg_peak.bed'

    df_pos = pd.read_csv(pos,delim_whitespace=True,header=None,names=['chr','start','stop'])
    df_neg = pd.read_csv(neg,delim_whitespace=True,header=None,names=['chr','start','stop'])

    for idx,effect_snp_file in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        try:
            cmd_str_pos = "bedtools intersect -a {1} -b {0} -c | cut -f4".format(effect_snp_file,pos)
            raw_out_pos = subprocess.check_output(cmd_str_pos,shell=True)
            out_list_pos = np.array(raw_out_pos.split('\n')[:-1],dtype=np.int)
            df_pos[pwm_name] = out_list_pos
        except:
            df_pos[pwm_name] = 0
        try:
            cmd_str_neg = "bedtools intersect -a {1} -b {0} -c | cut -f4".format(effect_snp_file,neg)
            raw_out_neg = subprocess.check_output(cmd_str_neg,shell=True)
            out_list_neg = np.array(raw_out_neg.split('\n')[:-1],dtype=np.int)
            df_neg[pwm_name] = out_list_neg
        except:
            df_neg[pwm_name] = 0

        if idx % 100 == 0:
            df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos.h5','df',mode='w')
            df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg.h5','df',mode='w')

    df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos.h5','df',mode='w')
    df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg.h5','df',mode='w')

def make_effect_snp_mask_matrix_3():
    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)
    pos = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_pos_peak.bed'
    neg = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_neg_peak.bed'

    df_pos = pd.read_csv(pos,delim_whitespace=True,header=None,names=['chr','start','stop'])
    df_neg = pd.read_csv(neg,delim_whitespace=True,header=None,names=['chr','start','stop'])

    for idx,effect_snp_file in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        try:
            cmd_str_pos = "less {0} | sed -E '/1$/d' | bedtools intersect -a {1} -b stdin -c | cut -f4".format(effect_snp_file,pos)
            raw_out_pos = subprocess.check_output(cmd_str_pos,shell=True)
            out_list_pos = np.array(raw_out_pos.split('\n')[:-1],dtype=np.int)
            df_pos[pwm_name] = out_list_pos
        except:
            df_pos[pwm_name] = 0
        try:
            cmd_str_neg = "less {0} | sed -E '/1$/d' | bedtools intersect -a {1} -b stdin -c | cut -f4".format(effect_snp_file,neg)
            raw_out_neg = subprocess.check_output(cmd_str_neg,shell=True)
            out_list_neg = np.array(raw_out_neg.split('\n')[:-1],dtype=np.int)
            df_neg[pwm_name] = out_list_neg
        except:
            df_neg[pwm_name] = 0

        if idx % 100 == 0:
            df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos_1.h5','df',mode='w')
            df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg_1.h5','df',mode='w')

    df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos_1.h5','df',mode='w')
    df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg_1.h5','df',mode='w')

def get_pos_neg_dnase_locations(fasta_files_list,pos_neg='pos'):
    """
    Use the python module 'pydnase' to get DNase counts
    Slower than X8b method
    """
    # neg_list = ['/wsu/home/al/al37/al3786/gm12878_sequence_sets/nullseqs_gm12878_shared.1.1.fa',
    # '/wsu/home/al/al37/al3786/gm12878_sequence_sets/nullseqs_gm12878_shared.2.1.fa',
    # '/wsu/home/al/al37/al3786/gm12878_sequence_sets/nullseqs_gm12878_shared.3.1.fa',
    # '/wsu/home/al/al37/al3786/gm12878_sequence_sets/nullseqs_gm12878_shared.4.1.fa',
    # '/wsu/home/al/al37/al3786/gm12878_sequence_sets/nullseqs_gm12878_shared.5.1.fa']


    dnase_bed_arr_all = []
    for fasta_file in fasta_files_list:
        f = Fasta(fasta_file)
        dna_window_list = f.keys()
        dnase_bed_arr = []

        for idx, pos in enumerate(dna_window_list):
            tmp1 = pos.replace(":","\t").replace("-","\t")
            tmp2 = tmp1.split("\t")
            tmp2[1]=str(int(tmp2[1]) -1)

            dnase_bed_arr.append("\t".join(tmp2))
        dnase_bed_arr_all = dnase_bed_arr_all + dnase_bed_arr

    dnase_bed_arr_all = natsort.natsorted(dnase_bed_arr_all)
    dnase_bed_arr_all = [x.split("\t") for x in dnase_bed_arr_all]
    dnase_bed_arr_all = np.array(dnase_bed_arr_all)
    df = pd.DataFrame(dnase_bed_arr_all[:,0])
    df[1] = dnase_bed_arr_all[:,1]
    df[2] = dnase_bed_arr_all[:,2]
    df.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_{0}_peak.bed'.format(pos_neg), index=False, header=None, sep="\t")

def make_training_labels():
    pos = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_pos_peak.bed'
    neg = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_neg_peak.bed'
    clss = ['pos','neg']
    df = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])
    # df['dnase_peak'] = 0
    dnase_peak = np.zeros((df.shape[0]))
    dnase_peak = np.zeros((df.shape[0]))

    for idx,peak in enumerate([pos,neg]):
        # pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed -b {0} -c | cut -f4".format(peak)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = np.array(raw_out.split('\n')[:-1],dtype=np.int)
        if idx == 0:
            dnase_peak[np.array(out_list) > 0] = 1
        else:
            dnase_peak[np.array(out_list) > 0] = -1

    df_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/effect_snp_mat.h5','df')
    df_mat['dnase_peak'] = dnase_peak
    df_mat.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/model_mat.h5','df',mode='w')

def make_training_labels_2():
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    pos = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_pos_peak.bed'
    neg = '/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_neg_peak.bed'

    df_pos = pd.read_csv(pos,delim_whitespace=True,header=None,names=['chr','start','stop'])
    df_neg = pd.read_csv(neg,delim_whitespace=True,header=None,names=['chr','start','stop'])

    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str_pos = "bedtools intersect -a {1} -b {0} -c | cut -f4".format(pwm_file,pos)
        cmd_str_neg = "bedtools intersect -a {1} -b {0} -c | cut -f4".format(pwm_file,neg)
        raw_out_pos = subprocess.check_output(cmd_str_pos,shell=True)
        raw_out_neg = subprocess.check_output(cmd_str_neg,shell=True)
        out_list_pos = np.array(raw_out_pos.split('\n')[:-1],dtype=np.int)
        out_list_neg = np.array(raw_out_neg.split('\n')[:-1],dtype=np.int)
        df_pos[pwm_name] = out_list_pos
        df_neg[pwm_name] = out_list_neg

        if idx % 100 == 0:
            df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_pos.h5','df',mode='w')
            df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_neg.h5','df',mode='w')

    df_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_pos.h5','df',mode='w')
    df_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_neg.h5','df',mode='w')

def intersect_beers_test_data():
    ## Load up Beers data
    df = pd.read_excel('/nfs/rprscratch/deep_learning/beer_data/ng.3331-S2.xlsx')
    df = df[['chrom_hg19','pos_hg19','allele1','allele2','label']]
    df.columns = ['chr','stop','alt','ref','label']
    df['start'] = df['stop']
    df['stop'] = df['start'] + 1

    ## Load big matrix
    df_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/model_mat.h5','df')
    beers_pos_vec = np.zeros((df_mat.shape[0]))
    beers_neg_vec = np.zeros((df_mat.shape[0]))

    ## Write bed file and add intersect to data matrix
    ##Positives dsQTLs
    pos_file_name = "/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed"
    df[df.label>0].to_csv(pos_file_name,index=False,header=False,sep='\t',columns=['chr','start','stop'])
    cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed -b {0} -c | cut -f4".format(pos_file_name)
    raw_out = subprocess.check_output(cmd_str,shell=True)
    out_list = np.array(raw_out.split('\n')[:-1],dtype=np.int)
    beers_pos_vec[np.array(out_list) > 0] = 1
    df_mat['beers_test_pos'] = beers_pos_vec

    ##Negative dsQTLs
    neg_file_name = "/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed"
    df[df.label<0].to_csv(neg_file_name,index=False,header=False,sep='\t',columns=['chr','start','stop'])
    cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed -b {0} -c | cut -f4".format(neg_file_name)
    raw_out = subprocess.check_output(cmd_str,shell=True)
    out_list = np.array(raw_out.split('\n')[:-1],dtype=np.int)
    beers_neg_vec[np.array(out_list) > 0] = 1
    df_mat['beers_test_neg'] = beers_neg_vec

    df_mat.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/model_mat2.h5','df',mode='w')

def make_beers_test_matrix():
    ##Make flanking windows around positive dsqtl SNPs
    df_beers_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)
    df_beers_pos = CH.add_flanks(df_beers_pos, 150)
    df_beers_pos.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed',index=False,header=False,sep='\t')

    ##Make flanking windows around negative dsqtl SNPs
    df_beers_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)
    df_beers_neg = CH.add_flanks(df_beers_neg, 150)
    df_beers_neg.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed',index=False,header=False,sep='\t')

    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    # pwm_files = sorted(pwm_files)
    pwm_files = natsort.natsorted(pwm_files)

    # df_ = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])

    ### Populate positive matrix using beers windows and pwm files
    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed -b {0} -c | cut -f4".format(pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        df_beers_pos[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.h5','df',mode='w')

    df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.h5py','df',mode='w')

    ### Populate neg matrix using beers windows and pwm files
    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed -b {0} -c | cut -f4".format(pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        df_beers_neg[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.h5','df',mode='w')

    df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.h5','df',mode='w')

def make_beers_test_matrix_single_snp():
    """
    Use only snps to create the dsQTL test data - instead of windowing around snps.
    """

    df_beers_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)
    df_beers_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)
    pwm_files = glob.glob('/nfs/rprdata/Anthony/data/combo/combo/*.gz')
    pwm_files = sorted(pwm_files)
    # pwm_files = natsort.natsorted(pwm_files)

    # df_ = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/input_windows.bed',delim_whitespace=True,header=None,names=['chr','start','stop'])

    ### Populate positive matrix using beers windows and pwm files
    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools slop -g /wsu/home/groups/piquelab/footprint.fungei/byTreatAllRepsMerged/centipede/x8bFiles/new.chromSizes.txt -l 0 -r 1 -i {0} | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed -b stdin -c | cut -f4".format(pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        df_beers_pos[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_snps_only.h5','df',mode='w')

    df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_snps_only.h5','df',mode='w')

    ### Populate neg matrix using beers windows and pwm files
    for idx,pwm_file in enumerate(pwm_files):
        pwm_name = os.path.basename(pwm_file).split(".")[0]
        cmd_str = "bedtools slop -g /wsu/home/groups/piquelab/footprint.fungei/byTreatAllRepsMerged/centipede/x8bFiles/new.chromSizes.txt -l 0 -r 1 -i {0} | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed -b stdin -c | cut -f4".format(pwm_file)
        # cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed -b {0} -c | cut -f4".format(pwm_file)
        raw_out = subprocess.check_output(cmd_str,shell=True)
        out_list = raw_out.split('\n')[:-1]
        df_beers_neg[pwm_name] = np.array(out_list,dtype='uint8')

        if idx % 100 == 0:
            df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_snps_only.h5','df',mode='w')

    df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_snps_only.h5','df',mode='w')

def make_beers_effect_snp_test_matrix():

    df_beers_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    df_beers_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)

    ### Populate positive matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        try:
            pwm_name = os.path.basename(effect_snp_file).split(".")[0]
            cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed -b {0} -c | cut -f4".format(effect_snp_file)
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_pos[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_pos[pwm_name] = 0

        if idx % 100 == 0:
            df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows.h5','df',mode='w')

    df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows.h5py','df',mode='w')

    ### Populate neg matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        try:
            pwm_name = os.path.basename(effect_snp_file).split(".")[0]
            cmd_str = "bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed -b {0} -c | cut -f4".format(effect_snp_file)
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_neg[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_neg[pwm_name] = 0
        if idx % 100 == 0:
            df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows.h5','df',mode='w')

    df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows.h5','df',mode='w')

def make_beers_effect_snp_test_matrix_1():

    df_beers_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    df_beers_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)

    ### Populate positive matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        try:
            pwm_name = os.path.basename(effect_snp_file).split(".")[0]
            cmd_str = "less {0} | sed -E '/1$/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.bed -b stdin -c | cut -f4".format(effect_snp_file)
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_pos[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_pos[pwm_name] = 0

        if idx % 100 == 0:
            df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows_1.h5','df',mode='w')

    df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows_1.h5','df',mode='w')

    ### Populate neg matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        try:
            pwm_name = os.path.basename(effect_snp_file).split(".")[0]
            cmd_str = "less {0} | sed -E '/1$/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.bed -b stdin -c | cut -f4".format(effect_snp_file)
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_neg[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_neg[pwm_name] = 0
        if idx % 100 == 0:
            df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows_1.h5','df',mode='w')

    df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows_1.h5','df',mode='w')

def make_beers_effect_snp_test_matrix_single_snp():

    df_beers_pos = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    df_beers_neg = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed',header=None,names=['chr','start','stop'],delim_whitespace=True)

    effect_snps = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/fullAllgenome/summaryPerMotif/anno2/*.gz')
    effect_snps = sorted(effect_snps)

    ### Populate positive matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        cmd_str = "less {0} | sed -E '/1$/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos.bed -b stdin -c | cut -f4".format(effect_snp_file)
        try:
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_pos[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_pos[pwm_name] = 0

        if idx % 100 == 0:
            df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snps_only.h5','df',mode='w')

    df_beers_pos.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snps_only.h5','df',mode='w')

    ### Populate neg matrix using beers windows and pwm files
    for idx,effect_snp_file in enumerate(effect_snps):
        pwm_name = os.path.basename(effect_snp_file).split(".")[0]
        cmd_str = "less {0} | sed -E '/1$/d' | bedtools intersect -a /wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg.bed -b stdin -c | cut -f4".format(effect_snp_file)
        try:
            raw_out = subprocess.check_output(cmd_str,shell=True)
            out_list = raw_out.split('\n')[:-1]
            df_beers_neg[pwm_name] = np.array(out_list,dtype='uint8')
        except:
            df_beers_neg[pwm_name] = 0
        if idx % 100 == 0:
            df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snps_only.h5','df',mode='w')

    df_beers_neg.to_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snps_only.h5','df',mode='w')

def load_beers_dsqtl_test_data():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.h5','df')

    X_test_pos =np.array(df_pos.iloc[:,3:])
    X_test_neg =np.array(df_neg.iloc[:,3:])

    X_dsqtl=np.vstack([X_test_pos,X_test_neg])
    X_dsqtl[X_dsqtl>1] = 1

    Y_dsqtl=np.hstack([np.ones(X_test_pos.shape[0]),np.zeros(X_test_neg.shape[0])])

    return(X_dsqtl,Y_dsqtl)

def load_beers_dsqtl_test_data_single_snp():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_snps_only.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_snps_only.h5','df')

    X_test_pos =np.array(df_pos.iloc[:,3:])
    X_test_neg =np.array(df_neg.iloc[:,3:])

    X_dsqtl=np.vstack([X_test_pos,X_test_neg])
    X_dsqtl[X_dsqtl>1] = 1

    Y_dsqtl=np.hstack([np.ones(X_test_pos.shape[0]),np.zeros(X_test_neg.shape[0])])

    return(X_dsqtl,Y_dsqtl)

def load_beers_dsqtl_effect_snp_test_data():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows.h5','df')

    X_test_pos =np.array(df_pos.iloc[:,3:])
    X_test_neg =np.array(df_neg.iloc[:,3:])

    X_dsqtl=np.vstack([X_test_pos,X_test_neg])
    X_dsqtl[X_dsqtl>1] = 1

    Y_dsqtl=np.hstack([np.ones(X_test_pos.shape[0]),np.zeros(X_test_neg.shape[0])])

    return(X_dsqtl,Y_dsqtl)

def load_beers_dsqtl_effect_snp_test_data_1():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snp_windows_1.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snp_windows_1.h5','df')

    X_test_pos =np.array(df_pos.iloc[:,3:])
    X_test_neg =np.array(df_neg.iloc[:,3:])

    X_dsqtl=np.vstack([X_test_pos,X_test_neg])
    X_dsqtl[X_dsqtl>1] = 1

    Y_dsqtl=np.hstack([np.ones(X_test_pos.shape[0]),np.zeros(X_test_neg.shape[0])])

    return(X_dsqtl,Y_dsqtl)

def load_beers_dsqtl_effect_snp_test_data_single_snp():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_effect_snps_only.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_effect_snps_only.h5','df')

    X_test_pos =np.array(df_pos.iloc[:,3:])
    X_test_neg =np.array(df_neg.iloc[:,3:])

    X_dsqtl=np.vstack([X_test_pos,X_test_neg])
    X_dsqtl[X_dsqtl>1] = 1

    Y_dsqtl=np.hstack([np.ones(X_test_pos.shape[0]),np.zeros(X_test_neg.shape[0])])

    return(X_dsqtl,Y_dsqtl)

def make_CENNTIPEDE_training_data():
    # from CNN_Models import cnn_helpers2 as CH
    pos_data = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_pos_df.txt',delim_whitespace=True)
    neg_data = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/dnase_neg_df.txt',delim_whitespace=True)
    X_pos =np.array(pos_data.iloc[:,3:-2])
    X_neg =np.array(neg_data.iloc[:,3:-2])
    X = np.vstack([X_pos,X_neg])
    Y=np.hstack([np.array(pos_data.dnase_peak),np.array(neg_data.dnase_peak)])
    Y[Y==-1] = 0
    X[X>1] = 1
    return(make_train_test_data(X,Y))

def make_CENNTIPEDE_training_data_2():
    # from CNN_Models import cnn_helpers2 as CH
    pos_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_pos.h5','df')
    neg_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_neg.h5','df')
    X_pos =np.array(pos_data.iloc[:,3:])
    X_neg =np.array(neg_data.iloc[:,3:])
    X = np.vstack([X_pos,X_neg])
    Y=np.hstack([np.ones(X_pos.shape[0]),np.zeros(X_neg.shape[0])])
    X[X>1] = 1
    return(make_train_test_data(X,Y))

def make_CENNTIPEDE_effect_snp_training_data():
    # from CNN_Models import cnn_helpers2 as CH
    pos_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos.h5','df')
    neg_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg.h5','df')
    X_pos =np.array(pos_data.iloc[:,3:])
    X_neg =np.array(neg_data.iloc[:,3:])
    X = np.vstack([X_pos,X_neg])
    Y=np.hstack([np.ones(X_pos.shape[0]),np.zeros(X_neg.shape[0])])
    X[X>1] = 1
    return(make_train_test_data(X,Y))

def make_CENNTIPEDE_effect_snp_training_data_1():
    # from CNN_Models import cnn_helpers2 as CH
    pos_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_pos_1.h5','df')
    neg_data = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_training_effectSNP_neg_1.h5','df')
    X_pos =np.array(pos_data.iloc[:,3:])
    X_neg =np.array(neg_data.iloc[:,3:])
    X = np.vstack([X_pos,X_neg])
    Y=np.hstack([np.ones(X_pos.shape[0]),np.zeros(X_neg.shape[0])])
    X[X>1] = 1
    return(make_train_test_data(X,Y))
def make_train_test_data(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, stratify=Y,random_state=172345)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test,random_state=1723450)
    data = CH.make_seq_data_dict([X_train, X_val, X_test, Y_train, Y_val,Y_test])
    return(data)

