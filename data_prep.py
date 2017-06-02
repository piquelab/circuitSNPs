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
from CNN_Models import cnn_helpers2 as CH
from keras import regularizers

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

def load_beers_dsqtl_test_data():
    df_pos = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_pos_windows.h5','df')
    df_neg = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/beers_test_neg_windows.h5','df')

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


def make_CENNTIPEDE_model(data):
    ada = Adadelta()
    input0 = Input(shape=(data['train_data_X'].shape[1],))
    lay1 = Dense(500,activation='relu',name='HL1',use_bias=True)(input0)
    lay2 = Dense(500,activation='relu',name='HL2',use_bias=True)(lay1)
    do1 = Dropout(0.25)(lay2)
    lay3 = Dense(500,activation='relu',name='HL3',use_bias=True)(do1)
    do2 = Dropout(0.25)(lay3)
    predictions = Dense(1, activation='sigmoid')(do2)

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


def get_prc_roc_validation(model, data):
    x = data['test_data_X']
    y = data['test_data_Y']
    y_pred = model.predict(x, verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    prec_at_10_rec = prec[np.abs(rec-0.1).argmin()]
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    prec_10 = '{:0.4f}'.format(prec_at_10_rec)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred,'prec_10':prec_10})

def plot_auprc_CENNTIPEDE_validation(*met_dicts):
    plt.figure(figsize=(8,8))
    for idx,met_dict in enumerate(met_dicts):
        plt.plot(met_dict['rec'],met_dict['prec'],label="Model {1} $|$ {0} $|$ {2}".format(met_dict['auprc'],idx,met_dict['prec_10']))
    plt.grid()
    plt.axvline(0.1,color='grey',ls="--")
    plt.ylabel('Precision',fontsize = 20)
    plt.xlabel('Recall',fontsize = 20)
    leg = plt.legend(loc='upper right',title='Model $|$ auPRC $|$ Prec @ 10\% Rec',fontsize=16,fancybox=True,shadow=False)
    plt.setp(leg.get_title(),fontsize=20)
    plt.title("CENNTIPEDE",fontsize=22)

def permute_ES_preds(X_dsqtl,X_dsqtl_pred,X_mask,model):
    preds = np.zeros(X_dsqtl.shape[0])
    for idx_d,i in enumerate(X_dsqtl):
        max_diff = 0
        pred1 = X_dsqtl_pred[idx_d]
        i_copy = i.copy()
        for idx in np.where(X_mask[idx_d]==1)[0]:
            i_copy1 = i_copy.copy()
            i_copy1[idx] = 0
            pred = model.predict(np.expand_dims(i_copy1,axis=0))
            diff = np.absolute(log_diffs(pred1,pred))
            if diff > max_diff:
                preds[idx_d] = diff
                max_diff = diff
    return(preds)

def get_prc_roc_validation_joint_model(model, data1,data2,data3):
    # from CNN_Models import cnn_helpers2 as CH
    x = [data1['test_data_X'],data2['test_data_X'],data3['test_data_X']]
    y = data1['test_data_Y']
    y_pred = model.predict(x, verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_validation_pwm_model(model, data1,data2,data3,good_pwms_idx):
    # from CNN_Models import cnn_helpers2 as CH
    x = [data1['test_data_X'][:,good_pwms_idx],data2['test_data_X'][:,good_pwms_idx],data3['test_data_X']]
    y = data1['test_data_Y']
    y_pred = model.predict(x, verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_test(model, x,y):
    y_pred = model.predict(x, verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_test_effect_SNP_model(model, x1,x2,y):
    y_pred = model.predict([x1,x2], verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_test_joint_model(model, x1,x2,x3,y):
    y_pred = model.predict([x1,x2,x3], verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_test_pwm_model(model, x1,x2,x3,y,good_pwms_idx):
    y_pred = model.predict([x1[:,good_pwms_idx],x2[:,good_pwms_idx],x3], verbose=0)
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred})

def get_prc_roc_prediction(predictions, labels):
    # y_pred = model.predict(x, verbose=0)
    y_pred=predictions
    y = labels
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc})

def log_diffs(prob1,prob2):
    ref_pred = prob1
    alt_pred = prob2
    ref_pred[ref_pred == 1.] = 0.999999
    ref_pred[ref_pred == 0.] = 0.000001
    alt_pred[alt_pred == 1.] = 0.999999
    alt_pred[alt_pred == 0.] = 0.000001
    log_odds = (np.log(ref_pred)-np.log(1-ref_pred)) - (np.log(alt_pred) -np.log(1-alt_pred))
    return(log_odds)

def permute_mask(X_dsqtl,X_dsqtl_pred,X_mask,model):
    preds = np.zeros(X_dsqtl.shape[0])
    for idx_d,i in enumerate(X_dsqtl):
        max_diff = 0
        pred1 = X_dsqtl_pred[idx_d]
        i_copy = i.copy()
        for idx in np.where(X_mask[idx_d]==1)[0]:
            i_copy1 = i_copy.copy()
            i_copy1[idx] = 0
            pred = model.predict(np.expand_dims(i_copy1,axis=0))
            diff = np.absolute(log_diffs(pred1,pred))
            if diff > max_diff:
                preds[idx_d] = diff
                max_diff = diff
    return(preds)

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

def fit_CENNTIPEDE_Effect_SNP_model(model, data1, data2):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'],data2['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'],data2['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def fit_CENNTIPEDE_CNNtipede_model(model, data1, data2, data3):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'],data2['train_data_X'],data3['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'],data2['val_data_X'],data3['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)

def fit_CENNTIPEDE_CNNtipede_pwm_model(model, data1, data2, data3,good_pwms_idx):
    model_earlystopper = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
    model.fit([data1['train_data_X'][:,good_pwms_idx],data2['train_data_X'][:,good_pwms_idx],data3['train_data_X']], data1['train_data_Y'],
        epochs=30,
        shuffle=True,
        validation_data = ([data1['val_data_X'][:,good_pwms_idx],data2['val_data_X'][:,good_pwms_idx],data3['val_data_X']],data1['val_data_Y']),
        callbacks=[model_earlystopper],
        verbose=2)
    return(model)


def make_train_test_data(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y,random_state=7234)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test,random_state=72340)
    data = CH.make_seq_data_dict([X_train, X_val, X_test, Y_train, Y_val,Y_test])
    return(data)

def make_pwm_conv_filters(kernel_size,width=4,rev_comp=True):
    # pwms = glob.glob("/wsu/home/al/al37/al3786/EncodeDreamAnalysis/NewJaspar/PwmFiles/*.pfm")
    pwms = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/recalibratedMotifs/*.pwm')
    num_filt = len(pwms)
    bad_pwm = 0
    pwm_arr = np.zeros((kernel_size,width,num_filt))
    idx = 0
    good_pwms_idx=[]
    for idx_pwm, i in enumerate(pwms):
        # pwm = pd.read_csv(i,delim_whitespace=True,header=None,dtype=np.float64)
        pwm = pd.read_csv(i,delim_whitespace=True,comment='#',dtype=np.float64)
        # w = pwm.shape[1]
        w = pwm.shape[0]
        if w >= kernel_size:
            bad_pwm +=1
            continue
        # pwm=np.fliplr(pwm)
        # pwm = pwm.T / np.sum(pwm.T,axis=1)[:,None]
        # pwm[pwm<0.001] = 0.001
        pwm = np.log2(pwm)+2

        start = np.round((kernel_size/2)-(w/ 2))
        if width==4:
            pwm_arr[start:start+w,:,idx] = pwm
        if width==8:
            pwm_arr[start:start+w,:4,idx] = pwm
            pwm_arr[start:start+w,4:,idx] = pwm
        idx += 1
        good_pwms_idx.append(idx_pwm)
    pwm_arr = pwm_arr[:,:,:num_filt-bad_pwm]
    if rev_comp:
        conv_weights = np.concatenate([pwm_arr,pwm_arr[::-1,:,::-1]],axis=2)
    else:
        conv_weights = pwm_arr

    num_filts = conv_weights.shape[-1]

    return(conv_weights, num_filts, good_pwms_idx)