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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import minmax_scale

from keras.models import load_model
from CNN_Models import cnn_helpers2 as CH
from collections import OrderedDict
from scipy.stats import linregress

# from CENNTIPEDE import utils as CEUT

def load_model_replicates():
    repSNPs = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_50_10.tab',delim_whitespace=True)

    repSNPs_small = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_10_5.tab',delim_whitespace=True)

    repSNPs_big = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_200_40.tab',delim_whitespace=True)

    repSNPs_xsmall = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_5_3.tab',delim_whitespace=True)

    repSNPs_one_unit = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_1_0.tab',delim_whitespace=True)

    repSNPs_zero_unit = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds_0_0.tab',delim_whitespace=True)
    return([repSNPs_big,repSNPs,repSNPs_small,repSNPs_xsmall,repSNPs_one_unit,repSNPs_zero_unit])


def circD_beers_replicates(df,model_prefix):

    m,n = df.shape
    # y_pred = np.empty(m)
    y_pred = np.empty((m,10))
    model_dir = '/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/circuitSNP_model/replicate_test'
    models = glob.glob("{0}/{1}_*.h5".format(model_dir,model_prefix))
    for mod_idx,mod in enumerate(models):
        model = load_model(mod)
        print(mod_idx)
        for idx in np.arange(m):
            foot_vect_t = df.loc[idx,:'PBM0207'].copy()
            ES_ref = np.where(foot_vect_t==1)[0]
            ES_alt = np.where(foot_vect_t==2)[0]

            tmp_ref = foot_vect_t[None,:].copy()
            tmp_ref[0][ES_alt] = 0
            es_pred_ref = np.squeeze(model.predict(tmp_ref))

            # prediction_t = np.squeeze(model.predict(foot_vect_t[None,:]))

            tmp_alt = foot_vect_t[None,:].copy()
            tmp_alt[0][ES_ref] = 0
            tmp_alt[0][ES_alt] = 1
            es_pred_alt = np.squeeze(model.predict(tmp_alt))
            lo = log_diffs(es_pred_ref,es_pred_alt)

            y_pred[idx,mod_idx]=lo
            # if idx % 100000 == 0:
            # print("Predicted {0} SNPs".format(idx))

        # np.save('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/circuitSNPD_compendium_predictions_{0}'.format(name),y_pred)
        # df_snps = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed',header=None, names=['chr','start','stop'],delim_whitespace=True)
    try:
        df2=pd.DataFrame(data=y_pred)
        df2['label'] = df['label']
        df2.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/beers_replicate_preds.tab',sep='\t',index=False)
        return(df2)
    except:
        return(y_pred)


def prep_for_visualize_sparsity(count_matrix,motif_array):
    idx = np.argsort(np.count_nonzero(count_matrix,axis=1))
    M = count_matrix[:,idx][idx]
    df = pd.DataFrame(data=M,columns=np.array(motif_array)[idx],index=np.array(motif_array)[idx])
    cmap = sns.cubehelix_palette(n_colors=50,light=0.95, as_cmap=True,reverse=False)
    return(df,cmap)

def load_cooccurrences_3fold():
    counts_3 = np.load('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_both_bind_3.0.npy')
    return(counts_3)

def load_master_circuitSNPs():
    cs = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
    return(cs)

def load_motif_array():
    motif_file = "/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/motif_columns.txt"
    with open(motif_file,"r") as in_file:
        # motifs = in_file.readlines().rstrip()
        motifs = [x.strip() for x in in_file.readlines()]
    return(np.array(motifs))

def motif_pairs_counts_to_factors(count_mat, n=1, how='factor'):
    motif_file = "/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/motif_columns.txt"

    factor_dict = CEUT.make_factor_dict()

    with open(motif_file,"r") as in_file:
        motifs = [x.strip() for x in in_file.readlines()]

    count_dict = {}
    count_tracker = []

    part_idx = np.argpartition(np.ravel(np.tril(count_mat)), -n)[-n:]
    for idx,rav_idx in enumerate(part_idx):
        count = np.tril(count_mat)[np.unravel_index(rav_idx,np.tril(count_mat).shape)]
        if not count in count_dict.keys():
            count_dict[count] =  np.unravel_index(rav_idx,np.tril(count_mat).shape)

    count_ordered = np.sort(count_dict.keys())[::-1]
    for idx in count_ordered:
        if how=='factor':
            fac1 = factor_dict[motifs[count_dict[idx][0]].split(".")[0]]
            fac2 = factor_dict[motifs[count_dict[idx][1]].split(".")[0]]
            print("Factors = {0} | {1}".format(fac1,fac2))
            print("Number of co-occurrences = {0}".format(int(idx)))
            print("\n")
        elif how=='motif':
            fac1 = motifs[count_dict[idx][0]]
            fac2 = motifs[count_dict[idx][1]]
            print("Motifs = {0} | {1}".format(fac1,fac2))
            print("Number of co-occurrences = {0}".format(int(idx)))
            print("\n")
        elif how=="both":
            fac1 = factor_dict[motifs[count_dict[idx][0]].split(".")[0]]
            fac2 = factor_dict[motifs[count_dict[idx][1]].split(".")[0]]
            mot1 = motifs[count_dict[idx][0]]
            mot2 = motifs[count_dict[idx][1]]
            print("Motifs = {0} | {1}".format(mot1,mot2))
            print("Factors = {0} | {1}".format(fac1,fac2))
            print("Number of co-occurrences = {0}".format(int(idx)))
            print("----------")


def find_ES_cooccurrence_matrix():
    cs = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
    motifs = cs.loc[0,'M00001':'PBM0207'].index.tolist()
    M = cs.loc[:,'M00001':'PBM0207'].values
    M[M==2] = 1
    m,n = M.shape

    del cs

    count_mat = np.zeros((n,n))
    sub_array = np.array_split(M,5000)

    del M

    for x in sub_array:
        sub_count = x.T.dot(x)
        np.fill_diagonal(sub_count, 0)
        count_mat = count_mat + sub_count

    np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs")

def find_ES_cooccurrence_matrix_gpu(all_updown_opposite='all',log_odds = 3.0):
    import theano
    from theano import tensor as T

    cs = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
    motifs = cs.loc[0,'M00001':'PBM0207'].index.tolist()

    cs = cs[cs['circuitSNPs-D'].abs() > log_odds ]
    M = cs.loc[:,'M00001':'PBM0207'].values
    m,n = M.shape

    del cs
    sub_array = np.array_split(M,5000)

    del M
    gpuM = T.matrix()
    gpuC = T.matrix()

    gpu_dotprod = T.dot(gpuM.T, gpuM)
    gpu_dotprod_opp = T.dot(gpuM.T, gpuC)

    add_two_mat = T.add(gpuM,gpuC)

    f = theano.function([gpuM], gpu_dotprod,allow_input_downcast=True)
    o = theano.function([gpuM,gpuC], gpu_dotprod_opp,allow_input_downcast=True)

    a = theano.function([gpuM,gpuC],add_two_mat,allow_input_downcast=True)



    if all_updown_opposite=='all':
        count_mat = np.zeros((n,n))
        for idx,x in enumerate(sub_array):
            x[x==2] = 1
            count_mat = a(f(x),count_mat)
            print("Mult {0} of 5000".format(idx))
        np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_both_bind_{0}".format(log_odds), count_mat)
        return(count_mat)
    elif all_updown_opposite == 'updown':
        count_mat_up = np.zeros((n,n))
        count_mat_down = np.zeros((n,n))
        for idx,x_up in enumerate(sub_array):
            x_down = x_up.copy()

            x_up[x_up==2] = 0
            x_down[x_down==1] = 0
            x_down[x_down==2] = 1

            count_mat_up = a(f(x_up),count_mat_up)
            count_mat_down = a(f(x_down),count_mat_down)
            print("Mult {0} of 5000".format(idx))

        np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_up_bind_{0}".format(log_odds), count_mat_up)
        np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_down_bind_{0}".format(log_odds), count_mat_down)
        return(count_mat_up,count_mat_down)

    elif all_updown_opposite == 'opposite':
        count_opp = np.zeros((n,n))

        for idx,x_up in enumerate(sub_array):
            x_down = x_up.copy()

            x_up[x_up==2] = 0
            x_down[x_down==1] = 0
            x_down[x_down==2] = 1

            count_a = o(x_up,x_down)
            count_b = o(x_down,x_up)

            count_opp = a(a(count_a,count_b),count_opp)
            # count_mat_down = a(f(x_down),count_mat_down)
            print("Mult {0} of 5000".format(idx))

        np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_opposite_bind_{0}".format(log_odds), count_opp)
        return(count_opp)


def find_ES_cooccurrence_matrix_gpu_with_footprint_snps(log_odds = 0.0):
    import theano
    from theano import tensor as T

    cs_pre = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
    motifs = cs_pre.loc[0,'M00001':'PBM0207'].index.tolist()

    cs = cs_pre[cs_pre['circuitSNPs-D'].abs() > log_odds ]
    M = cs.loc[:,'M00001':'PBM0207'].values
    m,n = M.shape

    del cs
    sub_array = np.array_split(M,5000)
    del M

    foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')
    foot = foot.loc[:,'M00001':'PBM0207'].values
    foot = foot[cs_pre['circuitSNPs-D'].abs() > log_odds ]
    del cs_pre
    sub_array_foot = np.array_split(foot,5000)
    
    gpuM = T.matrix()
    gpuC = T.matrix()

    gpu_dotprod = T.dot(gpuM.T, gpuM)
    gpu_dotprod_opp = T.dot(gpuM.T, gpuC)

    add_two_mat = T.add(gpuM,gpuC)

    f = theano.function([gpuM], gpu_dotprod,allow_input_downcast=True)
    o = theano.function([gpuM,gpuC], gpu_dotprod_opp,allow_input_downcast=True)

    a = theano.function([gpuM,gpuC],add_two_mat,allow_input_downcast=True)

    count_mat_ES = np.zeros((n,n))
    count_mat_NonES = np.zeros((n,n))
    for idx,x_es in enumerate(sub_array):
        x_es[x_es==2] = 1
        x_foot = sub_array_foot[idx]
        x_foot[x_foot >1] = 1
        count_mat_NonES1 = o(x_es,x_foot)
        count_mat_NonES2 = o(x_foot,x_es)
        count_mat_NonES = a(a(count_mat_NonES1,count_mat_NonES2),count_mat_NonES)

        count_mat_ES = a(f(x_es),count_mat_ES)

        print("Mult {0} of 5000".format(idx))

    np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
    np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
    return(count_mat_ES,count_mat_NonES)

def prep_for_compendium_predictions(circuitSNPs_model,nn_model,name):
    """
    Usage:

    circuitSNPs_model = 'circuitSNPs' || 'circuitSNPs-D' || 'circuitSNPs-Window'

    nn_model = path to to trained circuitSNPs neural network model

    name = Used to name the output file. Describe the neural network model hidden units  i.e. '50_50' or '100_100'
    """
    try:
        model = load_model(nn_model)
    except Exception as e:
        raise e


    if circuitSNPs_model == 'circuitSNPs':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        # foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')


        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-D':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        # foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')

        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_ref_alt(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-lcl':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs_lcl.h5')
        # foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')

        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_ref_alt(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-Windows-lcl':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')

        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window_lcl_150.h5')

        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_windows(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-Window':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        # foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window_150.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window2_150.h5')


        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_windows(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-Windows-small':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window_5.h5')


        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_windows(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-Windows-100':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window_50.h5')


        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_windows(foot,es,model,name)

    else:
        raise ValueError(circuitSNPs_model + " is not a valid model")



def get_compendium_snp_predictions_windows(foot,es,model,name):
    m,n = foot.shape
    y_pred = np.empty(m)

    for idx in np.arange(m):
        foot_vect_t = foot[idx].copy()
        ES_ref = np.where(es[idx]==1)[0]
        ES_alt = np.where(es[idx]==2)[0]

        tmp_ref = foot_vect_t[None,:].copy()
        tmp_ref[0][ES_alt] = 0
        es_pred_ref = np.squeeze(model.predict(tmp_ref))

        # prediction_t = np.squeeze(model.predict(foot_vect_t[None,:]))

        tmp_alt = foot_vect_t[None,:].copy()
        tmp_alt[0][ES_ref] = 0
        tmp_alt[0][ES_alt] = 1
        es_pred_alt = np.squeeze(model.predict(tmp_alt))
        lo = log_diffs(es_pred_ref,es_pred_alt)

        y_pred[idx]=lo
        if idx % 100000 == 0:
            print("Predicted {0} SNPs".format(idx))

    np.save('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_predictions_windows_{0}'.format(name),y_pred)
    df_snps = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed',header=None, names=['chr','start','stop'],delim_whitespace=True)
    # df_snps['circuitSNPs'] = np.nan
    df_snps['circuitSNPs-W'] = y_pred
    df_snps.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs-W_predictions_{0}.tab'.format(name),sep='\t',index=False)


def get_compendium_snp_predictions_ref_alt(foot,es,model,name):
    m,n = foot.shape
    y_pred = np.empty(m)

    for idx in np.arange(m):
        foot_vect_t = foot[idx].copy()
        ES_ref = np.where(es[idx]==1)[0]
        ES_alt = np.where(es[idx]==2)[0]

        tmp_ref = foot_vect_t[None,:].copy()
        tmp_ref[0][ES_alt] = 0
        es_pred_ref = np.squeeze(model.predict(tmp_ref))

        # prediction_t = np.squeeze(model.predict(foot_vect_t[None,:]))

        tmp_alt = foot_vect_t[None,:].copy()
        tmp_alt[0][ES_ref] = 0
        tmp_alt[0][ES_alt] = 1
        es_pred_alt = np.squeeze(model.predict(tmp_alt))
        lo = log_diffs(es_pred_ref,es_pred_alt)

        y_pred[idx]=lo
        if idx % 100000 == 0:
            print("Predicted {0} SNPs".format(idx))

    np.save('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/circuitSNPD_compendium_predictions_{0}'.format(name),y_pred)
    df_snps = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed',header=None, names=['chr','start','stop'],delim_whitespace=True)
    df_snps['circuitSNPs-D'] = y_pred
    df_snps.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPsD_predictions_{0}.tab'.format(name),sep='\t',index=False)


def get_compendium_snp_predictions(foot,es,model,name):
    m,n = foot.shape
    y_pred = np.empty(m)

    for idx in np.arange(m):
        foot_vect_t = foot[idx].copy()
        ES_t = np.nonzero(es[idx])[0]

        prediction_t = np.squeeze(model.predict(foot_vect_t[None,:]))
        tmp = foot_vect_t[None,:].copy()
        tmp[0][ES_t] = 0
        es_pred_t = np.squeeze(model.predict(tmp))
        lo = log_diffs(prediction_t,es_pred_t)

        y_pred[idx]=lo

    np.save('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/circuitSNPs_compendium_predictions_{0}.npy'.format(name),y_pred)
    df_snps = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed',header=None, names=['chr','start','stop'],delim_whitespace=True)
    # df_snps['circuitSNPs'] = np.nan
    df_snps['circuitSNPs'] = y_pred
    df_snps.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs_predictions_{0}.tab'.format(name),sep='\t',index=False)


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

def make_pwm_conv_filters(kernel_size,width=4,rev_comp=True):
    pwms = glob.glob("/wsu/home/al/al37/al3786/EncodeDreamAnalysis/NewJaspar/PwmFiles/*.pfm")
    # pwms = glob.glob('/wsu/home/groups/piquelab/allCentipede/updatedModel/pwmRescan/recalibratedMotifs/*.pwm')
    num_filt = len(pwms)
    bad_pwm = 0
    pwm_arr = np.zeros((kernel_size,width,num_filt))
    idx = 0
    good_pwms_idx=[]
    for idx_pwm, i in enumerate(pwms):
        pwm = pd.read_csv(i,delim_whitespace=True,header=None,dtype=np.float64)
        # pwm = pd.read_csv(i,delim_whitespace=True,comment='#',dtype=np.float64)
        w = pwm.shape[1]
        # w = pwm.shape[0]
        if w >= kernel_size:
            bad_pwm +=1
            continue
        pwm=np.fliplr(pwm)
        pwm = pwm.T / np.sum(pwm.T,axis=1)[:,None]
        pwm[pwm<0.001] = 0.001
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
    prec_at_10_rec = prec[np.abs(rec-0.1).argmin()]
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    prec_10 = '{:0.4f}'.format(prec_at_10_rec)
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred,'prec_10':prec_10})

def circuitSNP_metrics(predictions, labels,model_name):
    # y_pred = model.predict(x, verbose=0)
    y_pred=predictions
    y = labels
    mets = CH.get_metrics(y, y_pred)
    prec,rec,_ = precision_recall_curve(y, y_pred)
    auPRC = average_precision_score(y,y_pred)
    fpr,tpr,_ = roc_curve(y,y_pred)
    auROC = auc(fpr,tpr)
    prec_at_10_rec = prec[np.abs(rec-0.1).argmin()]
    auprc = '{:0.4f}'.format(auPRC)
    auroc = '{:0.4f}'.format(auROC)
    prec_10 = '{:0.4f}'.format(prec_at_10_rec)
    rec_50 = '{:0.4f}'.format(mets['rec50'])
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred,'prec_10':prec_10,'model_name':model_name,'rec_50':rec_50})

def circuitSNP_test_new_model_metrics(test_data,test_y, model):
    x = test_data
    y_pred = model.predict(x, verbose=0)
    y = test_y
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

def log_diffs(prob1,prob2):
    ref_pred = prob1
    alt_pred = prob2
    ref_pred[ref_pred == 1.] = 0.999999
    ref_pred[ref_pred == 0.] = 0.000001
    alt_pred[alt_pred == 1.] = 0.999999
    alt_pred[alt_pred == 0.] = 0.000001
    log_odds = (np.log(ref_pred)-np.log(1-ref_pred)) - (np.log(alt_pred) -np.log(1-alt_pred))
    return(log_odds)

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



def load_motif_dict():
    with open('/wsu/home/al/al37/al3786/CENNTIPEDE/motif_dict.pkl',"rb") as pkl_file:
        return(pickle.load(pkl_file))

def save_CNN_model(seq_model, model_name):
    seq_model.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/circuitSNP_model/circuitSNP_model_{0}.h5".format(model_name))

def make_factor_dict():
    factor_df = pd.read_csv('/wsu/home/al/al37/al3786/factorNames.txt',header=None,delimiter="\t")
    # factor_df[0] = factor_df[0].str.replace('\.[0-9]','')
    factor_dict = dict((zip(factor_df[0],factor_df[1])))
    return factor_dict

def make_factor_cluster_dict():
    df = pd.read_csv('/nfs/rprdata/Anthony/data/combo/clusterFactorsAvg10.named.txt',index_col=0,delimiter='\t')
    df.reset_index(drop=True,inplace=True)
    df = df.drop(labels='Unnamed: 5',axis=1)
    clust_dict = {}
    for r in df.iterrows():
        clust_dict[r[1]['pwmId']] = r[1]['clusterId']
    return(clust_dict)

def get_best_motifs_and_factors_neg(df,min_score=7):
    """
    pandas df should have the motifs as column names, and one column named "score" - from the circuitSNP prediction, and one column named "label" - the true class label
    """
    factor_dict = make_factor_dict()
    new_list = []

    for i in df[(df.label == 0.0) & (df.score<=min_score)].sort_values(by=['score'],ascending=False).iterrows():

        new_dict={}
        new_dict['score'] = i[1].score
        motifs = df.columns[[i[1]==1][0]].tolist()[:-1]

        new_dict["motifs"] = motifs
        factors = [factor_dict[m] for m in motifs]
        factors = list(OrderedDict.fromkeys(factors))
        new_dict["factors"] = factors
        new_list.append(new_dict)
    return(pd.DataFrame(new_list))

def get_best_motifs_and_factors(df,min_score=7):
    """
    pandas df should have the motifs as column names, and one column named "score" - from the circuitSNP prediction, and one column named "label" - the true class label
    """
    factor_dict = make_factor_dict()
    new_list = []

    for i in df[(df.label == 1.0) & (df.score>=min_score)].sort_values(by=['score'],ascending=False).iterrows():

        new_dict={}
        new_dict['score'] = i[1].score
        motifs = df.columns[[i[1]==1][0]].tolist()[:-1]

        new_dict["motifs"] = motifs
        factors = [factor_dict[m] for m in motifs]
        factors = list(OrderedDict.fromkeys(factors))
        new_dict["factors"] = factors

        new_list.append(new_dict)
    return(pd.DataFrame(new_list))