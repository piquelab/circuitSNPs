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

import seaborn as sns; sns.set(color_codes=True)
from CENNTIPEDE import utils as CEUT

def prep_motif_cluster_heatmap(count_matrix,motifs,clust_dict,cutoff=50,scale=False,reorder=False):

    clust_names = [clust_dict[x] for x in motifs]
    vals, inverse, count = np.unique(clust_names,
                                    return_inverse=True,
                                    return_counts=True)

    idx_vals_repeated = np.where(count > 0)[0]
    vals_repeated = vals[idx_vals_repeated]

    rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])


    M = np.zeros((len(res),count_matrix.shape[0]))
    MM = np.zeros((len(res),len(res)))
    for idx,i in enumerate(res):
        M[idx] = count_matrix[i].sum(axis=0)

    for idx, i in enumerate(res):
        MM[idx] = M[:,i].sum(axis=1)

    clust_names_reduce = []
    for idx,i in enumerate(res):
        clust_names_reduce.append(np.array(clust_names)[i[0]])

    df_clust = pd.DataFrame(MM,columns=clust_names_reduce,index=clust_names_reduce)

    if reorder:
        clust_idx = np.argsort(np.count_nonzero(df_clust.values,axis=1))

        M = df_clust.values[:,clust_idx][clust_idx]
        df_clust_sort = pd.DataFrame(data=M,columns=np.array(df_clust.columns.tolist())[clust_idx],index=np.array(df_clust.index.tolist())[clust_idx])

        df_clust_sort_small = df_clust_sort.iloc[-cutoff:,-cutoff:].copy()

        if scale:
            D = np.add.outer(np.diag(df_clust_sort_small),np.diag(df_clust_sort_small))
            df_clust_sort_small = np.divide(df_clust_sort_small,D)
            np.fill_diagonal(df_clust_sort_small.values,1)

        return(df_clust_sort, df_clust_sort_small)

    else:
        return(df_clust)



def prep_for_visualize_sparsity(count_matrix,motif_array):
    idx = np.argsort(np.count_nonzero(count_matrix,axis=1))
    M = count_matrix[:,idx][idx]
    df = pd.DataFrame(data=M,
                    columns=np.array(motif_array)[idx],
                    index=np.array(motif_array)[idx])
    cmap = sns.cubehelix_palette(n_colors=50,light=0.95, as_cmap=True,reverse=False)
    return(df,cmap)

def load_cooccurrences_3fold():
    counts_3 = np.load('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_both_bind_3.0.npy')
    return(counts_3)

def load_cooccurrences_both():
    counts = np.load('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu.npy')
    return(counts)

def plot_circuitSNPs_vs_gkm_svm(df,tissue,name,direction=True,all_4=False,save=False):
    if direction:
        if not all_4:
            plt.figure(figsize=(8,8))
            plt.ylabel("gkm\_scores",fontsize=16)
            plt.xlabel("circuitSNP-D prediction",fontsize=16)
            reg = linregress(df['circuitSNPs-D'],df['gkm_score'])
            x = np.arange(df['circuitSNPs-D'].min(),
                            df['circuitSNPs-D'].max()+1,
                            1)
            abline_values = [reg[0] * i + reg[1] for i in x]
            p_val = '{:0.3e}'.format(reg[3])
            r_val = '{:0.3f}'.format(reg[2])
            plt.scatter(df['circuitSNPs-D'],
                        df.gkm_score)
            plt.plot(x,abline_values,'r')
            plt.grid()
            plt.title("r = {0}    p\_val = {1}".format(r_val,p_val))
            if save:
                plt.savefig("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/1KG_predictions/circuitSNPs_D.{0}.png".format(tissue),dpi=300,transparent=True)
        else:

            f, ax_arr = plt.subplots(2,2,figsize=(8,8))
            ax_arr = ax_arr.reshape(-1)
            gkm_order=["gkm\_scores","$|$gkm scores$|$","gkm scores","$|$gkm scores$|$"]
            circuitSNP_order=["circuitSNP-D prediction","circuitSNP-D prediction","$|$circuitSNP-D prediction$|$","$|$circuitSNP-D prediction$|$"]
            for idx in [0,1,2,3]:
                ax_arr[idx].set_ylabel(gkm_order[idx],fontsize=16)
                ax_arr[idx].set_xlabel(circuitSNP_order[idx],fontsize=16)
                # ax_arr[idx].set_ylim()
                # ax_arr[idx].axhline(true_vals[idx], color='green')
                if idx==0:
                    reg = linregress(df['circuitSNPs-D'],df['gkm_score'])
                    x = np.arange(df['circuitSNPs-D'].min(),
                                    df['circuitSNPs-D'].max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    # p_val = reg[3]
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df['circuitSNPs-D'],
                                df.gkm_score)
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                elif idx==1:
                    reg = linregress(df['circuitSNPs-D'],df['gkm_score'].abs())
                    x = np.arange(df['circuitSNPs-D'].min(),
                                    df['circuitSNPs-D'].max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df['circuitSNPs-D'],
                                df.gkm_score.abs())
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                elif idx==2:
                    reg = linregress(df['circuitSNPs-D'].abs(),df['gkm_score'])
                    x = np.arange(df['circuitSNPs-D'].abs().min(),
                                    df['circuitSNPs-D'].abs().max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df['circuitSNPs-D'].abs(),
                                df.gkm_score)
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                else:
                    reg = linregress(df['circuitSNPs-D'].abs(),df['gkm_score'].abs())
                    x = np.arange(df['circuitSNPs-D'].abs().min(),
                                    df['circuitSNPs-D'].abs().max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df['circuitSNPs-D'].abs(),
                                df.gkm_score.abs())
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                ax_arr[idx].grid()

                f.suptitle("circuitSNP-D \& gkmSVM scores in {0}".format(tissue),fontsize=20)
                f.tight_layout()
                f.subplots_adjust(top=0.9)
                if save:
                    plt.savefig("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/1KG_predictions/circuitSNPs_D.{0}_{1}.png".format(tissue,name),dpi=300,transparent=True)
    else:
        if not all_4:
            plt.figure(figsize=(8,8))
            plt.ylabel("gkm\_scores",fontsize=16)
            plt.xlabel("circuitSNPs prediction",fontsize=16)
            reg = linregress(df['circuitSNPs'],df['gkm_score'])
            x = np.arange(df['circuitSNPs'].min(),
                            df['circuitSNPs'].max()+1,
                            1)
            abline_values = [reg[0] * i + reg[1] for i in x]
            p_val = '{:0.3e}'.format(reg[3])
            r_val = '{:0.3f}'.format(reg[2])
            plt.scatter(df['circuitSNPs'],
                        df.gkm_score)
            plt.plot(x,abline_values,'r')
            plt.title("r = {0}    p\_val = {1}".format(r_val,p_val))
            if save:
                plt.savefig("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/1KG_predictions/circuitSNPs.{0}.png".format(tissue),dpi=300,transparent=True)
        else:
            f, ax_arr = plt.subplots(2,2,figsize=(8,8))
            ax_arr = ax_arr.reshape(-1)
            gkm_order=["gkm\_scores","$|$gkm scores$|$","gkm scores","$|$gkm scores$|$"]
            circuitSNP_order=["circuitSNPs prediction","circuitSNPs prediction","$|$circuitSNPs prediction$|$","$|$circuitSNPs prediction$|$"]
            for idx in [0,1,2,3]:
                ax_arr[idx].set_ylabel(gkm_order[idx],fontsize=16)
                ax_arr[idx].set_xlabel(circuitSNP_order[idx],fontsize=16)
                # ax_arr[idx].set_ylim()
                # ax_arr[idx].axhline(true_vals[idx], color='green')
                if idx==0:
                    reg = linregress(df['circuitSNPs'],df['gkm_score'])
                    x = np.arange(df['circuitSNPs'].min(),
                                    df['circuitSNPs'].max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    # p_val = reg[3]
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df.circuitSNPs,
                                df.gkm_score)
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                elif idx==1:
                    reg = linregress(df['circuitSNPs'],df['gkm_score'].abs())
                    x = np.arange(df['circuitSNPs'].min(),
                                    df['circuitSNPs'].max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df.circuitSNPs,
                                df.gkm_score.abs())
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                elif idx==2:
                    reg = linregress(df['circuitSNPs'].abs(),df['gkm_score'])
                    x = np.arange(df['circuitSNPs'].abs().min(),
                                    df['circuitSNPs'].abs().max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df.circuitSNPs.abs(),
                                df.gkm_score)
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                else:
                    reg = linregress(df['circuitSNPs'].abs(),df['gkm_score'].abs())
                    x = np.arange(df['circuitSNPs'].abs().min(),
                                    df['circuitSNPs'].abs().max()+1,
                                    1)
                    abline_values = [reg[0] * i + reg[1] for i in x]
                    p_val = '{:0.3e}'.format(reg[3])
                    r_val = '{:0.3f}'.format(reg[2])
                    ax_arr[idx].scatter(df.circuitSNPs.abs(),
                                df.gkm_score.abs())
                    ax_arr[idx].plot(x,abline_values,'r')
                    ax_arr[idx].set_title("r = {0}    p\_val = {1}".format(r_val,p_val))
                ax_arr[idx].grid()
            f.suptitle("circuitSNP \& gkmSVM scores in {0}".format(tissue),fontsize=20)
            f.tight_layout()
            f.subplots_adjust(top=0.9)
            if save:
                plt.savefig("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/1KG_predictions/circuitSNPs.{0}_{1}.png".format(tissue,name),dpi=300,transparent=True)

def plot_auprc_circuitSNPs_predictions(y_true,y_preds,model_names):
    plt.figure(figsize=(10,10))
    for idx,y_pred in enumerate(y_preds):
        prec,rec,_ = precision_recall_curve(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        prec_at_10_rec = prec[np.abs(rec-0.1).argmin()]
        auprc = '{:0.4f}'.format(auprc)
        prec_at_10_rec = '{:0.4f}'.format(prec_at_10_rec)
        plt.plot(rec,prec,label="{0} $|$ {1} $|$ {2}".format(model_names[idx],auprc,prec_at_10_rec),lw=3)
    plt.grid()
    plt.axvline(0.1,color='grey',ls="--")
    plt.ylabel('Precision',fontsize = 20)
    plt.xlabel('Recall',fontsize = 20)
    leg = plt.legend(loc='upper right',title='Model $|$ auPRC $|$ Prec @ 10\% Rec',fontsize=16,fancybox=True,shadow=False)
    plt.setp(leg.get_title(),fontsize=20)
    plt.title("circuitSNPs Prediction",fontsize=22)


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

def plot_auprc_circuitSNPs_validation(*met_dicts):
    plt.figure(figsize=(10,10))
    for idx,met_dict in enumerate(met_dicts):
        plt.plot(met_dict['rec'],met_dict['prec'],label="Model {1} $|$ {0} $|$ {2}".format(met_dict['auprc'],idx,met_dict['prec_10']),lw=3)
    plt.grid()
    plt.axvline(0.1,color='grey',ls="--")
    plt.ylabel('Precision',fontsize = 24)
    plt.xlabel('Recall',fontsize = 24)
    leg = plt.legend(loc='upper right',title='Model $|$ auPRC $|$ Prec @ 10\% Rec',fontsize=16,fancybox=True,shadow=False)
    plt.setp(leg.get_title(),fontsize=20)
    plt.title("circuitSNPs Model Validation",fontsize=28)
    plt.tight_layout()
    save_fig=True
    if save_fig:
        plt.savefig("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/model_validation_prc.pdf")

def plot_auprc_circuitSNPs_predictions_2(*met_dicts):
    plt.figure(figsize=(8,8))
    for idx,met_dict in enumerate(met_dicts):
        plt.plot(met_dict['rec'],met_dict['prec'],label="{1} $|$ {0} $|$ {2}".format(met_dict['auprc'],met_dict['model_name'],met_dict['prec_10']))
    plt.grid()
    plt.axvline(0.1,color='grey',ls="--")
    plt.ylabel('Precision',fontsize = 20)
    plt.xlabel('Recall',fontsize = 20)
    leg = plt.legend(loc='upper right',title='Model $|$ auPRC $|$ Prec @ 10\% Rec',fontsize=16,fancybox=True,shadow=False)
    plt.setp(leg.get_title(),fontsize=20)
    plt.title("circuitSNPs",fontsize=22)

def plot_auprc_circuitSNPs_predictions_3(*met_dicts):
    plt.figure(figsize=(8,8))
    for idx,met_dict in enumerate(met_dicts):
        plt.plot(met_dict['rec'],met_dict['prec'],label="{1} $|$ {0} $|$ {2}".format(met_dict['auprc'],met_dict['model_name'],met_dict['prec_10']))
    plt.grid()
    plt.axvline(0.1,color='grey',ls="--")
    plt.ylabel('Precision',fontsize = 20)
    plt.xlabel('Recall',fontsize = 20)
    plt.xlim(0,0.2)
    leg = plt.legend(loc='upper right',title='Model $|$ auPRC $|$ Prec @ 10\% Rec',fontsize=16,fancybox=True,shadow=False)
    plt.setp(leg.get_title(),fontsize=20)
    plt.title("circuitSNPs",fontsize=22)