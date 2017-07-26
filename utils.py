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
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints.h5')

        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-D':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints.h5')

        es = es.iloc[:,3:].values
        foot = foot.iloc[:,3:].values
        get_compendium_snp_predictions_ref_alt(foot,es,model,name)

    elif circuitSNPs_model == 'circuitSNPs-Window':
        es = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs.h5')
        foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprint_window_150.h5')

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
    es['circuitSNPs-D'] = y_pred
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
    return({'prec':prec,'rec':rec,'auprc':auprc,'fpr':fpr,'tpr':tpr,'auroc':auroc,'y_pred':y_pred,'prec_10':prec_10,'model_name':model_name})

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

def load_motif_dict():
    with open('/wsu/home/al/al37/al3786/CENNTIPEDE/motif_dict.pkl',"rb") as pkl_file:
        return(pickle.load(pkl_file))

def save_CNN_model(seq_model, model_name):
    seq_model.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/circuitSNP_model_{0}.h5".format(model_name))

def make_factor_dict():
    factor_df = pd.read_csv('/wsu/home/al/al37/al3786/factorNames.txt',header=None,delimiter="\t")
    factor_df[0] = factor_df[0].str.replace('\.[0-9]','')
    factor_dict = dict((zip(factor_df[0],factor_df[1])))
    return factor_dict

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