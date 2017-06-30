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
from CNN_Models import cnn_helpers2 as CH
from collections import OrderedDict



def get_compendium_snp_predictions(foot,es):
    m,n = foot.shape
    y_pred = np.empty(m)

    for idx in np.arange(m):
        foot_vect_t = foot[idx].copy()
        ES_t = np.nonzero(es[idx])[0]

        prediction_t = np.squeeze(model.predict(foot_vect_t[None,:]))
        tmp = foot_vect_t[None,:].copy()
        tmp[0][ES_t] = 0
        es_pred_t = np.squeeze(model.predict(tmp))
        lo = DP.log_diffs(prediction_t,es_pred_t)

        y_pred[idx]=lo

    np.save('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_predictions',y_pred)
    df_snps = pd.read_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps.bed',header=None, names=['chr','start','stop'],delim_whitespace=True)
    df_snps['circuitSNPs'] = np.nan
    df_snps['circuitSNPs'] = y_pred
    df_snps.to_csv('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_effect_snps_and_predictions.tab',sep='\t',index=False)


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