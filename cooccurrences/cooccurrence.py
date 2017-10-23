import os,sys
import glob
import subprocess
import numpy as np
import pandas as pd
import natsort
import h5py
import cPickle as pickle

import re
import random

import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
import theano
from theano import tensor as T

def cooccurrence_pvals1(four_arr_table, count_thresh=250,odds_cut=5,fdr=0.3):
    
    
    def np_fisher(a,b,c,d):
        oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]])
        return(oddsratio,pvalue)
    np_fisher2 = np.vectorize(np_fisher)

    mats_mod1 = np.array(four_arr_table)
    
    np.fill_diagonal(mats_mod1[0],0)
    np.fill_diagonal(mats_mod1[1],0)
    np.fill_diagonal(mats_mod1[2],0)
    np.fill_diagonal(mats_mod1[3],0)

    ct = count_thresh
    idx = np.where((mats_mod1[0]>=ct) & (mats_mod1[1]>=ct) & (mats_mod1[2]>=ct) & (mats_mod1[3]>=ct))

    idx1 = np.unique(idx[0])
    idx2 = np.unique(idx[1])
    idx3 = np.union1d(idx1,idx2)

    mats_thresh = mats_mod1[:,idx3,:][:,:,idx3]

    _, pvals = np_fisher2(mats_thresh[0],mats_thresh[1],mats_thresh[2],mats_thresh[3])
    
    odds = (mats_thresh[0]+1/mats_thresh[1]+1)/(mats_thresh[2]+1/mats_thresh[3]+1)
    bh_adjusted = multipletests(pvals.ravel(), alpha=fdr,method='fdr_bh')

    pvals_adj = np.reshape(bh_adjusted[1],pvals.shape)

    return(mats_thresh,idx3,odds,pvals,pvals_adj)

def cooccurrence_pvals_rasqual(four_arr_table, count_thresh=250,odds_cut=5,fdr=0.3):
    
    
    def np_fisher(a,b,c,d):
        oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]])
        return(oddsratio,pvalue)
    np_fisher2 = np.vectorize(np_fisher)

    mats_mod1 = np.array(four_arr_table)
    
    np.fill_diagonal(mats_mod1[0],0)
    np.fill_diagonal(mats_mod1[1],0)
    np.fill_diagonal(mats_mod1[2],0)
    np.fill_diagonal(mats_mod1[3],0)

    ct = count_thresh
    idx = np.where((mats_mod1[0]>=ct) & (mats_mod1[1]>=ct) & (mats_mod1[2]>=ct) & (mats_mod1[3]>=ct))

    idx1 = np.unique(idx[0])
    idx2 = np.unique(idx[1])
    idx3 = np.union1d(idx1,idx2)

    mats_thresh = mats_mod1[:,idx3,:][:,:,idx3]

    _, pvals = np_fisher2(mats_thresh[0],mats_thresh[1],mats_thresh[2],mats_thresh[3])
    
    odds = (mats_thresh[0]+1/mats_thresh[1]+1)/(mats_thresh[2]+1/mats_thresh[3]+1)
    bh_adjusted = multipletests(pvals.ravel(), alpha=fdr,method='fdr_bh')

    pvals_adj = np.reshape(bh_adjusted[1],pvals.shape)

    return(mats_thresh,idx3,odds,pvals,pvals_adj)

def load_motif_array():
    motif_file = "/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/motif_columns.txt"
    with open(motif_file,"r") as in_file:
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

def cooccurrence_enrichment_compendium(data_mat,circuitSNP_thresh = 3.0, test_type='foot',split_size=5000):

    ES = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    FS = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    FS_ES = data_mat.loc[:,'M00001':'PBM0207'].values >= 1
    
    CS = np.abs(data_mat['circuitSNPs-5-3-5'].values) >= circuitSNP_thresh
    CS = CS[:,np.newaxis] 
    
    m,n = ES.shape
    
    ES = np.array_split(ES,split_size)
    CS = np.array_split(CS,split_size)
    FS = np.array_split(FS,split_size)
    FS_ES = np.array_split(FS_ES,split_size)

    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(ES,CS,FS,FS_ES)):
        es=z[0]
        cs=z[1]
        fs=z[2]
        fs_es = z[3]
        
        es = es.astype(np.int)
        cs = cs.astype(np.int)
        fs = fs.astype(np.int)
        fs_es = fs_es.astype(np.int)
        

        if test_type=='foot':
            esd = fs_es
            e1sd = 1-fs_es
        elif test_type=='effect':
            esd = es
            e1sd = 1-es
        elif test_type=='circuit':
            esd = es*cs
            e1sd = 1-(es*cs)
        #A
        count_mat_A = add_mat_th(dotprod_2th(esd,fs_es),count_mat_A)

        #B
        count_mat_B = add_mat_th(dotprod_2th(e1sd,fs_es),count_mat_B)

        #C
        count_mat_C = add_mat_th(dotprod_2th(esd,1-fs_es),count_mat_C)

        #D
        count_mat_D = add_mat_th(dotprod_2th(e1sd,1-fs_es),count_mat_D)

        if idx % 20 == 0:
            print("{0} of {1}".format(idx,split_size))

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)

def cooccurrence_enrichment_gpu_1_rev2(log_odds = 3.0,data_mat=None):
    import theano
    from theano import tensor as T

    if data_mat is None:
        print("Loading circuitSNPs Data")
        data_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs_with_footsnps.h5')
        data_mat = data_mat[(data_mat.snp==1.0) & (data_mat['es-count']>0)]
    else:
        print("circuitSNPs Data provided")

    F = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    
    S = data_mat['circuitSNPs-D'].abs() >= log_odds 

    X = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    m,n = X.shape

    del data_mat
    del foot
    X = np.array_split(X,5000)
    S = np.array_split(S,5000)
    F = np.array_split(F,5000)
    
    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(X,S,F)):
        e=z[0]
        s=z[1]
        f=z[2]

        e = e.astype(np.int)
        s = s[:,None].astype(np.int)
        f = f.astype(np.int)
        
        #A
        es = e*s
        count_mat_A = add_mat_th(dotprod_2th(es,f),count_mat_A)

        #B
        e1s = e*(1-s)
        count_mat_B = add_mat_th(dotprod_2th(e1s,f),count_mat_B)

        #C
        # es = e*s
        count_mat_C = add_mat_th(dotprod_2th(es,(1-f)),count_mat_C)

        #D
        # d = e*(1-s)
        count_mat_D = add_mat_th(dotprod_2th(e1s,(1-f)),count_mat_D)

        if idx % 500 == 0:
            print("Mult {0} of 5000".format(idx))

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)    

def cooccurrence_enrichment_gpu_rasqual(data_mat,circuitSNP_thresh = 3.0, test_type='foot',split_size=1):

    ES_up = data_mat.loc[:,'M00001':'PBM0207'].values == 3
    ES_down = data_mat.loc[:,'M00001':'PBM0207'].values == 2
    

    # FS = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    # FS_ES = data_mat.loc[:,'M00001':'PBM0207'].values >= 1
    
    CS = np.abs(data_mat['circuitSNPs-5-3-5'].values) >= circuitSNP_thresh
    CS = CS[:,np.newaxis] 
    
    RAS = np.sign(np.log(data_mat['effect_size']/(1-data_mat['effect_size'])))
    RAS = RAS[:,np.newaxis]
    
    m,n = ES_up.shape
    
    ES_up = np.array_split(ES_up,split_size)
    ES_down = np.array_split(ES_down,split_size)
    CS = np.array_split(CS,split_size)
    # FS = np.array_split(FS,split_size)
    RAS = np.array_split(RAS,split_size)
    # FS_ES = np.array_split(FS_ES,split_size)

    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(ES_up,ES_down,CS,RAS)):
        es_up=z[0]
        es_down=z[1]
        cs=z[2]
        ras=z[3]
        
        es_up = es_up.astype(np.int)
        es_down = -1*es_down.astype(np.int)
        es = es_up + es_down

        ((np.sign(A) * np.sign(B[:,np.newaxis])) ==1).astype(np.int)

        cs = cs.astype(np.int)
        ras = ras.astype(np.int)
        # fs = fs.astype(np.int)
        # fs_es = fs_es.astype(np.int)
        

        # if test_type=='foot':
        esd = fs_es*ras
            e1sd = fs_es*(1-ras)
        # elif test_type=='effect':
            # esd = es
            # e1sd = 1-es
        # elif test_type=='circuit':
            # esd = es*cs
            # e1sd = 1-(es*cs)
        #A
        count_mat_A = add_mat_th(dotprod_2th(esd,fs_es),count_mat_A)

        #B
        count_mat_B = add_mat_th(dotprod_2th(e1sd,fs_es),count_mat_B)

        #C
        count_mat_C = add_mat_th(dotprod_2th(esd,1-fs_es),count_mat_C)

        #D
        count_mat_D = add_mat_th(dotprod_2th(e1sd,1-fs_es),count_mat_D)

        if idx % 20 == 0:
            print("{0} of {1}".format(idx,split_size))

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)    

def cooccurrence_enrichment_gpu_1_dsqtl(data_mat,circuitSNP_thresh = 3.0, test_type='circuit'):

    ES = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    FS = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    FS_ES = data_mat.loc[:,'M00001':'PBM0207'].values >= 1
    
    CS = np.abs(data_mat['5'].values) >= circuitSNP_thresh
    CS = CS[:,np.newaxis] 
    
    DSQTL = data_mat['label_x'].values == 1
    DSQTL = DSQTL[:,np.newaxis]
    
    m,n = ES.shape
    
    ES = np.array_split(ES,2)
    CS = np.array_split(CS,2)
    FS = np.array_split(FS,2)
    DSQTL = np.array_split(DSQTL,2)
    FS_ES = np.array_split(FS_ES,2)

    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(ES,CS,FS,DSQTL,FS_ES)):
        es=z[0]
        cs=z[1]
        fs=z[2]
        dsqtl=z[3]
        fs_es = z[4]
        
        es = es.astype(np.int)
        cs = cs.astype(np.int)
        dsqtl = dsqtl.astype(np.int)
        fs = fs.astype(np.int)
        fs_es = fs_es.astype(np.int)
        

        if test_type=='foot':
            esd = fs_es*dsqtl
            e1sd = fs_es*(1-dsqtl)
        elif test_type=='effect':
            esd = es
            e1sd = 1-es
        elif test_type=='circuit':
            esd = es*cs
            e1sd = 1-(es*cs)
        #A
        count_mat_A = add_mat_th(dotprod_2th(esd,fs_es),count_mat_A)

        #B
        count_mat_B = add_mat_th(dotprod_2th(e1sd,fs_es),count_mat_B)

        #C
        count_mat_C = add_mat_th(dotprod_2th(esd,1-fs_es),count_mat_C)

        #D
        count_mat_D = add_mat_th(dotprod_2th(e1sd,1-fs_es),count_mat_D)

        if idx % 20 == 0:
            print("Mult {0} of 20".format(idx))

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)


def cooccurrence_enrichment_gpu_1_compendium(data_mat,circuitSNP_thresh=3.0):

    ES = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    # FS = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    FS_ES = data_mat.loc[:,'M00001':'PBM0207'].values >= 1
    
    CS = np.abs(data_mat['circuitSNPs-5-3-5'].values) >= circuitSNP_thresh
    CS = CS[:,np.newaxis] 
    
    # DSQTL = data_mat['label_x'].values == 1
    # DSQTL = DSQTL[:,np.newaxis]
    
    m,n = ES.shape
    
    ES = np.array_split(ES,5000)
    CS = np.array_split(CS,5000)
    # FS = np.array_split(FS,2)
    # DSQTL = np.array_split(DSQTL,2)
    FS_ES = np.array_split(FS_ES,5000)

    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(ES,CS,FS_ES)):
    # for idx,z in enumerate(FS_ES):
        es=z[0]
        cs=z[1]
        # fs=z[2]
        # dsqtl=z[3]
        fs_es = z[2]
        # fs_es = z
        
        es = es.astype(np.int)
        cs = cs.astype(np.int)
        # dsqtl = dsqtl.astype(np.int)
        # fs = fs.astype(np.int)
        fs_es = fs_es.astype(np.int)
        

        # if test_type=='foot':
            # esd = fs_es*dsqtl
            # e1sd = fs_es*(1-dsqtl)
        esd = cs*es
        e1sd = (1-cs)*es
        # elif test_type=='effect':
            # esd = es*dsqtl
            # e1sd = es*(1-dsqtl)
        # elif test_type=='circuit':
            # esd = es*cs*dsqtl
            # e1sd = es*cs*(1-dsqtl)
        #A
        try:
            
        
            count_mat_A = add_mat_th(dotprod_2th(esd,fs_es),count_mat_A)

            #B
            count_mat_B = add_mat_th(dotprod_2th(e1sd,fs_es),count_mat_B)

            #C
            count_mat_C = add_mat_th(dotprod_2th(esd,1-fs_es),count_mat_C)

            #D
            count_mat_D = add_mat_th(dotprod_2th(e1sd,1-fs_es),count_mat_D)
        except:
            return(esd, e1sd,esd,e1sd)
        if idx % 250== 0:
            print("Mult {0} of 5000".format(idx))
        # print(idx)

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)

def cooccurrence_enrichment_gpu_2_dsqtl(data_mat,circuitSNP_thresh = 3.0, dsqtl_test=False):

    ES = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    FS = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    FS_ES = data_mat.loc[:,'M00001':'PBM0207'].values >= 1
    
    CS = np.abs(data_mat['5'].values) >= circuitSNP_thresh
    CS = CS[:,np.newaxis] 
    
    DSQTL = data_mat['label_x'].values == 1
    DSQTL = DSQTL[:,np.newaxis]
    
    
        
    m,n = ES.shape
    
    ES = np.array_split(ES,2)
    CS = np.array_split(CS,2)
    FS = np.array_split(FS,2)
    DSQTL = np.array_split(DSQTL,2)
    FS_ES = np.array_split(FS_ES,2)

    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(ES,CS,FS,DSQTL,FS_ES)):
        es=z[0]
        cs=z[1]
        fs=z[2]
        dsqtl=z[3]
        fs_es = z[4]
        
        es = es.astype(np.int)
        cs = cs.astype(np.int)
        dsqtl = dsqtl.astype(np.int)
        fs = fs.astype(np.int)
        fs_es = fs_es.astype(np.int)
        
        #A
        if dsqtl_test:
            esd = es*cs*dsqtl
        else:
            esd = es*cs
        count_mat_A = add_mat_th(dotprod_2th(esd,es),count_mat_A)

        #B
        if dsqtl_test:
            e1sd = es*(1-cs)*dsqtl
        else:
            e1sd = es*(1-cs)
        count_mat_B = add_mat_th(dotprod_2th(e1sd,es),count_mat_B)

        #C
        count_mat_C = add_mat_th(dotprod_2th(esd,1-fs),count_mat_C)

        #D
        count_mat_D = add_mat_th(dotprod_2th(e1sd,1-fs),count_mat_D)

        if idx % 20 == 0:
            print("Mult {0} of 20".format(idx))

    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)


def cooccurrence_enrichment_gpu_2_rev2(log_odds = 3.0,data_mat=None):
    import theano
    from theano import tensor as T

    if data_mat is None:
        print("Loading circuitSNPs Data")
        data_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs_with_footsnps.h5')
        data_mat = data_mat[(data_mat.snp==1.0) & (data_mat['es-count']>0)]
    else:
        print("circuitSNPs Data provided")

    F = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    S = data_mat['circuitSNPs-D'].abs() >= log_odds 
    X = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    m,n = X.shape

    del data_mat
    X = np.array_split(X,5000)
    S = np.array_split(S,5000)
    F = np.array_split(F,5000)
    
    gpuX = T.matrix()
    gpuS = T.matrix()

    gpu_dotprod = T.dot(gpuX.T, gpuX)
    gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
    add_two_mat = T.add(gpuX,gpuS)

    dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
    dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
    add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

    count_mat_A = np.zeros((n,n))
    count_mat_B = np.zeros((n,n))
    count_mat_C = np.zeros((n,n))
    count_mat_D = np.zeros((n,n))

    for idx,z in enumerate(zip(X,S,F)):
        e=z[0]
        s=z[1]
        f=z[2]

        e = e.astype(np.int)
        s = s[:,None].astype(np.int)
        f = f.astype(np.int)
        
        #A
        es = e*s
        count_mat_A = add_mat_th(dotprod_2th(es,e),count_mat_A)

        #B
        e1s = e*(1-s)
        count_mat_B = add_mat_th(dotprod_2th(e1s,e),count_mat_B)

        #C
        fe = f*(1-e)
        count_mat_C = add_mat_th(dotprod_2th(es,fe),count_mat_C)

        #D
        count_mat_D = add_mat_th(dotprod_2th(e1s,fe),count_mat_D)

        if idx % 500 == 0:
            dprint("Mult {0} of 5000".format(idx))

    
    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)