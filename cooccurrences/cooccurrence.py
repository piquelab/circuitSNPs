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

# def find_ES_cooccurrence_matrix_gpu(all_updown_opposite='all',log_odds = 3.0):
#     import theano
#     from theano import tensor as T

#     cs = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
#     motifs = cs.loc[0,'M00001':'PBM0207'].index.tolist()

#     cs = cs[cs['circuitSNPs-D'].abs() > log_odds ]
#     M = cs.loc[:,'M00001':'PBM0207'].values
#     m,n = M.shape

#     del cs
#     sub_array = np.array_split(M,5000)

#     del M
#     gpuM = T.matrix()
#     gpuC = T.matrix()

#     gpu_dotprod = T.dot(gpuM.T, gpuM)
#     gpu_dotprod_opp = T.dot(gpuM.T, gpuC)

#     add_two_mat = T.add(gpuM,gpuC)

#     f = theano.function([gpuM], gpu_dotprod,allow_input_downcast=True)
#     o = theano.function([gpuM,gpuC], gpu_dotprod_opp,allow_input_downcast=True)

#     a = theano.function([gpuM,gpuC],add_two_mat,allow_input_downcast=True)



#     if all_updown_opposite=='all':
#         count_mat = np.zeros((n,n))
#         for idx,x in enumerate(sub_array):
#             x[x==2] = 1
#             count_mat = a(f(x),count_mat)
#             print("Mult {0} of 5000".format(idx))
#         np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_both_bind_{0}".format(log_odds), count_mat)
#         return(count_mat)
#     elif all_updown_opposite == 'updown':
#         count_mat_up = np.zeros((n,n))
#         count_mat_down = np.zeros((n,n))
#         for idx,x_up in enumerate(sub_array):
#             x_down = x_up.copy()

#             x_up[x_up==2] = 0
#             x_down[x_down==1] = 0
#             x_down[x_down==2] = 1

#             count_mat_up = a(f(x_up),count_mat_up)
#             count_mat_down = a(f(x_down),count_mat_down)
#             print("Mult {0} of 5000".format(idx))

#         np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_up_bind_{0}".format(log_odds), count_mat_up)
#         np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_down_bind_{0}".format(log_odds), count_mat_down)
#         return(count_mat_up,count_mat_down)

#     elif all_updown_opposite == 'opposite':
#         count_opp = np.zeros((n,n))

#         for idx,x_up in enumerate(sub_array):
#             x_down = x_up.copy()

#             x_up[x_up==2] = 0
#             x_down[x_down==1] = 0
#             x_down[x_down==2] = 1

#             count_a = o(x_up,x_down)
#             count_b = o(x_down,x_up)

#             count_opp = a(a(count_a,count_b),count_opp)
#             # count_mat_down = a(f(x_down),count_mat_down)
#             print("Mult {0} of 5000".format(idx))

#         np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/full_pairs_gpu_opposite_bind_{0}".format(log_odds), count_opp)
#         return(count_opp)


# def cooccurrence_enrichment_gpu_1(log_odds = 3.0,cs_pre = None,foot=None):
#     import theano
#     from theano import tensor as T

#     if cs_pre is None:
#         print("Loading circuitSNPs Data")
#         cs_pre = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
#         cs_pre = cs_pre[(cs_pre.snp==1.0) & (cs_pre['es-count']>0)]
#     else:
#         print("circuitSNPs Data provided")

#     if foot is None:
#         print("Loading footprint data")
#         foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')
#         foot = foot.iloc[cs_pre.index]
#     else:
#         print("footprint data provided")


#     F = foot.iloc[:,3:].values
#     F[F>1] = 1

#     S = cs_pre['circuitSNPs-D'].abs() >= log_odds 
#     S = S.astype(np.int).values

#     X = cs_pre.loc[:,'M00001':'PBM0207'].values
#     X[X>1] = 1
#     m,n = X.shape

#     del cs_pre
#     del foot
#     X = np.array_split(X,5000)
#     S = np.array_split(S,5000)
#     F = np.array_split(F,5000)
    
#     gpuX = T.matrix()
#     gpuS = T.matrix()

#     gpu_dotprod = T.dot(gpuX.T, gpuX)
#     gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
#     add_two_mat = T.add(gpuX,gpuS)

#     dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
#     dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
#     add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

#     count_mat_A = np.zeros((n,n))
#     count_mat_B = np.zeros((n,n))
#     count_mat_C = np.zeros((n,n))
#     count_mat_D = np.zeros((n,n))

#     for idx,z in enumerate(zip(X,S,F)):
#         e=z[0]
#         s=z[1]
#         f=z[2]

#         # e = x==2
#         # e = e.astype(np.int)
#         s = s[:,None]
#         # f = f-e
#         # f = f.astype(np.int)
        
#         #A
#         es = e*s
#         count_mat_A = add_mat_th(dotprod_2th(es,f),count_mat_A)

#         #B
#         e1s = e*(1-s)
#         count_mat_B = add_mat_th(dotprod_2th(e1s,f),count_mat_B)

#         #C
#         # es = e*s
#         count_mat_C = add_mat_th(dotprod_2th(es,(1-f)),count_mat_C)

#         #D
#         # d = e*(1-s)
#         count_mat_D = add_mat_th(dotprod_2th(e1s,(1-f)),count_mat_D)

#         if idx % 500 == 0:
#             print("Mult {0} of 5000".format(idx))

#     # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
#     # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
#     return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)

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
    
    # F[F>1] = 1

    S = data_mat['circuitSNPs-D'].abs() >= log_odds 
    # S = S.astype(np.int).values

    X = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    # X[X>1] = 1
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

        # e = x==2
        e = e.astype(np.int)
        s = s[:,None].astype(np.int)
        # f = f-e
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

    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)    

def cooccurrence_enrichment_gpu_1_dsqtl(log_odds = 3.0,data_mat=None):


    if data_mat is None:
        print("Loading circuitSNPs Data")
        data_mat = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/compendium_circuitSNPs_with_footsnps.h5')
        data_mat = data_mat[(data_mat.snp==1.0) & (data_mat['es-count']>0)]
    else:
        print("circuitSNPs Data provided")

    F = data_mat.loc[:,'M00001':'PBM0207'].values == 1
    
    # F[F>1] = 1

    S = data_mat['circuitSNPs-D'].abs() >= log_odds 
    # S = S.astype(np.int).values

    X = data_mat.loc[:,'M00001':'PBM0207'].values > 1
    # X[X>1] = 1
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

        # e = x==2
        e = e.astype(np.int)
        s = s[:,None].astype(np.int)
        # f = f-e
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

    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)

# def cooccurrence_enrichment_gpu_2(log_odds = 3.0,cs_pre = None,foot=None):
#     import theano
#     from theano import tensor as T

#     if cs_pre is None:
#         print("Loading circuitSNPs Data")
#         cs_pre = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNP_predictions_50_50_full.h5')
#         cs_pre = cs_pre[(cs_pre.snp==1.0) & (cs_pre['es-count']>0)]
#     else:
#         print("circuitSNPs Data provided")

#     if foot is None:
#         print("Loading footprint data")
#         foot = pd.read_hdf('/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/centiSNPs_footprints_pad.h5')
#         foot = foot.iloc[cs_pre.index]
#     else:
#         print("footprint data provided")


#     F = foot.iloc[:,3:].values
#     F[F>1] = 1

#     S = cs_pre['circuitSNPs-D'].abs() >= log_odds 
#     S = S.astype(np.int).values

#     X = cs_pre.loc[:,'M00001':'PBM0207'].values
#     X[X>1] = 1
#     m,n = X.shape

#     del cs_pre
#     del foot
#     X = np.array_split(X,5000)
#     S = np.array_split(S,5000)
#     F = np.array_split(F,5000)
    
#     gpuX = T.matrix()
#     gpuS = T.matrix()

#     gpu_dotprod = T.dot(gpuX.T, gpuX)
#     gpu_dotprod_2 = T.dot(gpuX.T, gpuS)
#     add_two_mat = T.add(gpuX,gpuS)

#     dotprod_1th = theano.function([gpuX], gpu_dotprod,allow_input_downcast=True)
#     dotprod_2th = theano.function([gpuX,gpuS], gpu_dotprod_2,allow_input_downcast=True)
#     add_mat_th = theano.function([gpuX,gpuS],add_two_mat,allow_input_downcast=True)

#     count_mat_A = np.zeros((n,n))
#     count_mat_B = np.zeros((n,n))
#     count_mat_C = np.zeros((n,n))
#     count_mat_D = np.zeros((n,n))

#     for idx,z in enumerate(zip(X,S,F)):
#         e=z[0]
#         s=z[1]
#         f=z[2]

#         # e = x==2
#         s = s[:,None]
#         # f = f-e
        
#         #A
#         es = e*s
#         count_mat_A = add_mat_th(dotprod_2th(es,e),count_mat_A)

#         #B
#         e1s = e*(1-s)
#         count_mat_B = add_mat_th(dotprod_2th(e1s,e),count_mat_B)

#         #C
#         fe = f*(1-e)
#         count_mat_C = add_mat_th(dotprod_2th(es,fe),count_mat_C)

#         #D
#         count_mat_D = add_mat_th(dotprod_2th(e1s,fe),count_mat_D)

#         if idx % 500 == 0:
#             print("Mult {0} of 5000".format(idx))

#     # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
#     # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
#     return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)

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

    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_es_bind_{0}".format(log_odds), count_mat_ES)
    # np.save("/wsu/home/al/al37/al3786/CENNTIPEDE/circuitSNPs/compendium_predictions/factor_pairs/test1_footsnp_{0}".format(log_odds), count_mat_NonES)
    return(count_mat_A,count_mat_B,count_mat_C,count_mat_D)