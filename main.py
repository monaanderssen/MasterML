#!/usr/bin/env python3

from __future__ import division

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from numba import jit, cuda

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import argparse
import pickle
import itertools

from ROOT import TLorentzVector, RooStats
import uproot
import numpy as np
import pandas as pd
#from pandas import HDFStore
import seaborn as sns

import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from sklearn import preprocessing, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, classification_report, make_scorer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.utils import shuffle, resample 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import pylatex
from matplotlib.colors import *

import logging

logging.basicConfig(format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)


from importdata import *
from plotting import *
from samples import *
from datasetnumber_bkg import *
#from samples import l_bkg, l_sig_low, l_sig_inter, l_sig_high, d_sig, l_data
#from samples import slepslep_low, slepslep_inter, slepslep_high, slepsnu_low, slepsnu_inter, slepsnu_high, WW_low, WW_inter, WW_high, MonoZ_low, MonoZ_inter, MonoZ_high
#from hepFunctions import invariantMass

#@cuda.jit
def main():

    # Start timer
    

    parser = argparse.ArgumentParser()
    group_model = parser.add_mutually_exclusive_group() 
    group_model.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
    group_model.add_argument('-n', '--nn', action='store_true', help='Run neural network')
    group_read_dataset = parser.add_mutually_exclusive_group() 
    group_read_dataset.add_argument('-p', '--prepare_hdf5', action='store_true', help='Prepare input datasets for ML and store in HDF5 file')
    group_read_dataset.add_argument('-r', '--read_hdf5', action='store_true', help='Read prepared datasets from HDF5 file')
    group_read_dataset.add_argument('-d', '--direct_read', action='store_true', help='Read unprepared datasets from ROOT file')
    parser.add_argument('-B', '--N_sig_events', type=lambda x: int(float(x)), default=1e3, help='Number of signal events to read from the dataset')
    parser.add_argument('-S', '--N_bkg_events', type=lambda x: int(float(x)), default=1e3, help='Number of background events to read from the dataset for each class')
    parser.add_argument('-e', '--event_weight', action='store_true', help='Apply event weights during training')
    parser.add_argument('-c', '--class_weight', action='store_true', help='Apply class weights to account for unbalanced dataset')

    parser.add_argument('--sig', action = 'store_true', help= 'Run for signal samples')
    parser.add_argument('--bkg', action = 'store_true', help= 'Run for background samples')
    parser.add_argument('--data', action = 'store_true', help= 'Run for data samples')  

    parser.add_argument('--low', action = 'store_true', help= 'Run for low mass splitting signal samples')
    parser.add_argument('--inter', action = 'store_true', help= 'Run for intermediate mass splitting signal samples')
    parser.add_argument('--high', action = 'store_true', help= 'Run for high mass splitting signal samples')
    
    parser.add_argument('--load_pretrained_model', action = 'store_true', help= 'Run with a pretrained model')

    parser.add_argument('--slepslep', action = 'store_true', help= 'Run for direct slepton production signal samples')
    parser.add_argument('--slepsnu', action = 'store_true', help= 'Run for chargino pair and slepton/sneutrino signal samples')
    parser.add_argument('--WW', action = 'store_true', help= 'Run for W-bosons and slepton/sneutrino signal samples')
    parser.add_argument('--monoZ', action = 'store_true', help= 'Run for mono-Z signal samples')

    parser.add_argument('--high_level', action = 'store_true', help= 'Run with high level features')
    parser.add_argument('--low_level', action = 'store_true', help= 'Run with low level features')

    args = parser.parse_args()

    global df_feat #, df_bkg_feat, df_data_feat

    #sample_name = d_sig['high']

    if args.sig:
        sample_type = 'sig'
        if args.slepslep:
            if args.low:
                sample_name = d_sig_slepslep['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepslep_low_preprocessed.h5'
            elif args.inter:
                sample_name = d_sig_slepslep['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepslep_inter_preprocessed.h5'
            elif args.high:
                sample_name = d_sig_slepslep['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepslep_high_preprocessed.h5'
        elif args.slepsnu:
            if args.low:
                sample_name = d_sig_slepsnu['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepsnu_low_preprocessed.h5'
            elif args.inter:
                sample_name = d_sig_slepsnu['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepsnu_inter_preprocessed.h5'
            elif args.high:
                sample_name = d_sig_slepsnu['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_slepsnu_high_preprocessed.h5'
        elif args.WW:
            if args.low:
                sample_name = d_sig_WW['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_WW_low_preprocessed.h5'
            elif args.inter:
                sample_name = d_sig_WW['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_WW_inter_preprocessed.h5'
            elif args.high:
                sample_name = d_sig_WW['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_WW_high_preprocessed.h5'
        elif args.monoZ:
            if args.low:
                sample_name = d_sig_monoZ['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_monoZ_low_preprocessed.h5'
            elif args.inter:
                sample_name = d_sig_monoZ['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_monoZ_inter_preprocessed.h5'
            elif args.high:
                sample_name = d_sig_monoZ['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_monoZ_high_preprocessed.h5'
    elif args.bkg:
        sample_type = 'bkg'
        sample_name = l_bkg
        filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) + '_NY_preprocessed.h5'
    elif args.data:
        sample_type = 'data'
        sample_name = l_data
        filename_preprocessed = '/storage/monande/hdf5Files/' + str(sample_type) +'_preprocessed.h5'

    if args.prepare_hdf5:
        """Read input dataset in chunks, select features and perform cuts,
        before storing DataFrame in HDF5 file"""

        # Import OpenData ntuples as flat pandas DataFrames
        exists = os.path.isfile(filename_preprocessed)
        if exists:
            os.remove(filename_preprocessed)
            print("Removed existing file", filename_preprocessed)
        store = pd.HDFStore(filename_preprocessed)
        print("Created new store with name", filename_preprocessed)

        print("\nIn main(): n_events_chunk =", n_events_chunk)
    
    
        #return
        path = "/storage/eirikgr/NEWnTuples_EIRIK/"
        #sig_path = path + "SUSY_Signal18/SlepSlep_direct_100p0_1p0_merged_processed.root"
        #bkg_path = path + "SUSY_Bkg18/diboson_merged_processed.root"
        #data_path = path + "SUSY_Data18/data18_merged_processed.root"
        """
        path = "/eos/user/m/manderss/nTuples/"
        sig_path = path + "SUSY_Signal18/SlepSlep_direct_800p0_1p0_merged_processed.root"
        #bkg_path = path + "SUSY_Bkg18/diboson_merged_processed.root"
        #data_path = path + "SUSY_Data18/data18_merged_processed.root"
        """

        if sample_type is 'sig':
            if args.slepslep:
                for elm in sample_name:
                    filepath = path + str(elm)
                    line = filepath.split("/")
                    print(line)
                    line = line[8].split("_merged")
                
                    line = line[0]
                    print(line)
                    treename = line + "_NoSys"
                    store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)
            elif args.slepsnu:
                for elm in sample_name:
                    filepath = path + str(elm)
                    line = filepath.split("/")
                    print(line)
                    line = line[8].split("_merged")
                
                    line = line[0]
                    print(line)
                    treename = line + "_NoSys"
                    store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)
            elif args.WW:
                for elm in sample_name:
                    filepath = path + str(elm)
                    line = filepath.split("/")
                    print(line)
                    line = line[8].split("_merged")
                
                    line = line[0]
                    print(line)
                    treename = line + "_NoSys"
                    store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)
            elif args.monoZ:
                for elm in sample_name:
                    filepath = path + str(elm)
                    line = filepath.split("/")
                    print(line)
                    line = line[8].split("_merged")
                
                    line = line[0]
                    print(line)
                    treename = line + "_NoSys"
                    store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)
        elif sample_type is 'bkg':
            for elm in sample_name:
                filepath = path + 'SUSY_Bkg/' + str(elm)
                line = filepath.split("/")
                line = line[7].split("_merged")
                line = line[0]
                treename = line + "_NoSys"
                store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)
                #filepath = path + 'SUSY_Bkg18/' + str(sample_name)
        elif sample_type is 'data':
           # print('HALLO', sample_name)
            for elm in sample_name:
                filepath = path + 'SUSY_Data/'+ str(elm)
                line = filepath.split("/")
                print(line)
                line = line[7].split("_merged")
                line = line[0]
                treename = line
                store = prepareInput(store, sample_type, sample_name,treename, filepath, filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)

        # Prepare and store signal dataset
        
        # Prepare and store background dataset
        #store = prepareInput(store, sample_type='bkg', filename=filename_preprocessed, chunk_size=1e5, n_chunks=100000, entrystart=0)

        #store = prepareInput(store, sample_type='data', filename=filename_preprocessed, chunk_size=1e5, n_chunks=100, entrystart=0)
        
        print("\nReturned from prepareInput()")
        print("\nstore:\n", store)
        print("\nstore.keys()", store.keys())
        

        df_feat = store[sample_type]
        #df_sig_feat = store['sig']
        #df_bkg_feat = store['bkg']
        #df_data_feat = store['data']
        print("\ndf_feat.head():\n", df_feat.head())
        #print("\ndf_bkg_feat.head():\n", df_bkg_feat.head())

        store.close()
        print("Closed store")
        
        return

    elif args.read_hdf5:

        if args.xgboost:
            if args.slepslep:
                if args.low:
                    if args.high_level:
                        main_path = 'BDTSlepSlepHLlow'
                    elif args.low_level:
                        main_path = 'BDTSlepSlepLLlow'
                    else:
                        main_path = 'BDTSlepSlepALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'BDTSlepSlepHLinter'
                    elif args.low_level:
                        main_path = 'BDTSlepSlepLLinter'
                    else:
                        main_path = 'BDTSlepSlepALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'BDTSlepSlepHLhigh'
                    elif args.low_level:
                        main_path = 'BDTSlepSlepLLhigh'
                    else:
                        main_path = 'BDTSlepSlepALhigh'

            elif args.slepsnu:
                if args.low:
                    if args.high_level:
                        main_path = 'BDTSlepSnuHLlow'
                    elif args.low_level:
                        main_path = 'BDTSlepSnuLLlow'
                    else:
                        main_path = 'BDTSlepSnuALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'BDTSlepSnuHLinter'
                    elif args.low_level:
                        main_path = 'BDTSlepSnuLLinter'
                    else:
                        main_path = 'BDTSlepSnuALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'BDTSlepSnuHLhigh'
                    elif args.low_level:
                        main_path = 'BDTSlepSnuLLhigh'
                    else:
                        main_path = 'BDTSlepSnuALhigh'

            elif args.WW:
                if args.low:
                    if args.high_level:
                        main_path = 'BDTWWHLlow'
                    elif args.low_level:
                        main_path = 'BDTWWLLlow'
                    else:
                        main_path = 'BDTWWALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'BDTWWHLinter'
                    elif args.low_level:
                        main_path = 'BDTWWLLinter'
                    else:
                        main_path = 'BDTWWALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'BDTWWHLhigh'
                    elif args.low_level:
                        main_path = 'BDTWWLLhigh'
                    else:
                        main_path = 'BDTWWALhigh'

            elif args.monoZ:
                if args.low:
                    if args.high_level:
                        main_path = 'BDTmonoZHLlow'
                    elif args.low_level:
                        main_path = 'BDTmonoZLLlow'
                    else:
                        main_path = 'BDTmonoZALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'BDTmonoZHLinter'
                    elif args.low_level:
                        main_path = 'BDTmonoZLLinter'
                    else:
                        main_path = 'BDTmonoZALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'BDTmonoZHLhigh'
                    elif args.low_level:
                        main_path = 'BDTmonoZLLhigh'
                    else:
                        main_path = 'BDTmonoZALhigh'


        elif args.nn:
            if args.slepslep:
                if args.low:
                    if args.high_level:
                        main_path = 'NNSlepSlepHLlow'
                    elif args.low_level:
                        main_path = 'NNSlepSlepLLlow'
                    else:
                        main_path = 'NNSlepSlepALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'NNSlepSlepHLinter'
                    elif args.low_level:
                        main_path = 'NNSlepSlepLLinter'
                    else:
                        main_path = 'NNSlepSlepALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'NNSlepSlepHLhigh'
                    elif args.low_level:
                        main_path = 'NNSlepSlepLLhigh'
                    else:
                        main_path = 'NNSlepSlepALhigh'

            elif args.slepsnu:
                if args.low:
                    if args.high_level:
                        main_path = 'NNSlepSnuHLlow'
                    elif args.low_level:
                        main_path = 'NNSlepSnuLLlow'
                    else:
                        main_path = 'NNSlepSnuALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'NNSlepSnuHLinter'
                    elif args.low_level:
                        main_path = 'NNSlepSnuLLinter'
                    else:
                        main_path = 'NNSlepSnuALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'NNSlepSnuHLhigh'
                    elif args.low_level:
                        main_path = 'NNSlepSnuLLhigh'
                    else:
                        main_path = 'NNSlepSnuALhigh'

            elif args.WW:
                if args.low:
                    if args.high_level:
                        main_path = 'NNWWHLlow'
                    elif args.low_level:
                        main_path = 'NNWWLLlow'
                    else:
                        main_path = 'NNWWALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'NNWWHLinter'
                    elif args.low_level:
                        main_path = 'NNWWLLinter'
                    else:
                        main_path = 'NNWWALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'NNWWHLhigh'
                    elif args.low_level:
                        main_path = 'NNWWLLhigh'
                    else:
                        main_path = 'NNWWALhigh'

            elif args.monoZ:
                if args.low:
                    if args.high_level:
                        main_path = 'NNmonoZHLlow'
                    elif args.low_level:
                        main_path = 'NNmonoZLLlow'
                    else:
                        main_path = 'NNmonoZALlow'
                elif args.inter:
                    if args.high_level:
                        main_path = 'NNmonoZHLinter'
                    elif args.low_level:
                        main_path = 'NNmonoZLLinter'
                    else:
                        main_path = 'NNmonoZALinter'
                elif args.high:
                    if args.high_level:
                        main_path = 'NNmonoZHLhigh'
                    elif args.low_level:
                        main_path = 'NNmonoZLLhigh'
                    else:
                        main_path = 'NNmonoZALhigh'




        if args.slepslep:
            process_name = 'slepslep'
            """
            background_file_Wjets = '/storage/monande/hdf5Files/W_jets_NY_monoZ_preprocessed.h5'
            background_file_Zjets = '/storage/monande/hdf5Files/Zjets_NY_monoZ_preprocessed.h5'
            background_file_Higgs = '/storage/monande/hdf5Files/higgs_NY_monoZ_preprocessed.h5'
            background_file_DY = '/storage/monande/hdf5Files/lowMassDY_NY_monoZ_preprocessed.h5'
            background_file_singleTop = '/storage/monande/hdf5Files/singleTop_NY_monoZ_preprocessed.h5'
            background_file_ttbar = '/storage/monande/hdf5Files/ttbar_NY_monoZ_preprocessed.h5'
            background_file_topOther = '/storage/monande/hdf5Files/topOther_NY_monoZ_preprocessed.h5'
            background_file_Diboson = '/storage/monande/hdf5Files/diboson_NY_monoZ_preprocessed.h5'
            background_file_Triboson = '/storage/monande/hdf5Files/triboson_NY_monoZ_preprocessed.h5'
            """
            if args.low:
                sig_type = 'low'
                sample_name = d_sig_slepslep['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepslep_low_preprocessed.h5'
            elif args.inter:
                sig_type = 'inter'
                sample_name = d_sig_slepslep['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepslep_inter_preprocessed.h5'
            elif args.high:
                sig_type = 'high'
                sample_name = d_sig_slepslep['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepslep_high_preprocessed.h5'
        elif args.slepsnu:
            process_name = 'slepsnu'
            if args.low:
                sig_type = 'low'
                sample_name = d_sig_slepsnu['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepsnu_low_preprocessed.h5'
            elif args.inter:
                sig_type = 'inter'
                sample_name = d_sig_slepsnu['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepsnu_inter_preprocessed.h5'
            elif args.high:
                sig_type = 'high'
                sample_name = d_sig_slepsnu['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_slepsnu_high_preprocessed.h5'
        elif args.WW:
            process_name = 'WW'
            if args.low:
                sig_type = 'low'
                sample_name = d_sig_WW['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_WW_low_preprocessed.h5'
            elif args.inter:
                sig_type = 'inter'
                sample_name = d_sig_WW['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_WW_inter_preprocessed.h5'
            elif args.high:
                sig_type = 'high'
                sample_name = d_sig_WW['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_WW_high_preprocessed.h5'
        elif args.monoZ:
            process_name = 'monoZ'

            if args.low:
                sig_type = 'low'
                sample_name = d_sig_monoZ['low']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_monoZ_low_preprocessed.h5'
            elif args.inter:
                sig_type = 'inter'
                sample_name = d_sig_monoZ['inter']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_monoZ_inter_preprocessed.h5'
            elif args.high:
                sig_type = 'high'
                sample_name = d_sig_monoZ['high']
                filename_preprocessed = '/storage/monande/hdf5Files/' + main_path + '/sig_monoZ_high_preprocessed.h5'

        background_file_Wjets = '/storage/monande/hdf5Files/' + main_path + '/Wjets_NY_preprocessed.h5'
        background_file_Zjets = '/storage/monande/hdf5Files/' + main_path + '/Zjets_NY_preprocessed.h5'
        background_file_Higgs = '/storage/monande/hdf5Files/' + main_path + '/higgs_NY_preprocessed.h5'
        background_file_DY = '/storage/monande/hdf5Files/' + main_path + '/lowMassDY_NY_preprocessed.h5'
        background_file_singleTop = '/storage/monande/hdf5Files/' + main_path + '/singleTop_NY_preprocessed.h5'
        background_file_ttbar = '/storage/monande/hdf5Files/' + main_path + '/ttbar_NY_preprocessed.h5'
        background_file_topOther = '/storage/monande/hdf5Files/' + main_path + '/topOther_NY_preprocessed.h5'
        background_file_Diboson = '/storage/monande/hdf5Files/' + main_path + '/diboson_NY_preprocessed.h5'
        background_file_Triboson = '/storage/monande/hdf5Files/' + main_path + '/triboson_NY_preprocessed.h5'

        store = pd.HDFStore(filename_preprocessed)
        df_sig_feat = store['sig']
        df_sig_feat['Type'] = 'Signal'

        store = pd.HDFStore(background_file_Wjets)
        df_bkg_feat_Wjets = store['bkg']
        df_bkg_feat_Wjets['Type'] = 'Wjets'
        
        store = pd.HDFStore(background_file_Zjets)
        df_bkg_feat_Zjets = store['bkg']
        df_bkg_feat_Zjets['Type'] = 'Zjets'

        store = pd.HDFStore(background_file_Higgs)
        df_bkg_feat_Higgs = store['bkg']
        df_bkg_feat_Higgs['Type'] = 'Higgs'

        store = pd.HDFStore(background_file_DY)
        df_bkg_feat_DY = store['bkg']
        df_bkg_feat_DY['Type'] = 'Higgs'

        store = pd.HDFStore(background_file_singleTop)
        df_bkg_feat_singleTop = store['bkg']
        df_bkg_feat_singleTop['Type'] = 'singleTop'

        store = pd.HDFStore(background_file_ttbar)
        df_bkg_feat_ttbar = store['bkg']
        df_bkg_feat_ttbar['Type'] = 'ttbar'

        store = pd.HDFStore(background_file_topOther)
        df_bkg_feat_topOther = store['bkg']
        df_bkg_feat_topOther['Type'] = 'topOther'

        store = pd.HDFStore(background_file_Diboson)
        df_bkg_feat_Diboson = store['bkg']
        df_bkg_feat_Diboson['Type'] = 'Diboson'

        store = pd.HDFStore(background_file_Triboson)
        df_bkg_feat_Triboson = store['bkg']
        df_bkg_feat_Triboson['Type'] = 'Triboson'

        bkg_df = [df_bkg_feat_Wjets, df_bkg_feat_Zjets, df_bkg_feat_Higgs, df_bkg_feat_DY, df_bkg_feat_singleTop, df_bkg_feat_ttbar, df_bkg_feat_topOther, df_bkg_feat_Diboson, df_bkg_feat_Triboson]

        df_bkg_feat = pd.concat(bkg_df)








        print("\ndf_sig_feat.head():\n", df_sig_feat.head())
        print("\ndf_bkg_feat.head():\n", df_bkg_feat.head())
        
        df_sig_feat = df_sig_feat.loc[:,~df_sig_feat.columns.duplicated()]
        df_bkg_feat = df_bkg_feat.loc[:,~df_bkg_feat.columns.duplicated()]
        
        store.close()
        print("Closed store")
        #df_bkg_feat = resample(df_bkg_feat, replace=False, n_samples=len(df_sig_feat), random_state=42, stratify=None)
        print("\n======================================")
        print("df_sig_feat.shape        =", df_sig_feat.shape)
        print("df_bkg_feat.shape        =", df_bkg_feat.shape)
        print("======================================")
        print(df_sig_feat.columns)
        print(df_bkg_feat.columns)
        # make array of features
        df_X = pd.concat([df_bkg_feat, df_sig_feat], axis=0)#, sort=False)

        df_X = df_X.loc[:,~df_X.columns.duplicated()]
        #df_X.drop('channel', axis=1, inplace=True)

        df_X['eventweight']*= 44300
        df_X.loc[df_X['RandomRunNumber'] < 320000, 'eventweight'] *= 36200/44300
        df_X.loc[df_X['RandomRunNumber'] > 348000, 'eventweight'] *= 58500/44300

        # make array of labels
        y_bkg = np.zeros(len(df_bkg_feat))
        y_sig = np.ones(len(df_sig_feat))
        y = np.concatenate((y_bkg, y_sig), axis=0).astype(int)
        df_X["ylabel"] = y
        
        print("\n- Input arrays:")
        print("df_X.drop(['eventweight', 'ylabel']) :", df_X.drop(["eventweight", "ylabel"], axis=1).shape)
        print("df_X.eventweight                     :", df_X.eventweight.shape)
        print("df_X.ylabel                          :", df_X.ylabel.shape)
            
        sum_sig_weights_before = df_sig_feat['eventweight'].sum(axis=0)
        sum_bkg_weights_before = df_bkg_feat['eventweight'].sum(axis=0)
        sum_weights_before = df_X['eventweight'].sum(axis=0)

        # Split the dataset in train and test sets
        test_size = 0.33
        seed = 42

        X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=test_size, random_state=seed, stratify=y)
        
        sum_bkg_scaled_weights_train_mid = X_train.query('ylabel==0').loc[:,'eventweight'].sum(axis=0)

        X_train_prev= len(df_bkg_feat)

        print(X_train.head())

        N_train_sig = len(X_train.query('ylabel==1'))
        N_train_bkg = N_train_sig
        # Draw balanced training datasets where the number of signal and background events are equal
        X_train_sig = resample(X_train.query('ylabel==1'), replace=False, n_samples=N_train_sig, random_state=42)#, stratify=None)
        X_train_bkg = resample(X_train.query('ylabel==0'), replace=False, n_samples=N_train_bkg, random_state=42)#, stratify=None)
        X_train = pd.concat([X_train_bkg, X_train_sig], axis=0)

        X_train_current = len(X_train.query('ylabel==0'))

        scale_fac = X_train_prev/X_train_current 
        print(scale_fac)

        y_train = X_train.ylabel
        y_test = X_test.ylabel
        
        high_level_features = ['mll', 'mt2', 'HT', 'met_HT', 'deltaPhi', 'deltaRll', 'pTdiff']
        low_level_features = ['lepPt1', 'lepPt2', 'lepEta1', 'lepEta2', 'lepPhi1', 'lepPhi2', 'nJet20', 'nJet30', 'nBJet20_MV2c10_FixedCutBEff_85', 'met_Et', 'met_Sign',  'isEM', 'isEE', 'isMM', 'isOS', 'isSS']

        if args.low_level:
            level_type = 'Low_level'
            print(X_train.head())
            X_train.drop(high_level_features, axis=1, inplace=True)
            X_test.drop(high_level_features, axis=1, inplace=True)
            print(X_train)
        elif args.high_level:
            level_type = 'High_level'
            print(X_train.head())
            X_train.drop(low_level_features, axis=1, inplace=True)
            X_test.drop(low_level_features, axis=1, inplace=True)
            print(X_train)
        else:
            level_type = 'All_level'



        # Draw validation set as subsample of test set, for quicker evaluation of validation loss during training
        n_val_samples = 1e5
        val_size = 0.1
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
        #X_val = resample(X_test, replace=False, n_samples=n_val_samples, random_state=42, stratify=X_test.ylabel)
        #y_val = X_val.ylabel
        
        #y_train = X_train.ylabel
        #y_test = X_test.ylabel
        sum_sig_weights_after = X_train_sig.query('ylabel==1').loc[:,'eventweight'].sum(axis=0)
        sum_bkg_weights_after = X_train_bkg.query('ylabel==0').loc[:,'eventweight'].sum(axis=0)
        sum_sig_scaled_weights_test = X_test.query('ylabel==1').loc[:,'eventweight'].sum(axis=0)
        sum_bkg_scaled_weights_test = X_test.query('ylabel==0').loc[:,'eventweight'].sum(axis=0)
        
        # Making a copy of the DFs with only feature columns
        X_train_feat_only = X_train.copy()
        X_test_feat_only = X_test.copy()
        X_val_feat_only = X_val.copy()
        l_non_features = ['eventweight', "DatasetNumber", 'ylabel', 'RandomRunNumber', 'EventNumber', 'Type']
        X_train_feat_only.drop(l_non_features, axis=1, inplace=True)
        X_test_feat_only.drop(l_non_features, axis=1, inplace=True)
        X_val_feat_only.drop(l_non_features, axis=1, inplace=True)
                
        print("\nX_train_feat_only:", X_train_feat_only.columns)
        print("X_test_feat_only:", X_test_feat_only.columns)
        print("X_val_feat_only:", X_val_feat_only.columns)
            
        print("\nX_train_feat_only:", X_train_feat_only.shape)
        print("X_test_feat_only:", X_test_feat_only.shape)
        print("X_val_feat_only:", X_val_feat_only.shape)

        # Feature scaling
        # Scale all variables to the interval [0,1]
        #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        #scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        #print("\nscaler.fit_transform(X_train_feat_only)")
        
        """
        sum_sig_weights_after = X_train.query('ylabel==1').loc[:,'eventweight'].sum(axis=0)
        sum_bkg_weights_after = X_train.query('ylabel==0').loc[:,'eventweight'].sum(axis=0)
        sum_weights_after = X_train['eventweight'].sum(axis=0)

        sum_sig_scaled_weights_test = X_test.query('ylabel==1').loc[:,'eventweight'].sum(axis=0)
        sum_bkg_scaled_weights_test = X_test.query('ylabel==0').loc[:,'eventweight'].sum(axis=0)
        """
        # y_train_sig = resample(y_train.query('ylabel==1'), replace=False, n_samples=N_train_sig, random_state=42)#, stratify=None)
        # y_train_bkg = resample(y_train('ylabel==0'), replace=False, n_samples=N_train_bkg, random_state=42)#, stratify=None)
        # y_train = pd.concat([y_train_bkg, y_train_sig], axis=0)
        """
        print("\nX_train:", X_train.columns)
        print("X_test:", X_test.columns)
            
        print("\nX_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)

        print("\nisinstance(X_train, pd.DataFrame):", isinstance(X_train, pd.DataFrame))
        print("isinstance(X_test, pd.DataFrame):", isinstance(X_test, pd.DataFrame))
        print("isinstance(y_train, pd.Series):", isinstance(y_train, pd.Series))
        print("isinstance(y_test, pd.Series):", isinstance(y_test, pd.Series))
        """
        # Making a copy of the train and test DFs with only feature columns
        #X_train_feat_only = X_train.copy()
        #X_test_feat_only = X_test.copy()

        #print(X_test_feat_only.head())
        #X_train_feat_only.drop(["eventweight", "DatasetNumber", "ylabel"], axis=1, inplace=True)
        #X_test_feat_only.drop(["eventweight", "DatasetNumber", "ylabel"], axis=1, inplace=True)
        #print(X_test_feat_only.head())
        # Feature scaling
        # Scale all variables to the interval [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_scaled = min_max_scaler.fit_transform(X_train_feat_only)
        X_test_scaled = min_max_scaler.transform(X_test_feat_only)
        #X_train_scaled = min_max_scaler.transform(X_train_feat_only)
        print("scaler.transform(X_test_feat_only)")
        #X_test_scaled = min_max_scaler.transform(X_test_feat_only)
        print("scaler.transform(X_val_feat_only)")
        X_val_scaled = min_max_scaler.transform(X_val_feat_only)

        print(X_test_scaled)


        if args.data:
            #print("\nRunning XGBoost BDT")
            sample_type = 'data15-16'
            sample_name = l_data
            filename_preprocessed = '/storage/monande/hdf5Files/'+ main_path +'/' + str(sample_type) +'_preprocessed.h5'
            store = pd.HDFStore(filename_preprocessed)
            df_data_feat_15_16 = store['data']

            sample_type = 'data17'
            sample_name = l_data
            filename_preprocessed = '/storage/monande/hdf5Files/'+ main_path +'/' + str(sample_type) +'_preprocessed.h5'
            store = pd.HDFStore(filename_preprocessed)
            df_data_feat_17 = store['data']

            sample_type = 'data18'
            sample_name = l_data
            filename_preprocessed = '/storage/monande/hdf5Files/'+ main_path +'/' + str(sample_type) +'_preprocessed.h5'
            store = pd.HDFStore(filename_preprocessed)
            df_data_feat_18 = store['data']

            data_df = [df_data_feat_15_16, df_data_feat_17, df_data_feat_18]

            df_data_feat = pd.concat(data_df)

            print(df_data_feat.columns)

            df_data_feat = df_data_feat.loc[:,~df_data_feat.columns.duplicated()]
            #df_data_feat.drop('channel', axis=1, inplace=True)

            if args.low_level:
                df_data_feat.drop(high_level_features, axis=1, inplace=True)

            elif args.high_level:
                df_data_feat.drop(low_level_features, axis=1, inplace=True)


            print(df_data_feat.columns)

            print(df_data_feat)
            X_data_feat_only = df_data_feat.copy()
            X_data_feat_only.drop(["DatasetNumber", "eventweight", 'RandomRunNumber', 'EventNumber'], axis=1, inplace=True)
            #min_max_scaler = preprocessing.MinMaxScaler()

            X_data_scaled = min_max_scaler.transform(X_data_feat_only)
            print('JUHU!')
            print(X_data_scaled)
            

            print("\ndf_data_feat.head():\n", df_data_feat.head())

            store.close()
            print("Closed store")

            


            print("\n======================================")
            print("df_data_feat.shape        =", df_data_feat.shape)
            print("======================================")
    
            
        """
        sum_sig_weights = sum_sig_weights_before/sum_sig_weights_after
        sum_bkg_weights = sum_bkg_weights_before/sum_bkg_weights_after
        sum_weights = sum_weights_before/sum_weights_after
        sum_sig_scaled_weights = sum_sig_weights_after/sum_sig_scaled_weights_test
        sum_bkg_scaled_weights = sum_bkg_weights_after/sum_bkg_scaled_weights_test
        sum_bkg_scaled_weights_train = sum_bkg_scaled_weights_train_mid/sum_bkg_weights_after
        print(sum_sig_weights_before,sum_sig_weights_after)
        print(sum_bkg_weights_before,sum_bkg_weights_after)
        print(sum_weights_before,sum_weights_after)
        print(sum_sig_weights_after,sum_sig_scaled_weights_test)
        print(sum_bkg_weights_after,sum_bkg_scaled_weights_test)

        print(sum_sig_weights, sum_bkg_weights, sum_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights)
        """










    print("\n\n//////////////////// ML part ////////////////////////")
    t_start = time.time()
    global model
    scale_pos_weight = 1
    event_weight = None
    class_weight = None

    #if args.event_weight:
    if args.xgboost:
        Event_weight = X_train.eventweight
    """
    if args.class_weight:
        if args.xgboost:
            # XGBoost: Scale signal events up by a factor n_bkg_train_events / n_sig_train_events
            scale_pos_weight = len(X_train[X_train.ylabel == 0]) / len(X_train[X_train.ylabel == 1]) 
        else:
            # sciki-learn: Scale overrespresented sample down (bkg) and underrepresented sample up (sig)
            class_weight = "balanced"
    else:
        class_weight = None
    """
    if args.sig:
        print("\n# bkg train events / # sig train events = {0:d} / {1:d}".format(len(X_train[X_train.ylabel == 0]), len(X_train[X_train.ylabel == 1])))
        print("scale_pos_weight =", scale_pos_weight)

        classes = np.unique(y)
        class_weight_vect = compute_class_weight(class_weight, classes, y)
        class_weight_dict = {0: class_weight_vect[0], 1: class_weight_vect[1]}

    # Run XGBoost BDT
    if args.xgboost:
        method_type = 'BDT'
        if args.slepslep:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_low_slepslep.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_inter_slepslep.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_high_slepslep.joblib"
        elif args.slepsnu:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_low_slepsnu.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_inter_slepsnu.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_high_slepsnu.joblib"
        if args.WW:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_low_WW.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_inter_WW.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_high_WW.joblib"
        if args.monoZ:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_low_monoZ.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_inter_monoZ.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedBDT_" + level_type +"_high_monoZ.joblib"
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        if not args.load_pretrained_model:
            print("\nRunning XGBoost BDT")
            model = XGBClassifier(max_depth=3, 
                                  learning_rate=0.1, 
                                  n_estimators=10000000, 
                                  verbosity=3, 
                                  objective="binary:logistic", 
                                  scale_pos_weight=scale_pos_weight)
            history = model.fit(X_train_scaled, y_train, 
                                sample_weight=event_weight,
                                early_stopping_rounds=20,
                                eval_metric = 'logloss',
                                eval_set=[(X_val_scaled, y_val)],
                                verbose = True)
            joblib.dump(model,trainedModelPath)
            print("\nSaving the trained model")
            print("\nBuilding and training BDT\n")
            print("\nX_train_scaled.shape\n", X_train_scaled.shape)

            #d_val_loss = {'Training loss': history.history['loss']}
            #df_val_loss = pd.DataFrame(d_val_loss)
            """
            plt.figure(20)
            sns.set()
            ax = sns.lineplot(data=df_val_loss)
            ax.set(xlabel='Estimators', ylabel='Loss')
            plt.savefig('Loss_' + sample_type + '_' + process_name + '_' + level_type + '_' + sig_type +'_' + method_type+ '.pdf')
            plt.show()
            plt.clf()
            """
        elif args.load_pretrained_model:
            print("\nReading pretrained model:", trainedModelPath)
            model = joblib.load(trainedModelPath)

    # Run neural network
    elif args.nn:
        method_type = 'NN'
        if args.slepslep:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_low_slepslep.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_inter_slepslep.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_high_slepslep.joblib"
        elif args.slepsnu:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_low_slepsnu.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_inter_slepsnu.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_high_slepsnu.joblib"
        if args.WW:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_low_WW.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_inter_WW.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_high_WW.joblib"
        if args.monoZ:
            if args.low:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_low_monoZ.joblib"
            elif args.inter:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_inter_monoZ.joblib"
            elif args.high:
                trainedModelPath = "/storage/monande/trainedModels/trainedNN_" + level_type +"_high_monoZ.joblib"
        print("\nRunning neural network")
        n_inputs = X_train_scaled.shape[1]
        n_nodes = 300
        n_hidden_layers = 5
        dropout_rate = 0.
        batch_size = 32
        epochs = 100000
        l1 = 0.
        l2 = 1e-3
        lr = 1e-5
        print(X_val_scaled.shape, y_val.shape)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights = True)
        #@cuda.jit
        if not args.load_pretrained_model:
            model = KerasClassifier(build_fn=create_model,
                                    n_inputs=n_inputs,
                                    n_hidden_layers=n_hidden_layers,
                                    n_nodes=n_nodes,
                                    dropout_rate=dropout_rate,
                                    l1=l1,
                                    l2=l2,
                                    lr=lr,
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    verbose=1)

            history = model.fit(X_train_scaled, y_train, 
                                sample_weight=event_weight, 
                                class_weight=class_weight_dict,
                                verbose=1,
                                callbacks=[es],
                                validation_data=(X_val_scaled, y_val)
                                #validation_data=(X_test_scaled, y_test)
                            )

            print("\nmodel.model.summary()\n", model.model.summary())
            d_val_loss = {'Training loss': history.history['loss'], 'Validation loss': history.history['val_loss']}
            df_val_loss = pd.DataFrame(d_val_loss)
            
            d_val_accuracy = {'Training accuracy': history.history['acc'], 'Validation accuracy': history.history['val_acc']}
            df_val_accuracy = pd.DataFrame(d_val_accuracy)
            
            joblib.dump(model,trainedModelPath)
            print("\nSaving the trained model")
            
            plt.figure(20)
            sns.set()
            ax = sns.lineplot(data=df_val_loss)
            ax.set(xlabel='Epochs', ylabel='Loss')
            plt.savefig('Loss_' + sample_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.pdf')
            plt.show()
            plt.clf()

            plt.figure(21)
            sns.set()
            ax = sns.lineplot(data=df_val_accuracy)
            ax.set(xlabel='Epochs', ylabel='Accuracy')
            plt.savefig('Accuracy_' + sample_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.pdf')
            plt.show()
            plt.clf()

        elif args.load_pretrained_model:
            print("\nReading pretrained model:", trainedModelPath)
            model = joblib.load(trainedModelPath)
    t_end = time.time()
    if not args.data:
        # Get predicted signal probabilities for train and test sets
        if args.xgboost:
            output_train = model.predict_proba(X_train_scaled, ntree_limit=model.best_iteration)
            output_test = model.predict_proba(X_test_scaled, ntree_limit=model.best_iteration)

        elif args.nn:
            output_train = model.predict_proba(X_train_scaled)
            output_test = model.predict_proba(X_test_scaled)
        X_train = X_train.copy()
        X_test = X_test.copy()

        print('HALLLLLLLOOOOO!')
        #pred = model.predict(X_test_scaled)
        #acc = accuracy_score(X_test_scaled, pred)
        #print(acc)
        print("\nBefore X_train['output'] = output_train[:,1]: X_train =\n", X_train.head())
        X_train["output"] = output_train[:,1]
        X_test["output"] = output_test[:,1]
        print("\nAfter X_train['output'] = output_train[:,1]: X_train =\n", X_train.head())
        X_train.to_hdf('/storage/monande/significance/TrainOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.h5', key = 'trainOutput', mode = 'w')
        X_test.to_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.h5', key = 'testOutput', mode = 'w')
        #print(process_name, sample_name)
        #store_output = ('/storage/monande/significance/outputFromML_' + process_name + '.h5')
        #store_output.append(X_test)
        #store_output.close()
        #print(type(X_test))
        
        

    elif args.data:
        #print(X_test_scaled.columns)
        print(df_data_feat.shape, X_test.shape, X_data_scaled.shape, X_test_scaled.shape)
        print(X_data_scaled, X_test_scaled)
        if args.xgboost:
            output_train = model.predict_proba(X_train_scaled, ntree_limit=model.best_iteration)
            output_test = model.predict_proba(X_test_scaled, ntree_limit=model.best_iteration)
            output_data = model.predict_proba(X_data_scaled, ntree_limit=model.best_iteration)

        elif args.nn:
            output_train = model.predict_proba(X_train_scaled)
            output_test = model.predict_proba(X_test_scaled)
            output_data = model.predict_proba(X_data_scaled)

        print(type(output_test))
        X_train = X_train.copy()
        X_test = X_test.copy()
        print("\nBefore X_test['output'] = output_test[:,1]: X_test =\n", X_test.head())
        X_train["output"] = output_train[:,1]
        X_test["output"] = output_test[:,1]
            
        print(output_data)
        #print(type(output_data))
        X_data = df_data_feat.copy()
        X_data["output"] = output_data[:,1]
        print("\nAfter X_data['output'] = output_data[:,1]: X_data =\n", X_data.head())
        #X_train.to_hdf('/storage/monande/significance/TrainOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.h5', key = 'trainOutput', mode = 'w')
        #X_test.to_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.h5', key = 'testOutput', mode = 'w')
        X_data.to_hdf('/storage/monande/significance/DataOutputFromML_' + method_type + '_' + process_name + '_'+ level_type + '_' + sig_type + '.h5', key = 'dataOutput', mode = 'w')
    #X_train['Type'] = 'Signal'

    """

    l_bkg_DID = ['Diboson', 'Higgs', 'TopOther', 'SingleTop', 'ttbar', 'Triboson', 'Wjets', 'Zjets']

    X_train_type = X_train.copy()

    X_train_type['Type'] = 'Signal'

    X_test_type = X_test.copy()

    X_test_type['Type'] = 'Signal'

    

    #df1.loc[df1['stream'] == 2, 'feat'] = 10
    

    for elm in Diboson:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Diboson'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Diboson'
    
    for elm in SingleTop:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'SingleTop'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'SingleTop'

    for elm in TopOther:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'TopOther'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'TopOther'

    for elm in ttbar:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'ttbar'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'ttbar'

    for elm in Higgs:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Higgs'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Higgs'

    for elm in Triboson:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Triboson'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Triboson'

    for elm in DY:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'DY'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'DY'

    for elm in Wjets:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Wjets'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Wjets'

    for elm in Zjets:
        X_train_type.loc[X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Zjets'
        X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Zjets'



    print(X_train)
    print(X_train_type)
    """
    """
    X_train = pd.concat([X_train, X_train_type], axis=0)
    X_train.dropna(axis='index', inplace=True)
    print(X_train)
    #print(X_train['Type'])


    X_test = pd.concat([X_test, X_test_type], axis=0)
    X_test.dropna(axis='index', inplace=True)

    #foo = X_train.query('ylabel==0 & DatasetNumber==' + str(363355.0)).loc[:,'output']
    print('Skjera???')
    #print(X_train)
    print('wooooop')
    #print(foo)


    bkg_samples = {'Wjets':{'color':"orange"},
                   'Zjets':{'color':"gold"},
                   'SingleTop':{'color':"lightskyblue"},
                   'TopOther':{'color':'steelblue'},
                   'ttbar':{'color':'dodgerblue'},
                   'Higgs':{'color':'mediumvioletred'},
                   'DY':{'color':'purple'},
                   'Diboson':{'color':"lightgreen"},
                   'Triboson':{'color':'yellowgreen'}}

    print(X_test)
    ML_output = []
    ML_weights = []
    ML_colors = []
    ML_labels = []

    Sig_output = []
    Sig_weights = []
    Sig_colors = []
    Sig_labels = []
    fig_text = []

    ML_data = []
    ML_data_errors = []

    var = 'output'
    top = -999

    stack_order =  ['Data', 'Signal', 'Wjets', 'Triboson', 'DY', 'SingleTop', 'Higgs', 'TopOther', 'ttbar', 'Diboson', 'Zjets']


    data_x = []
    nmax = 1
    nmin = 0
    binw = 20

    bins = np.linspace(0,1,21)#[nmin + (x*binw) for x in range(int((nmax-nmin)/binw)+1)]
    for i in range(len(bins)-1):
        print(bins[i])
        data_x.append(bins[i]+(bins[i+1]-bins[i])/2)

    for s in stack_order:
        print(s)
        if s == 'Data':
            ML_data, _ = np.histogram(X_data.as_matrix(columns=X_data.columns[X_data.columns.get_loc(var):X_data.columns.get_loc(var)+1]), bins=bins)
            ML_data_errors = np.sqrt(ML_data)
        
        elif s == 'Signal':
            rslt_df = X_test.loc[X_test['Type'] == 'Signal']
            print(rslt_df)
            for i in slepslep_high.keys():
                #print(slepslep_high[i])
                j = i.split(',')
                figuretext = r'm($\~l$, $\~\chi_{0}^{1}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                fig_text.append(figuretext)
                rslt_df_sig = rslt_df.loc[rslt_df['DatasetNumber'] == float(slepslep_high[i])]
                #print(rslt_df_sig)
                Sig_output.append(rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc(var):rslt_df_sig.columns.get_loc(var)+1]))
                #print(len(ML_output))
                #print(len(rslt_df))
                Sig_weights.append(3*rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc("eventweight"):rslt_df_sig.columns.get_loc("eventweight")+1]))
                #np.squeeze(Sig_output)
                print(np.shape(Sig_output))
                
                del [rslt_df_sig]

            #Sig_output.append(rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc(var):rslt_df.columns.get_loc(var)+1]))
            
            #print(len(rslt_df))
            #Sig_weights.append(3*rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc("eventweight"):rslt_df.columns.get_loc("eventweight")+1]))
            #print(rslt_df)
            #print('Endelig mandag!')
            #yo = rslt_df['DatasetNumber'].unique()
            #print(yo)
            del [rslt_df]

        else:
            rslt_df = X_test.loc[X_test['Type'] == s]
            ML_output.append(rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc(var):rslt_df.columns.get_loc(var)+1]))
            print(np.shape(ML_output))
            print(len(rslt_df))
            ML_weights.append(3*rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc("eventweight"):rslt_df.columns.get_loc("eventweight")+1]))
            ML_colors.append(bkg_samples[s]['color'])
            ML_labels.append(s)
            del [rslt_df]

        
    #print(X_test.loc[X_test['Type == Signal']])
    #print(Sig_output)
    #print(ML_output)
    print('HALO')
    #print(Sig_output)
    #print(Sig_output[0])

    #plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.hist(ML_output,bins=bins,weights=ML_weights,stacked=True,color=ML_colors, label=ML_labels); #weights=mc_weights,
    plt.errorbar( x=data_x, y=ML_data, yerr=ML_data_errors, fmt='ko', label='Data')
    plt.hist(Sig_output[2],bins=bins, weights= Sig_weights[2], histtype = 'step', color = 'black', label = fig_text[2])
    plt.yscale('log')
    plt.ylabel(r'Events',fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
    plt.xlabel(r'XGBoost output',fontname='sans-serif',horizontalalignment='right',x=1.0,fontsize=11)

    #plt.xlim(0,1)

    plt.ylim(bottom=0,top=10e7)

    
    ax = plt.gca()
    #plt.text(0.45,0.97,r'$\mathbf{{ATLAS}}$ Open Data',ha="left",va="top",family='sans-serif',transform=ax.transAxes,fontsize=13)
    #plt.text(0.45,0.92,'for education only',ha="left",va="top",family='sans-serif',transform=ax.transAxes,style='italic',fontsize=8)
    #plt.text(0.25,0.90,r'$\sqrt{s}=13\,\mathrm{TeV},\;\int L\,dt=58.5\,\mathrm{fb}^{-1}$',ha="left",va="top",family='sans-serif',transform=ax.transAxes)


    plt.legend(ncol = 2)

    plt.savefig('testingstacking.pdf')
    plt.show()

    """

    




    #print(X_train.ix['364126.0'])

    #backgrounds = [363355, 341421, 364280, 410644, 304014, 364242, 410470, 364156, 364100]
    

    #backgrounds = X_train.query('ylabel==0').loc[:, 'DatasetNumber'].drop_duplicates()
    #print(backgrounds)

    #fig, ax = plt.subplots(figsize=(10,7)
    #for i, j in enumerate(backgrounds):
    #    values = list(X_train[X_train['DatasetNumber']==j].loc[:, 'eventweight'])

    #    X_train[X_train['DatasetNumber']==j].plot.bar(x='output', y='eventweight', ax=ax, stacked=True)

   # plt.hist([X_train.query('ylabel==0 & DatasetNumber==363355').loc[:,'output'], X_train.query('ylabel==0 & DatasetNumber==363355').loc[:,'output']], bins=20, stacked=True)


    #X_train_bkg = [X_train[i_bkg]['output'].squeeze() for i_bkg in back]

    #print(X_train_bkg)

    #X_train.plot.barh(stacked=True)
    #plt.savefig('testingplot.pdf')
    #plt.show()

    #fig, ax = plt.subplots(figsize=(10,7))  

    #months = df['Month'].drop_duplicates()
    #margin_bottom = np.zeros(len(df['Year'].drop_duplicates()))
    #colors = ["#006D2C", "#31A354","#74C476"]

    #for num, month in enumerate(months):
    #    values = list(df[df['Month'] == month].loc[:, 'Value'])

    #    df[df['Month'] == month].plot.bar(x='Year',y='Value', ax=ax, stacked=True, 
    #                                bottom = margin_bottom, color=colors[num], label=month)
    #    margin_bottom += values

    #plt.show()



    #df2.plot.bar(stacked=True);
    #df2 = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])












    sum_sig_weights = 1
    sum_bkg_weights = 1
    sum_sig_scaled_weights = 1
    sum_bkg_scaled_weights = 1
    sum_bkg_scaled_weights_train = 1




    

    print("\n\n//////////////////// Plotting part ////////////////////////\n")
    if args.xgboost:
        xlabelText = 'XGBoost output'
        # Set seaborn style
        sns.set(color_codes=True)
    
        if not args.data:
            print("len(X_train.query('ylabel==0').loc[:,'eventweight'])", len(X_train.query('ylabel==0').loc[:,'eventweight']))
            print("len(X_train.query('ylabel==0').loc[:,'output'])", len(X_train.query('ylabel==0').loc[:,'output']))
            print("X_train.query('ylabel==0').loc[:,'eventweight']", X_train.query("ylabel==0").loc[:,"eventweight"].head())
            print("X_train.query('ylabel==0').loc[:,'output']", X_train.query("ylabel==0").loc[:,"output"].head())

            print("X_train[['eventweight', 'output']].min(): \n", X_train[['eventweight', 'output']].min())
            print("X_train[['eventweight', 'output']].max(): \n", X_train[['eventweight', 'output']].max())

            if args.slepslep:
                if args.low:
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/Low/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepslep_low.keys():
                        print(slepslep_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepslep_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/Inter/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepslep_inter.keys():
                        print(slepslep_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepslep_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/High/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepslep_high.keys():
                        print(slepslep_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepslep_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            elif args.slepsnu:
                if args.low:
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/Low/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepsnu_low.keys():
                        print(slepsnu_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepsnu_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/Inter/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepsnu_inter.keys():
                        print(slepsnu_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepsnu_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/High/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in slepsnu_high.keys():
                        print(slepsnu_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, slepsnu_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            elif args.WW:
                if args.low:
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/Low/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in WW_low.keys():
                        print(WW_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, WW_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/Inter/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in WW_inter.keys():
                        print(WW_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, WW_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/High/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in WW_high.keys():
                        print(WW_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, WW_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            if args.monoZ:
                if args.low:
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/Low/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in MonoZ_low.keys():
                        print(MonoZ_low[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, MonoZ_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/Inter/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in MonoZ_inter.keys():
                        print(MonoZ_inter[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, MonoZ_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/High/'
                    plotSetup(results_path, X_train, X_test, model)
                    for i in MonoZ_high.keys():
                        print(MonoZ_high[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoop(results_path, MonoZ_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)



        if args.data:
            if args.slepslep:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/Low/Data/'
                    for i in slepslep_low.keys():
                        print(slepslep_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/Inter/Data/'
                    for i in slepslep_inter.keys():
                        print(slepslep_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSlep/' + level_type +'/High/Data/'
                    for i in slepslep_high.keys():
                        print(slepslep_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.slepsnu:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/Low/Data/'
                    for i in slepsnu_low.keys():
                        print(slepsnu_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/Inter/Data/'
                    for i in slepsnu_inter.keys():
                        print(slepsnu_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/SlepSnu/' + level_type +'/High/Data/'
                    for i in slepsnu_high.keys():
                        print(slepsnu_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.WW:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/Low/Data/'
                    for i in WW_low.keys():
                        print(WW_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/Inter/Data/'
                    for i in WW_inter.keys():
                        print(WW_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/WW/' + level_type +'/High/Data/'
                    for i in WW_high.keys():
                        print(WW_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.monoZ:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/Low/Data/'
                    for i in MonoZ_low.keys():
                        print(MonoZ_low[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                        # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/Inter/Data/'
                    for i in MonoZ_inter.keys():
                        print(MonoZ_inter[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/BDT/MonoZ/' + level_type +'/High/Data/'
                    for i in MonoZ_high.keys():
                        print(MonoZ_high[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_high[i] + '.pdf')
                        plt.show()
                        plt.close()



        # Signal significance
        cut = 0.8
        cut_string = 'output > {:f}'.format(cut)
        print('\ncut_string', cut_string)
        bkg_exp = np.sum(3.33*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(3.33*X_test.query("ylabel == 1 & "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.3)
        #data_exp = np.sum(X_data["eventweight" + cut_string])
        print("\n///////////////// Signal significance /////////////////")
        print("\nS_exp =", sig_exp)
        print("B_exp =", bkg_exp)
        print("Z_N_exp =", Z_N_exp)
        #print("Data_exp =", data_exp)


    elif args.nn:
        xlabelText = 'Neural network output'
        # Set seaborn style
        sns.set(color_codes=True)
    
        if not args.data:
            print("len(X_train.query('ylabel==0').loc[:,'eventweight'])", len(X_train.query('ylabel==0').loc[:,'eventweight']))
            print("len(X_train.query('ylabel==0').loc[:,'output'])", len(X_train.query('ylabel==0').loc[:,'output']))
            print("X_train.query('ylabel==0').loc[:,'eventweight']", X_train.query("ylabel==0").loc[:,"eventweight"].head())
            print("X_train.query('ylabel==0').loc[:,'output']", X_train.query("ylabel==0").loc[:,"output"].head())

            print("X_train[['eventweight', 'output']].min(): \n", X_train[['eventweight', 'output']].min())
            print("X_train[['eventweight', 'output']].max(): \n", X_train[['eventweight', 'output']].max())

            if args.slepslep:
                if args.low:
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/Low/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepslep_low.keys():
                        print(slepslep_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepslep_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/Inter/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepslep_inter.keys():
                        print(slepslep_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepslep_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/High/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepslep_high.keys():
                        print(slepslep_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepslep_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            elif args.slepsnu:
                if args.low:
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/Low/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepsnu_low.keys():
                        print(slepsnu_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepsnu_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/Inter/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepsnu_inter.keys():
                        print(slepsnu_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepsnu_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/High/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in slepsnu_high.keys():
                        print(slepsnu_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, slepsnu_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            elif args.WW:
                if args.low:
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/Low/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in WW_low.keys():
                        print(WW_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, WW_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/Inter/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in WW_inter.keys():
                        print(WW_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, WW_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/High/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in WW_high.keys():
                        print(WW_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, WW_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

            elif args.monoZ:
                if args.low:
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/Low/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in MonoZ_low.keys():
                        print(MonoZ_low[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, MonoZ_low, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.inter:
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/Inter/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in MonoZ_inter.keys():
                        print(MonoZ_inter[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, MonoZ_inter, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)

                elif args.high:
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/High/'
                    plotSetupNN(results_path, X_train, X_test, model)
                    for i in MonoZ_high.keys():
                        print(MonoZ_high[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        plotLoopNN(results_path, MonoZ_high, i, X_train, X_test, scale_fac, sum_sig_weights, sum_bkg_weights, sum_sig_scaled_weights, sum_bkg_scaled_weights, sum_bkg_scaled_weights_train, figuretext)



        if args.data:
            if args.slepslep:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/Low/Data/'
                    for i in slepslep_low.keys():
                        print(slepslep_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{0}^{1}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/Inter/Data/'
                    for i in slepslep_inter.keys():
                        print(slepslep_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{0}^{1}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSlep/' + level_type +'/High/Data/'
                    for i in slepslep_high.keys():
                        print(slepslep_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~l$, $\~\chi_{0}^{1}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepslep_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepslep_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepslep_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.slepsnu:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/Low/Data/'
                    for i in slepsnu_low.keys():
                        print(slepsnu_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~l$/$\~\nu$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/Inter/Data/'
                    for i in slepsnu_inter.keys():
                        print(slepsnu_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~l$/$\~\nu$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/SlepSnu/' + level_type +'/High/Data/'
                    for i in slepsnu_high.keys():
                        print(slepsnu_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $\~l$/$\~\nu$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(slepsnu_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ slepsnu_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + slepsnu_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.WW:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/Low/Data/'
                    for i in WW_low.keys():
                        print(WW_low[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $W^{\pm}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/Inter/Data/'
                    for i in WW_inter.keys():
                        print(WW_inter[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $W^{\pm}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/WW/' + level_type +'/High/Data/'
                    for i in WW_high.keys():
                        print(WW_high[i])
                        j = i.split(',')
                        figuretext = r'm($\~\chi_{1}^{\pm}$, $W^{\pm}$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(WW_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ WW_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ WW_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + WW_high[i] + '.pdf')
                        plt.show()
                        plt.close()

            elif args.monoZ:
                if args.low:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/Low/Data/'
                    for i in MonoZ_low.keys():
                        print(MonoZ_low[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_low[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_low[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_low[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_low[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.inter:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/Inter/Data/'
                    for i in MonoZ_inter.keys():
                        print(MonoZ_inter[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_inter[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_inter[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_inter[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_inter[i] + '.pdf')
                        plt.show()
                        plt.close()

                elif args.high:
                    # Plot final signal vs background estimate for test set, scaled to 10.6/fb
                    results_path = '/storage/monande/plots/NN/MonoZ/' + level_type +'/High/Data/'
                    for i in MonoZ_high.keys():
                        print(MonoZ_high[i])
                        j = i.split(',')
                        figuretext = r'm(V, $\chi$) ('+j[0] + ',' + j[1]+r') $\rightarrow \Delta$ m = ' + j[2]
                        #figuretext = 'hallo'
                        print(figuretext)
                        plt.figure(MonoZ_high[i])
                        plotFinalTestOutputWithData(X_test.query("ylabel==0").loc[:,"output"],
                                                    3.33*X_test.query("ylabel==0").loc[:,"eventweight"],
                                                    X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_high[i]).loc[:,"output"],
                                                    3.33*X_test.query("ylabel==1 & DatasetNumber=="+ MonoZ_high[i]).loc[:,"eventweight"],
                                                    X_data["output"],
                                                    np.ones(len(X_data["output"])),
                                                    figuretext, xlabelText) #X_data["eventweight"])#,
                        plt.savefig(results_path + 'finaldata_' + MonoZ_high[i] + '.pdf')
                        plt.show()
                        plt.close()




        # Signal significance
        cut = 0.8
        cut_string = 'output > {:f}'.format(cut)
        print('\ncut_string', cut_string)
        bkg_exp = np.sum(3.33*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(3.33*X_test.query("ylabel == 1 & "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.3)
        #data_exp = np.sum(X_data["eventweight" + cut_string])
        print("\n///////////////// Signal significance /////////////////")
        print("\nS_exp =", sig_exp)
        print("B_exp =", bkg_exp)
        print("Z_N_exp =", Z_N_exp)
        #print("Data_exp =", data_exp)



    # Stop timer
    
    print("\nProcess time: {:4.2f} s".format(t_end - t_start))
    

if __name__ == "__main__":
    main()
