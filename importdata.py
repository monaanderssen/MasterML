import h5py
import uproot
import numpy as np
import pandas as pd
from ROOT import TLorentzVector
from sklearn.utils import shuffle

from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2, l1_l2

#from infofile import infos
#from hepFunctions import invariantMass


# Features/variables to use for classification
l_features = ['DatasetNumber', 'RandomRunNumber', 'lepPt1', 'lepPt2', 'lepEta1', 'lepEta2', 'lepPhi1', 'lepPhi2', 'nJet20', 'nJet30', 'nBJet20_MV2c10_FixedCutBEff_85', 'mll', 'mt2', 'met_Et', 'met_Sign', 'met_HT', 'deltaPhi', 'deltaRll', 'pTdiff', 'HT']
"""
["channelNumber",
              "lep_pt1", "lep_eta1", "lep_phi1", "lep_E1",
              "lep_pt2", "lep_eta2", "lep_phi2", "lep_E2",
              "met_et", "met_phi",
              "mll"]
"""
# Event weights to apply, except for xsec and sumw,
# which are being handled in getEventWeights
l_eventweights = ['eventWeight',
                  'pileupWeight',
                  'leptonWeight',
                  'bTagWeight',
                  'genWeight',
                  'jvtWeights',
                  'globalDiLepTrigSF']

#['mcWeight',
#                'scaleFactor_PILEUP',
#                'scaleFactor_ELE',
#                'scaleFactor_MUON',
#                'scaleFactor_BTAG',
#                'scaleFactor_LepTRIGGER',
#                'xsec_fb',
#                '1_over_sumw']

n_events_chunk = 999999999999


def importDatasetFromHDF5(filepath, dataset_name):
    """Read hdf5 input file"""

    with h5py.File(filepath, 'r') as hf:
        ds = hf[dataset_name]
        df = np.DataFrame(data=ds[:])

    return df


def importDatasetFromROOTFile(filepath, treename, entry_start=None, entry_stop=None, flatten_bool=None, branches="*"):
    """Import TTree with jagged arrays from ROOT file using uproot"""

    tree = uproot.open(filepath)[treename]
    df = tree.pandas.df(branches, entrystart=entry_start, entrystop=entry_stop, flatten=flatten_bool)

    if "OpenData" in filepath:
        if "jet" in branches:
            df = df.loc[:,"jet_pt":].copy()
        elif "lep" in branches:
            df = df.loc[:,"lep_truthMatched":].copy()

    elif "histfitter" in filepath:
        if "lep" in branches:
            df = df.loc[:,:"lepM"].copy()

    return df


def applyCuts(df, cuts):
    """Apply cuts to events/rows in the dataset"""

    df = df.query("&".join(cuts))

    return df


def getEventWeights(df, l_eventweight_column_names):
    """Return pandas Dataframe with a single combined eventweight calculated by
    multiplying the columns in the DataFrame that contain different 
    and normalizing to the integrated luminosity of the simulated samples"""

    # Multiply all eventweights except for xsec and sumw
    s_eventweight = df.loc[:,l_eventweight_column_names].agg('prod', axis='columns').copy()
    s_eventweight = s_eventweight.values.reshape(-1,1)
    print("\n- s_eventweight.shape =\n", s_eventweight.shape)
    print(type(df))
    #df['Lumi'] = 44300
    #df.loc[df['RandomRunNumber'] < 320000, 'Lumi'] = 36200
    #df.loc[df['RandomRunNumber'] > 348000, 'Lumi'] = 58500
    
    # Normalize to integrated luminosity: N = L*xsec =>  1/L = xsec/N
      # [fb-1] int lumi in data16 period A-D
    #xsec_pb_to_fb = 1e3  # convert xsec from pb to fb
    #s_eventweight *= 1 #df.loc[df['Lumi']] #* xsec_pb_to_fb
    #s_eventweight = s_eventweight[s_eventweight > 0]  # remove events with negative eventweights
    print("\nNumber of zeros or NaNs:", df.isna().sum().sum())
    #print("\n- After dropping events/rows where eventweight is 0:\ns_eventweight.shape", s_eventweight.shape)
    print("\nNumber of zeros or NaNs:", df.isna().sum().sum())

    return s_eventweight


def selectFeatures(df, l_column_names):
    """Select subset of variables/features in the df"""

    df_features = df.loc[:,l_column_names].copy()
    print("\n- After selecting features:\ndf_features.shape", df_features.shape)

    return df_features


def dropVariables(df, l_drop_columns):
    """Drop subset of variables in the df"""

    df_new = df.drop(columns=l_drop_columns, inplace=True)

    return df_new



def importData(sample_type, sample_name,treename, filepath, entrystart=None, entrystop=None):
    """Import OpenData ntuples"""

    print("\n=======================================")
    print("\n sample_type =", sample_type)
    print("\n=======================================")
    df_flat = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop)
    print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)
    print(df_flat['eventWeight'])
    global n_events_chunk
    n_events_chunk = len(df_flat)
    print("\nIn importData(): n_events_chunk =", n_events_chunk)

#############################################################
#                                                           #
#            Two leptons                                    #
#                                                           #
#############################################################
    # Preselection: Trigger + >= 2 lep + >= 2 jets
    l_cuts_presel = ["nLep_base == 2", "nLep_signal == 2"]#, "lepCharge[0] != lepCharge[1]"]
    df_flat = applyCuts(df_flat, l_cuts_presel)

    print("\n- Preselection cuts applied:")
    for i_cut in l_cuts_presel:
        print(i_cut)
    print("\n- After cuts:\ndf_flat.shape", df_flat.shape)
    #print(df_flat['eventWeight'])
    print("\n- After importing flat DF:\ndf_flat.shape", df_flat.shape)

    
#############################################################
#                                                           #
#           b-Jets                                          #
#                                                           #
#############################################################

    print("\n---------------------------------------")
    df_jet = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="nBJet*")
    #df_jet = df_jet.unstack()
    #print("\n- After importing jet DF:\ndf_flat.shape", df_jet.shape)
    #df_jet = df_jet.stack()
    
    #l_cuts_jet = ["jet_pt > 50e3", "abs(jet_eta) < 2.8"]
    l_cuts_bjet = ["nBJet20_MV2c10_FixedCutBEff_85 == 0"] 
    
    #df_jet = df_jet.astype({"jetEta":float})
    df_jet = applyCuts(df_jet, l_cuts_bjet)
    #print(df_jet)
    #df_jet.columns([' ', 'b-jet'])
    

    #print(df_jet.shape)
    n_jet_sig = df_jet.groupby(level=0).size().values
    #df_jet = df_jet.unstack()
    df_jet = df_jet[n_jet_sig == 2].copy()

    #df_jet = df_jet.stack()
    df_jet["subindex"] = df_jet.groupby(level=0).apply(lambda x : x.sort_values("nBJet20_MV2c10_FixedCutBEff_85", ascending=False)).groupby(level=0).cumcount()
    #print(df_jet)
    df_jet = df_jet.set_index("subindex", append=True)
    #df_jet_temp = df_jet.copy()
    #df_jet['b-jet'] = df_jet['nBJet20_MV2c10_FixedCutBEff_77']
    #df_jet = df_jet_temp.copy()
    #print(df_jet)
    #df_jet = df_jet.reset_index(level="subentry").drop("subentry", axis=1)
    df_jet = df_jet.unstack(level="subindex")

    #print(type(df_jet))#, type(col))

    #print(df_jet)

    df_jet.columns = [ str(col[0]) + str(col[1]+1) for col in df_jet.columns.values]
    #print(df_jet)
  
    df_jet.dropna(axis='columns', inplace=True)
    
    #print(df_jet)
    #print('YoHello!')
    #df_jet_temp = df_jet.copy()
    #df_jet_temp['b-jet'] = df_jet['nBJet20_MV2c10_FixedCutBEff_77']
    #df_jet["b-jet"] = df_jet 
    #df_jet["b-jet"] = df_jet["nBJet20_MV2c10_FixedCutBEff_77"]
    #df_jet = df_jet_temp.copy()
    #print(df_jet)

    #df_jet['b-jet'] = df_jet
    #print(df_jet["nBJet20_MV2c10_FixedCutBEff_77"])
    df_flat = pd.concat([df_flat, df_jet], axis=1, sort=False)
    print("\n- After concatenating jet DF with flat DF:\ndf_flat.shape", df_flat.shape)
    #print(df_flat['eventWeight'])

    print("\n---------------------------------------")
    
    
#############################################################
#                                                           #
#           MET                                             #
#                                                           #
#############################################################

    df_met = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="met_E*")
    cuts_met = ["met_Et > 40"]
    df_met = applyCuts(df_met, cuts_met)
    df_met.dropna(axis='columns', inplace=True)
    df_flat = pd.concat([df_flat, df_met], axis=1, sort=False)
    #print(df_flat['eventWeight'])
    
#############################################################
#                                                           #
#           OS lep                                          #
#                                                           #
#############################################################

    print("\n---------------------------------------")
    df_lep = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="lep*")
    #df_lep = df_lep.unstack()
    #print("\n- After importing lep DF:\ndf_flat.shape", df_lep.shape)
    #df_lep = df_lep.stack()

    
    #Fixing the feature names as lepPt1,2 osv.
    n_lep_sig = df_lep.groupby(level=0).size().values
    df_lep = df_lep.unstack()
    df_lep = df_lep[n_lep_sig == 2].copy()

    df_lep = df_lep.stack()
    df_lep_sorted = df_lep.groupby(level=0).apply(lambda x : x.sort_values("lepPt", ascending=False))
    #df_lep_sorted = df_lep_sorted.reset_index(level=0).drop("entry", axis=1)
    
    df_lep["subindex"] = df_lep_sorted.groupby(level=0).cumcount()
    df_lep = df_lep.set_index("subindex", append=True)
    df_lep = df_lep.reset_index(level="subentry").drop("subentry", axis=1)
    df_lep = df_lep.unstack(level="subindex")
  
    df_lep.columns = [col[0]+str(col[1]+1) for col in df_lep.columns.values]

    l_cuts_OS = ["lepCharge1 != lepCharge2"]
    #l_cuts_ZonShell = ['(mll < 81e3 | mll > 101e3)']
    df_lep = applyCuts(df_lep, l_cuts_OS)

    print("\n- Lepton cuts applied:")
    for i_cut in l_cuts_OS + ["nLep_signal == 2 == 2"] + l_cuts_OS:
        print(i_cut)
    print("\n- After cuts:\ndf_flat.shape", df_lep.shape)

    df_flat = pd.concat([df_flat, df_lep], axis=1, sort=False)
    print("\n- After concatenating lep DF with flat DF:\ndf_flat.shape", df_flat.shape)
    #print(df_flat['eventWeight'])

    df_flat.dropna(axis='index', inplace=True)
    print("\n- After dropping events/rows with NaNs:\ndf_flat.shape", df_flat.shape)
    #print(df_flat['eventWeight'])
    
    #df_flat = df_flat.loc[:,~df_flat.columns.duplicated()]

    #df_flat_type = df_flat.copy()
    #df_flat_type['Type'] = ' '

    #print(np.shape(df_flat), np.shape(df_flat_type))

    """
    if sample_type == 'sig' or 'bkg':
        dsid = treename.split('_NoSys')
        print(dsid[0])
        df_flat['Type'] = dsid[0]
    else:
        df_flat['Type'] = 'Data'
    
    #df_flat = pd.concat([df_flat, df_flat_type], axis = 0)
    df_flat.dropna(axis='index', inplace=True)
    print(df_flat)
    """
    return df_flat


def prepareInput(store, sample_type, sample_name,treename, filepath, filename, chunk_size=1e4, n_chunks=100, entrystart=0):
    """Read in dataset in chunks, preprocess for ML and store to HDF5"""

    print("\nPrepare input")
  

    for i_chunk in range(1, n_chunks+1):

        print("\nReading chunk #{:d}".format(i_chunk))
        entrystop = i_chunk*chunk_size  # entrystop exclusive

        df = importData(sample_type,sample_name,treename, filepath, entrystart, entrystop)
        df = shuffle(df)  # shuffle the rows/events
    
        df_feat = selectFeatures(df, l_features)
        #print(df_feat)
        df_feat = df_feat*1  # multiplying by 1 to convert booleans to integers
        #print(df_feat)
        df_feat["eventweight"] = getEventWeights(df, l_eventweights)
        print("\n- After adding calculating and adding eventweight column:\ndf_feat.shape", df_feat.shape)
        #print(df_feat)

        df_feat = df_feat[df_feat.eventweight > 0]
        print("\n- After removing events/rows with eventweight <= 0:\ndf_feat.shape", df_feat.shape)
        #print(df_feat)
        print("\ndf.head()\n", df_feat.head())
        #print("i_chunk = ",i_chunk)
        #print("df_feat = ", df_feat.dtypes)
        #print("\nstore.info\n", store.info)
        #print("\nstore.keys\n", store.keys)
        if i_chunk is 1:
            print(sample_type, df_feat)
            store.append(sample_type, df_feat.astype({"RandomRunNumber":np.int64,
                                                      "DatasetNumber":np.int64,
                                                      "nJet20":np.int64,
                                                      "nJet30":np.int64,
                                                      "nBJet20_MV2c10_FixedCutBEff_85":np.int64}))
            #print(pd.read_hdf(store))
            print("\nStored initial chunk in HDF5 file")
        else:
            print(sample_type, df_feat)
            store.append(sample_type, df_feat.astype({"RandomRunNumber":np.int64,
                                                      "DatasetNumber":np.int64,
                                                      "nJet20":np.int64,
                                                      "nJet30":np.int64,
                                                      "nBJet20_MV2c10_FixedCutBEff_85":np.int64}))
            print("\nAppended chunk #{:d} in HDF5 file".format(i_chunk))

        print("\nIn prepareInput(): n_events_chunk =", n_events_chunk)
        if n_events_chunk < chunk_size:
            print("\nReached end of dataset --> do not try to read in another chunk")
            break

        entrystart = entrystop  # entrystart inclusive

    return store



def create_model(n_inputs=20, n_hidden_layers=1, n_nodes=20, dropout_rate=0., l1=0., l2=0., lr=0.001):
  model = Sequential()

  print("\nSetting up neural network with {0} hidden layers and {1} nodes in each layer".format(n_hidden_layers, n_nodes))
  print("\n")

  kernel_regularizer = l1_l2(l1=l1, l2=l2)

  # Add input + first hidden layer
  model.add(Dense(n_nodes, input_dim=n_inputs, activation="relu", 
                  use_bias=True, 
                  kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                  kernel_regularizer=kernel_regularizer, bias_regularizer=None, activity_regularizer=None, 
                  kernel_constraint=None, bias_constraint=None))

  # Add hidden layers
  hidden_layers_counter = 1
  for i_hidden_layer in range(n_hidden_layers-1):

    if dropout_rate > 0.:
      # Add dropout layer before every normal hidden layer
      print("Adding droput layer with dropout rate of", dropout_rate)
      model.add(Dropout(dropout_rate, noise_shape=None, seed=None))

    hidden_layers_counter += 1
    print("Adding hidden layer #", hidden_layers_counter)
    print("\n")
    model.add(Dense(n_nodes, activation='relu',
                    use_bias=True, 
                    kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                    kernel_regularizer=kernel_regularizer, bias_regularizer=None, activity_regularizer=None, 
                    kernel_constraint=None, bias_constraint=None))

  # Add output layer
  model.add(Dense(1, activation="sigmoid"))

  model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                loss="binary_crossentropy", metrics=['accuracy']
                #metrics=['binary_crossentropy']#, 'accuracy']#, 'balanced_accuracy_score']
                )
  return model
