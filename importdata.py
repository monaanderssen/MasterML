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

# Event weights to apply, except for xsec and sumw,
# which are being handled in getEventWeights
l_eventweights = ['eventWeight',
                  'pileupWeight',
                  'leptonWeight',
                  'bTagWeight',
                  'genWeight',
                  'jvtWeights',
                  'globalDiLepTrigSF']


n_events_chunk = 999999999999

def label_isem(row):
    if (row["lepFlavor1"] == 1.0 and row["lepFlavor2"] == 2.0) or (row["lepFlavor1"] == 2.0 and row["lepFlavor2"] == 1.0):
        return 1
    return 0
def label_isee(row):
    if (row["lepFlavor1"] == 1.0 and row["lepFlavor2"] == 1.0):
        return 1
    return 0
def label_ismm(row):
    if (row["lepFlavor1"] == 2.0 and row["lepFlavor2"] == 2.0):
        return 1
    return 0
def label_isos(row):
    if (row["lepCharge1"]*row["lepCharge2"] < 0):
        return 1
    return 0
def label_isss(row):
    if (row["lepCharge1"]*row["lepCharge2"] > 0):
        return 1
    return 0


def importDatasetFromHDF5(filepath, dataset_name):
    """Read hdf5 input file"""

    with h5py.File(filepath, 'r') as hf:
        ds = hf[dataset_name]
        df = np.DataFrame(data=ds[:])

    return df


def importDatasetFromROOTFile(filepath, treename, entry_start=None, entry_stop=None, flatten_bool=None, branches="*"):
    """Import TTree with jagged arrays from ROOT file using uproot"""

    tree = uproot.open(filepath)[treename]
    global total_entries
    total_entries = tree.numentries
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
    global n_events_chunk
    n_events_chunk = len(df_flat)
    print("\nIn importData(): n_events_chunk =", n_events_chunk)

#############################################################
#                                                           #
#            Two leptons                                    #
#                                                           #
#############################################################
    # Preselection: Trigger + >= 2 lep + >= 2 jets
    l_cuts_presel = ["nLep_base == 2", "nLep_signal == 2"]
    df_flat = applyCuts(df_flat, l_cuts_presel)

    print("\n- Preselection cuts applied:")
    for i_cut in l_cuts_presel:
        print(i_cut)
    print("\n- After cuts:\ndf_flat.shape", df_flat.shape)


    
#############################################################
#                                                           #
#           b-Jets                                          #
#                                                           #
#############################################################

    print("\n---------------------------------------")
    df_jet = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="nBJet*")
    
    df_jet.dropna(axis='columns', inplace=True)
       
    df_flat = pd.concat([df_flat, df_jet], axis=1, sort=False)
    print("\n- After concatenating jet DF with flat DF:\ndf_flat.shape", df_flat.shape)
   
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
       
#############################################################
#                                                           #
#           OS lep                                          #
#                                                           #
#############################################################

    print("\n---------------------------------------")
    df_lep = importDatasetFromROOTFile(filepath, treename, entrystart, entrystop, flatten_bool=True, branches="lep*")
    
    n_lep_sig = df_lep.groupby(level=0).size().values

    for i in range(2,max(n_lep_sig)):
        df_lep.drop(index=i,level=1, inplace=True)
    df_lep = df_lep.unstack()
    df_lep = df_lep[n_lep_sig >= 2].copy()

    df_lep = df_lep.stack()
    df_lep_sorted = df_lep.groupby(level=0).apply(lambda x : x.sort_values("lepPt", ascending=False))
    
    df_lep["subindex"] = df_lep_sorted.groupby(level=0).cumcount()
    df_lep = df_lep.set_index("subindex", append=True)
    df_lep = df_lep.reset_index(level="subentry").drop("subentry", axis=1)
    df_lep = df_lep.unstack(level="subindex")
  
    df_lep.columns = [col[0]+str(col[1]+1) for col in df_lep.columns.values]

    print("\n- Lepton cuts applied:")
    for i_cut in  ["nLep_signal == 2"]: #l_cuts_OS +
        print(i_cut)
    print("\n- After cuts:\ndf_flat.shape", df_lep.shape)

    df_flat = pd.concat([df_flat, df_lep], axis=1, sort=False)
    print("\n- After concatenating lep DF with flat DF:\ndf_flat.shape", df_flat.shape)

    df_flat.dropna(axis='index', inplace=True)
    print("\n- After dropping events/rows with NaNs:\ndf_flat.shape", df_flat.shape)
    
    return df_flat


def prepareInput(store, sample_type, sample_name,treename, filepath, filename, chunk_size=1e4, n_chunks=100, entrystart=0):
    """Read in dataset in chunks, preprocess for ML and store to HDF5"""

    print("\nPrepare input")


    startTime = datetime.now()

    for i_chunk in range(1, n_chunks+1):

        if i_chunk > 1:
            diff = (datetime.now() - startTime)
            #print("diff",diff.total_seconds())
            ev_sec = chunk_size/diff.total_seconds()
            #print("ev_sec",ev_sec)
            ev_left = (total_entries - ((float(i_chunk)-1.0)*chunk_size))
            #print("ev_left",ev_left)
            time_left = ev_left/ev_sec
            m, s = divmod(time_left, 60)
            h, m = divmod(m, 60)
            print("#"*100)
            print('Reading {:.0f} ev/sec with chunk size {:d}.\n\nStarting chunk #{:02d} (of {:02d}). Estimated time left : {:d}h{:02d}m{:02d}s'.format(ev_sec,int(chunk_size),int(i_chunk),int(total_entries/chunk_size)+1,int(h), int(m), int(s))) # Python 3
            print("#"*100)
            #print(f'{h:d}:{m:02d}:{s:02d}') # Python 3.6+
            startTime = datetime.now()
        print("\nReading chunk #{:d}".format(i_chunk))
        entrystop = i_chunk*chunk_size  # entrystop exclusive

        df = importData(sample_type,sample_name,treename, filepath, entrystart, entrystop)
        df = shuffle(df)  # shuffle the rows/events

        # Making some new features
        #print("dataframe---->",df)
        print("Adding signa and flavour features")
        df["isEM"] = df.apply(lambda row: label_isem(row),axis = 1)
        df["isEE"] = df.apply(lambda row: label_isee(row),axis = 1)
        df["isMM"] = df.apply(lambda row: label_ismm(row),axis = 1)
        df["isOS"] = df.apply(lambda row: label_isos(row),axis = 1)
        df["isSS"] = df.apply(lambda row: label_isss(row),axis = 1)


        print("Shape after adding sign and flavour feature = ",df.shape)
        df_feat = selectFeatures(df, l_features + ["isEM","isEE","isMM","isOS","isSS"])

        df_feat = df_feat*1  # multiplying by 1 to convert booleans to integers

        df_feat["eventweight"] = getEventWeights(df, l_eventweights)
        print("\n- After adding calculating and adding eventweight column:\ndf_feat.shape", df_feat.shape)

        if i_chunk is 1:
            store.append(sample_type, df_feat.astype({"RandomRunNumber":np.int64,
                                                      "DatasetNumber":np.int64,
                                                      "nJet20":np.int64,
                                                      "nJet30":np.int64,
                                                      "nBJet20_MV2c10_FixedCutBEff_85":np.int64}))

            print("\nStored initial chunk in HDF5 file")
        else:
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
  """ Creating the Neural Network """
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
                )
  return model
