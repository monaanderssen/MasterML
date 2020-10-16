from __future__ import division

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba 
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import pylatex
from matplotlib.colors import *
from matplotlib import gridspec


from ROOT import TLorentzVector, RooStats
import uproot
import pandas as pd
import numpy as np
import argparse

from samples import *
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from importdata import *
from plotting import *
from samples import *
from datasetnumber_bkg import *

parser = argparse.ArgumentParser()
group_model = parser.add_mutually_exclusive_group() 
group_model.add_argument('-x', '--xgboost', action='store_true', help='Run gradient BDT')
group_model.add_argument('-n', '--nn', action='store_true', help='Run neural network')
group_read_dataset = parser.add_mutually_exclusive_group() 
group_read_dataset.add_argument('-p', '--prepare_hdf5', action='store_true', help='Prepare input datasets for ML and store in HDF5 file')
group_read_dataset.add_argument('-r', '--read_hdf5', action='store_true', help='Read prepared datasets from HDF5 file')
    
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

if args.xgboost:
    method_type = 'BDT'
    scalefac = 3.34#*(139/58.4501)
    xLabel = 'XGBoost output'
elif args.nn:
    method_type = 'NN'
    scalefac = 3.34#*(139/58.4501)
    xLabel = 'Neural network output'

if args.slepslep:

    process_name = 'slepslep'
    signal_low = '395984'
    figuretext_low = r'm($\~l$, $\~\chi_{0}^{1}$) (400, 300)'
    signalDict_low = slepslep_low
    signal_inter = '396014'
    figuretext_inter = r'm($\~l$, $\~\chi_{0}^{1}$) (600, 300)'
    signalDict_inter = slepslep_inter
    signal_high = '396033'
    figuretext_high = r'm($\~l$, $\~\chi_{0}^{1}$) (700, 1)'
    signalDict_high = slepslep_high
elif args.slepsnu:

    process_name = 'slepsnu'
    signal_low = '397115'
    figuretext_low = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (300, 200)'
    signalDict_low = slepsnu_low
    signal_inter = '397150'
    figuretext_inter = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (800, 400)'
    signalDict_inter = slepsnu_inter
    signal_high = '397169'
    figuretext_high = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (1000, 100)'
    signalDict_high = slepsnu_high
elif args.WW:

    process_name = 'WW'
    signal_low = '395268'
    figuretext_low = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (150, 25)'
    signalDict_low = WW_low
    signal_inter = '395320'
    figuretext_inter = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (350, 100)'
    signalDict_inter = WW_inter
    signal_high = '395330'
    figuretext_high = r'm($\~\chi_{1}^{\pm}$, $\~\chi_{1}^{0}$) (425, 25)'
    signalDict_high = WW_high
elif args.monoZ:

    process_name = 'monoZ'
    signal_low = '310604'
    figuretext_low = r'm(V, $\chi$) (150, 80)'
    signalDict_low = MonoZ_low
    signal_inter = '310613'
    figuretext_inter = r'm(V, $\chi$) (400, 150)'
    signalDict_inter = MonoZ_inter
    signal_high = '310617'
    figuretext_high = r'm(V, $\chi$) (650, 1)'
    signalDict_high = MonoZ_high
if args.high_level:
    level_type = 'High_level'
elif args.low_level:
    level_type = 'Low_level'
else:
    level_type = 'All_level'


X_test_low = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_low.h5', 'testOutput')
X_test_inter = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_inter.h5', 'testOutput')
X_test_high = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_high.h5', 'testOutput')

#X_train = pd.read_hdf('/storage/monande/TrainOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_high.h5', 'trainOutput')

X_data = pd.read_hdf('/storage/monande/significance/DataOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_high.h5', 'dataOutput')

"""
#X_train_type = #X_train.copy()

#X_train_type['Type'] = 'Signal'

X_test_type = X_test_high.copy()

X_test_type['Type'] = 'Signal'

X_test_type_low = X_test_low.copy()

X_test_type_low['Type'] = 'Signal'

X_test_type_inter = X_test_inter.copy()

X_test_type_inter['Type'] = 'Signal'

X_test_low = pd.concat([X_test_low, X_test_type_low], axis=0)
X_test_low.dropna(axis='index', inplace=True)

X_test_inter = pd.concat([X_test_inter, X_test_type_inter], axis=0)
X_test_inter.dropna(axis='index', inplace=True)

#df1.loc[df1['stream'] == 2, 'feat'] = 10


for elm in Diboson:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Diboson'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Diboson'
    
for elm in SingleTop:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'SingleTop'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'SingleTop'

for elm in TopOther:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'TopOther'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'TopOther'

for elm in ttbar:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'ttbar'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'ttbar'

for elm in Higgs:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Higgs'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Higgs'

for elm in Triboson:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Triboson'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Triboson'

for elm in DY:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'DY'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'DY'

for elm in Wjets:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Wjets'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Wjets'

for elm in Zjets:
    #X_train_type.loc[#X_train_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Zjets'
    X_test_type.loc[X_test_type['DatasetNumber'] ==  int(elm), 'Type'] = 'Zjets'



#print(#X_train)
#print(#X_train_type)


#X_train = pd.concat([#X_train, #X_train_type], axis=0)
#X_train.dropna(axis='index', inplace=True)
#print(#X_train)
#print(#X_train['Type'])


X_test_high = pd.concat([X_test_high, X_test_type], axis=0)
X_test_high.dropna(axis='index', inplace=True)

#foo = #X_train.query('ylabel==0 & DatasetNumber==' + str(363355.0)).loc[:,'output']
#print('Skjera???')
#print(#X_train)
#print('wooooop')
#print(foo)
"""

cuts = np.linspace(0,1,41)
mona = 0

N_events_bkg_low = []
N_events_bkg_inter = []
N_events_bkg_high = []


list_level = ['low', 'inter', 'high']
for level in list_level:
    mona = 0
    for cut in cuts[1:]:
        cut_string = 'output <= {:f}'.format(cut)
        print(cut_string)

        
        if level =='low':
            bkg_exp = np.sum(scalefac*X_test_low.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
            N_events_bkg_low.append(bkg_exp-mona)
        elif level =='inter':
            bkg_exp = np.sum(scalefac*X_test_inter.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
            N_events_bkg_inter.append(bkg_exp -mona)
        else:
            bkg_exp = np.sum(scalefac*X_test_high.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
            N_events_bkg_high.append(bkg_exp -mona)

        
        mona = bkg_exp
print('MOOOOONA')
print(N_events_bkg_low)
print(N_events_bkg_inter)
print(N_events_bkg_high)

bkg_samples = {'Wjets':{'color':"lightskyblue"},
               'Zjets':{'color':"dodgerblue"},
               'SingleTop':{'color':"#d6b4fc"},
               'TopOther':{'color':'#a552e6'},
               'ttbar':{'color':'#7e1e9c'},
               'Higgs':{'color':'mediumvioletred'},
               'DY':{'color':'#d767ad'},
               'Diboson':{'color':"lightgreen"},
               'Triboson':{'color':'yellowgreen'}}

#print(X_test)
ML_output = []
ML_weights = []
ML_colors = []
ML_labels = []
ML_errors = []

Sig_output = []
Sig_weights = []
Sig_colors = []
Sig_labels = []
fig_text = []

ML_data = []
ML_data_errors = []
data_errors_temp = []
data_errors_temp_2 = []

var = 'output'
top = -999

stack_order =  ['Data', 'Signal', 'Wjets', 'Triboson', 'DY', 'SingleTop', 'Higgs', 'TopOther', 'ttbar', 'Diboson', 'Zjets']


data_x = []
nmax = 1
nmin = 0
binw = 20

bins = np.linspace(0,1,41)#[nmin + (x*binw) for x in range(int((nmax-nmin)/binw)+1)]
for i in range(len(bins)-1):
    print(bins[i])
    data_x.append(bins[i]+(bins[i+1]-bins[i])/2)
        
for s in stack_order:
    print(s)
    if s == 'Data':
        ML_data, _ = np.histogram(X_data.as_matrix(columns=X_data.columns[X_data.columns.get_loc(var):X_data.columns.get_loc(var)+1]), bins=bins)
        data_errors_temp.append(ML_data)
        #print(len(ML_data))
        #print(data_errors_temp[0])
        temp_list = data_errors_temp[0]
        print(temp_list)
        temp = np.sqrt(temp_list)
        #temp = (1./i for i in temp)
        print(type(temp))
        data_errors_temp_2.append(data_errors_temp[0])
        #print(data_errors_temp_2)
        ML_data_errors = np.ndarray.tolist(temp)
        print(type(ML_data_errors))
        #print(ML_data_errors)
        temp_list_errors = [1/i for i in ML_data_errors]
        print(temp_list_errors)
        print(type(temp_list_errors))
        ML_data_errors = temp_list_errors

    elif s == 'Signal':
        rslt_df = X_test_low.loc[X_test_low['Type'] == 'Signal']
        figuretext = figuretext_low
        fig_text.append(figuretext)
        rslt_df_sig = rslt_df.loc[rslt_df['DatasetNumber'] == float(signal_low)]

        Sig_output.append(rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc(var):rslt_df_sig.columns.get_loc(var)+1]))
        Sig_weights.append(3.34*rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc("eventweight"):rslt_df_sig.columns.get_loc("eventweight")+1]))
            
        del [rslt_df_sig]
        del [rslt_df]

        rslt_df = X_test_inter.loc[X_test_inter['Type'] == 'Signal']
        figuretext = figuretext_inter
        fig_text.append(figuretext)
        rslt_df_sig = rslt_df.loc[rslt_df['DatasetNumber'] == float(signal_inter)]
        Sig_output.append(rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc(var):rslt_df_sig.columns.get_loc(var)+1]))
        Sig_weights.append(3.34*rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc("eventweight"):rslt_df_sig.columns.get_loc("eventweight")+1]))

        del [rslt_df_sig]
        del [rslt_df]

        rslt_df = X_test_high.loc[X_test_high['Type'] == 'Signal']
        figuretext = figuretext_high
        fig_text.append(figuretext)
        rslt_df_sig = rslt_df.loc[rslt_df['DatasetNumber'] == float(signal_high)]
        Sig_output.append(rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc(var):rslt_df_sig.columns.get_loc(var)+1]))
        Sig_weights.append(3.34*rslt_df_sig.as_matrix(columns=rslt_df_sig.columns[rslt_df_sig.columns.get_loc("eventweight"):rslt_df_sig.columns.get_loc("eventweight")+1]))
            
        del [rslt_df_sig]
        del [rslt_df]


    else:
        rslt_df = X_test_high.loc[X_test_high['Type'] == s]
        ML_output.append(rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc(var):rslt_df.columns.get_loc(var)+1]))
        #print(np.shape(ML_output))
        #print(len(rslt_df))
        
        ML_weights.append(3.34*rslt_df.as_matrix(columns=rslt_df.columns[rslt_df.columns.get_loc("eventweight"):rslt_df.columns.get_loc("eventweight")+1]))
        #print('Ml weights')
        #print(ML_weights)
        #print('ml output')
        #print(ML_output)
        ML_colors.append(bkg_samples[s]['color'])
        ML_labels.append(s)
        del [rslt_df]

print(ML_data_errors)
#ML_data_errors = (1./i for i in ML_data_errors)
print(ML_data_errors)

N_events_bkg_low = np.array(N_events_bkg_low)
N_events_bkg_inter = np.array(N_events_bkg_inter)
N_events_bkg_high = np.array(N_events_bkg_high)
N_events_data = np.array(temp_list)

f = open('/storage/monande/plots/ListOfConstants' + '_' + method_type + '_' + level_type + '_' + process_name + '.txt', 'w+')

low_calc = N_events_bkg_low/N_events_data
inter_calc = N_events_bkg_inter/N_events_data
high_calc = N_events_bkg_high/N_events_data

low_mean = np.mean(low_calc)
inter_mean = np.mean(inter_calc)
high_mean = np.mean(high_calc)

f.write('Low\n')
f.write(str(low_calc))
f.write(str(low_mean))
f.write('\nInter\n')
f.write(str(inter_calc))
f.write(str(inter_mean))
f.write('\nHigh\n')
f.write(str(high_calc))
f.write(str(high_mean))

f.close()

fig = plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k') 
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
ax0 = plt.subplot(gs[0])

#ax1.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
ax0.hist(ML_output,bins=bins,weights=ML_weights,stacked=True,color=ML_colors, label=ML_labels); #weights=mc_weights,
ax0.errorbar( x=data_x, y=ML_data, yerr=ML_data_errors, fmt='ko', label='Data')
ax0.hist(Sig_output,bins=bins, weights= Sig_weights, histtype = 'step', color = ['lime', 'magenta', 'cyan'], label = fig_text)
ax0.set_yscale('log')
ax0.set_ylabel(r'Events', fontsize = 14)
plt.tick_params(axis='both', which='major', labelsize=14)

ax1 = plt.subplot(gs[1])
ax1.set_xlabel(xLabel, fontsize = 14)
ax1.axhline(y=1, color = 'k', linestyle = '-')
#plt.xlim(0,1)
plt.tick_params(axis='both', which='major', labelsize=14)

#ax1.rc('xtick', labelsize=14)
#ax0.rc('ytick', labelsize=14)
#ax1.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
ax0.set_ylim([0,10e9])

ax1.errorbar(x = data_x, y=N_events_data/N_events_bkg_high,  yerr = ML_data_errors, fmt = 'ko')
ax1.set_ylim([0,2])
ax1.set_ylabel('Data/MC', fontsize = 14)

for ax in fig.get_axes():
    ax.label_outer()
ax = plt.gca()
#plt.text(0.45,0.97,r'$\mathbf{{ATLAS}}$ Open Data',ha="left",va="top",family='sans-serif',transform=ax.transAxes,fontsize=13)
#plt.text(0.45,0.92,'for education only',ha="left",va="top",family='sans-serif',transform=ax.transAxes,style='italic',fontsize=8)
#plt.text(0.25,0.90,r'$\sqrt{s}=13\,\mathrm{TeV},\;\int L\,dt=58.5\,\mathrm{fb}^{-1}$',ha="left",va="top",family='sans-serif',transform=ax.transAxes)


ax0.legend(ncol = 2, loc = 1)

plt.savefig('/storage/monande/plots/StackedPlots/stackedplot' + '_' + method_type + '_' + level_type + '_' + process_name + '.pdf')
#plt.show()












"""


Traceback (most recent call last):
  File "main.py", line 1887, in <module>
    main()
  File "main.py", line 786, in main
    output_train = model.predict_proba(X_train_scaled, ntree_limit=model.best_iteration)
AttributeError: 'KerasClassifier' object has no attribute 'best_iteration'
Closing remaining open files:/home/monaa/scratch2/hdf5Files/sig_slepslep_high_preprocessed.h5...done

"""
