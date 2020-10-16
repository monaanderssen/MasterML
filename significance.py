from ROOT import TLorentzVector, RooStats
import uproot
import pandas as pd
import numpy as np
import argparse
from numpy import inf

from samples import *
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



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
    scalefac = 3.33#(139/58.4501)
elif args.nn:
    method_type = 'NN'
    scalefac = 3.33#*(139/58.4501)

if args.slepslep:
    xLabel = r'm($\~l$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'slepslep'
    signal_low = '395984'
    signalDict_low = slepslep_low
    signal_inter = '396014'
    signalDict_inter = slepslep_inter
    signal_high = '396033'
    signalDict_high = slepslep_high
elif args.slepsnu:
    xLabel = r'm($\~\chi_1^{\pm}$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'slepsnu'
    signal_low = '397115'
    signalDict_low = slepsnu_low
    signal_inter = '397150'
    signalDict_inter = slepsnu_inter
    signal_high = '397169'
    signalDict_high = slepsnu_high
elif args.WW:
    xLabel = r'm($\~\chi_1^{\pm}$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'WW'
    signal_low = '395268'
    signalDict_low = WW_low
    signal_inter = '395320'
    signalDict_inter = WW_inter
    signal_high = '395330'
    signalDict_high = WW_high
elif args.monoZ:
    xLabel = r'm$_{V}$'
    yLabel = r'm$_{\chi}$'
    process_name = 'monoZ'
    signal_low = '310604'
    signalDict_low = MonoZ_low
    signal_inter = '310613'
    signalDict_inter = MonoZ_inter
    signal_high = '310617'
    signalDict_high = MonoZ_high
if args.high_level:
    level_type = 'High_level'
elif args.low_level:
    level_type = 'Low_level'
else:
    level_type = 'All_level'




def getBestCut(X_test, scalefac, signalSampleNumber):
    """ Gettting the best bin for the benchmark signal for each process """
    cuts = np.linspace(0,1,1001)
    best_Z_N_exp = -10
    for cut in cuts[:-1]:
        cut_string = 'output > {:f}'.format(cut)
        bkg_exp = np.sum(scalefac*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(scalefac*X_test.query("ylabel == 1 & DatasetNumber==" + signalSampleNumber + "& "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)
        if Z_N_exp == inf:
            Z_N_exp = -10
        if Z_N_exp > best_Z_N_exp and bkg_exp > 3:
            best_Z_N_exp = Z_N_exp
            best_cut = cut
    return best_Z_N_exp, best_cut



def calculateSignificances(X_test,scalefac, best_cut, signalDict):
    """ Calculating the significance for all signal samples """
    Z_N_exp_list = []
    mass_1 = []
    mass_2 = []
    delta_m = []
    for i in signalDict.keys():
        j = i.split(',')
        mass_1.append(j[0])
        mass_2.append(j[1])
        delta_m.append(j[2])

        cut_string = 'output > {:f}'.format(best_cut)
        bkg_exp = np.sum(scalefac*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(scalefac*X_test.query("ylabel == 1 & DatasetNumber==" + signalDict[i] + "& "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)

        if Z_N_exp == inf:
            Z_N_exp = 0
        Z_N_exp_list.append(Z_N_exp)
    return mass_1, mass_2, delta_m, Z_N_exp_list


def plotSignificanceWithZoom(mass_1, mass_2, sign, process_name, level_type, xLabel, yLabel, figureTitle):
    """ Plotting the significance for direct slepton production and mono-Z """
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('brg')
    sc = ax.scatter(mass_1, mass_2, c=sign, s=100,marker = 's', cmap=cm, vmin=0, vmax=3.5)

    plt.xlabel(xLabel + '[GeV]')
    plt.ylabel(yLabel + '[GeV]')

    cb = plt.colorbar(sc)
    cb.set_label('Significance')
    if args.slepslep:
    	axins = inset_axes(ax, height = 1.5, width = '53%', loc=2) # zoom = 6
    	x1, x2, y1, y2 = 75, 210, -10, 170
    elif args.monoZ:
    	axins = inset_axes(ax, height = 1.5, width = '40%', loc=2) # zoom = 6
    	x1, x2, y1, y2 = -10, 170, -10, 60

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.scatter(mass_1, mass_2, c=sign, s=100,marker = 's', cmap=cm, vmin= 0, vmax = 3.5)
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    n = [ '%.2f' % elem for elem in sign]

    for i, txt in enumerate(n):
        ax.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
        axins.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
    
    mark_inset(ax, axins, loc1=3, loc2=3, fc="none", ec="0.5")

    plt.draw()
    plt.tight_layout()
    plt.savefig('significanceplots/significance_' + method_type + '_' + process_name + '_' + level_type  + '.pdf')

def plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle):
    """ Plotting the significance for chargino production """
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('brg')
    sc = ax.scatter(mass_1, mass_2, c=Z_N_exp_list, s=100,marker = 's', cmap=cm, vmin = 0, vmax = 3.5)

    plt.xlabel(xLabel + '[GeV]')
    plt.ylabel(yLabel + '[GeV]')
    cb = plt.colorbar(sc)
    cb.set_label('Significance')
    
    n = [ '%.2f' % elem for elem in Z_N_exp_list]
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
    
    plt.savefig('significanceplots/significance_' + method_type + '_' + process_name + '_' + level_type + '.pdf')


#Low:
X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_low.h5')
best_Z_N_exp, best_cut_low = getBestCut(X_test, scalefac, signal_low)
print('Best cut for low mass splittings is %.4f' %best_cut_low)
mass_1_low, mass_2_low, delta_m_low, Z_N_exp_list_low = calculateSignificances(X_test, scalefac, best_cut_low, signalDict_low)

#Inter:
X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_inter.h5')
best_Z_N_exp, best_cut_inter = getBestCut(X_test, scalefac, signal_inter)
print('Best cut for intermediate mass splittings is %.4f' %best_cut_inter)
mass_1_inter, mass_2_inter, delta_m_inter, Z_N_exp_list_inter = calculateSignificances(X_test, scalefac, best_cut_inter, signalDict_inter)

#High:
X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_high.h5')
best_Z_N_exp, best_cut_high = getBestCut(X_test, scalefac, signal_high)
print('Best cut for high mass splittings is %.4f' %best_cut_high)

mass_1_high, mass_2_high, delta_m_high, Z_N_exp_list_high = calculateSignificances(X_test, scalefac, best_cut_high, signalDict_high)

mass_1 = mass_1_low + mass_1_inter + mass_1_high
mass_2 = mass_2_low + mass_2_inter + mass_2_high
delta_m = delta_m_low + delta_m_inter + delta_m_high
Z_N_exp_list = Z_N_exp_list_low + Z_N_exp_list_inter + Z_N_exp_list_high 

mass_1 = np.array([float(elem) for elem in mass_1])
mass_2 = np.array([float(elem) for elem in mass_2])
delta_m = np.array([float(elem) for elem in delta_m])
Z_N_exp_list = [float(elem) for elem in Z_N_exp_list]
figureTitle = r'%.2f for high, %.2f for intermediate and %.2f for low $\Delta$m' %(best_cut_high, best_cut_inter, best_cut_low)


if args.slepsnu or args.WW:
    plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)

elif args.slepslep or args.monoZ:
    plotSignificanceWithZoom(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)



