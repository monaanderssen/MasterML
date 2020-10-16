#import os
#import sys


#from ROOT import gROOT, gPad, gStyle, gSystem
#from ROOT import TCanvas, TFile, THStack, TH1F, TLegend, TChain, TH1, TH2, TAxis, TPad, TRatioPlot 
#from ROOT import kGreen, kYellow, kRed, kBlue, kOrange, kMagenta, kBlack, kFullCircle 
#from ROOT.TAxis import kCenterLabels 
#from ROOT import RooFit, RooStats
#from ROOT import TH2D
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
#from matplotlib.ticker import NullFormatter, FixedLocator
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


#import operator   # Used to sort the backgrounds after size
#import plotconfig as pc

# Set to true to hide canvases drawing when code is running
#gROOT.SetBatch(True)
#Same command?
#ROOT.gROOT.SetBatch(1)

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
    xAxis = 10
    yAxis = 10
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
    xAxis = 10
    yAxis = 10
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
    xAxis = 10
    yAxis = 10
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
    xAxis = 11
    yAxis = 10
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
"""
if args.low:
    sig_type = 'low'
elif args.inter:
    sig_type = 'inter'
elif args.high:
    sig_type = 'high'
"""




def getBestCut(X_test, scalefac, signalSampleNumber):
    cuts = np.linspace(0,1,1001)
    best_Z_N_exp = -10
    for cut in cuts[:-1]:
        cut_string = 'output > {:f}'.format(cut)
        #print(cut_string)

        bkg_exp = np.sum(scalefac*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(scalefac*X_test.query("ylabel == 1 & DatasetNumber==" + signalSampleNumber + "& "+cut_string).loc[:,"eventweight"])
        #print(bkg_exp)
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)
        #Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)
        if Z_N_exp == inf:
            Z_N_exp = -10
        if Z_N_exp > best_Z_N_exp and bkg_exp > 3:
            best_Z_N_exp = Z_N_exp
            best_cut = cut
            #print(best_Z_N_exp)
            #print(best_Z_N_exp, best_cut)
    return best_Z_N_exp, best_cut



def calculateSignificances(X_test,scalefac, best_cut, signalDict):
    Z_N_exp_list = []
    #best_cut_list = []
    #datasetnumber_list = []
    mass_1 = []
    mass_2 = []
    delta_m = []
    #print(signalDict)
    for i in signalDict.keys():
        #best_Z_N_exp = -10
        #print(signalDict[i])
        j = i.split(',')
        mass_1.append(j[0])
        mass_2.append(j[1])
        delta_m.append(j[2])

        cut_string = 'output > {:f}'.format(best_cut)
        #print(cut_string)
        
        bkg_exp = np.sum(scalefac*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(scalefac*X_test.query("ylabel == 1 & DatasetNumber==" + signalDict[i] + "& "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)

        #print(bkg_exp)
        #print(sig_exp)
        #print(Z_N_exp)

        if Z_N_exp == inf:
            Z_N_exp = 0
        Z_N_exp_list.append(Z_N_exp)
        #Z_N_exp_list[Z_N_exp_list == inf] = 0
        #best_cut_list.append(best_cut)
        #datasetnumber_list.append(slepslep_high[i])
    return mass_1, mass_2, delta_m, Z_N_exp_list


def plotSignificanceWithZoom(mass_1, mass_2, sign, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type):
    fig, ax = plt.subplots()

    #print(mass_1)
    #print(mass_2)
    print(sign) 
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
    #ax.clim(0,3.5)
    
    #n = [ '%.2f' % elem for elem in Z_N_exp_list]
    n = [ '%.2f' % elem for elem in sign]

    for i, txt in enumerate(n):
        ax.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
        
        if Z_N_exp > best_Z_N_exp:
            best_Z_N_exp = Z_N_exp
            best_cut = cut
            print(best_Z_N_exp)
            #print(best_Z_N_exp, best_cut)
    return best_Z_N_exp, best_cut



def calculateSignificances(X_test,scalefac, best_cut, signalDict):
    Z_N_exp_list = []
    #best_cut_list = []
    #datasetnumber_list = []
    mass_1 = []
    mass_2 = []
    delta_m = []
    #print(signalDict)
    for i in signalDict.keys():
        #best_Z_N_exp = -10
        #print(signalDict[i])
        j = i.split(',')
        mass_1.append(j[0])
        mass_2.append(j[1])
        delta_m.append(j[2])

        cut_string = 'output > {:f}'.format(best_cut)
        #print(cut_string)
        
        bkg_exp = np.sum(scalefac*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(scalefac*X_test.query("ylabel == 1 & DatasetNumber==" + signalDict[i] + "& "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.2)

        #print(bkg_exp)
        #print(sig_exp)
        #print(Z_N_exp)

        if Z_N_exp == inf:
            Z_N_exp = 0
        Z_N_exp_list.append(Z_N_exp)
        #Z_N_exp_list[Z_N_exp_list == inf] = 0
        #best_cut_list.append(best_cut)
        #datasetnumber_list.append(slepslep_high[i])
    return mass_1, mass_2, delta_m, Z_N_exp_list


def plotSignificanceWithZoom(mass_1, mass_2, sign, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type):
    fig, ax = plt.subplots()

    #print(mass_1)
    #print(mass_2)
    print(sign) 
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
    #ax.clim(0,3.5)
    
    #n = [ '%.2f' % elem for elem in Z_N_exp_list]
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
    
    #axins.imshow(Z2, extent=extent, interpolation="nearest",
             #origin="lower")

    # sub region of the original image


    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=3, loc2=3, fc="none", ec="0.5")

    plt.draw()
    plt.tight_layout()
    plt.savefig('significanceplots/significance_' + method_type + '_' + process_name + '_' + level_type  + '.pdf')

    
    #plt.show()

def plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type):
    fig, ax = plt.subplots()
    #cm = plt.cm.get_cmap('plasma')
    #sc = ax.scatter(mass_1, mass_2, c=Z_N_exp_list, s=100,marker = 's', cmap=cm)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #plt.xticks(rotation=90)
    #plt.locator_params(axis='x', nbins=5)
    #ax.xaxis.set_major_locator(plt.MaxNLocator(xAxis))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(yAxis))
    #x = [100, 750]
    #y = [100, 750]
    #plt.plot(x,y, 'k-')
    #plt.plot()
    #ax.set_yscale('log')
    #ax.set_xscale([0,800])
    #plt.axes('tight')
    #ax.yaxis.set_minor_formatter(NullFormatter())
    #ax.yaxis.set_major_locater(FixedLocater(np.arrange(0,800,100)))
    #plt.axis([[0,200],[200,800],[0,100],[100,800]])
    print(Z_N_exp_list)
    #print(mass_2)

    #linearAxis = plt.gca()

    cm = plt.cm.get_cmap('brg')
    sc = ax.scatter(mass_1, mass_2, c=Z_N_exp_list, s=100,marker = 's', cmap=cm, vmin = 0, vmax = 3.5)

    #linearAxis.plot(x, y)
    #linearAxis.set_ylim((200, 850))
    #divider = make_axes_locatable(linearAxis)
    #logAxis = divider.append_axes("bottom", size=1, pad=0, sharex=linearAxis)
    #logAxis.plot(mass_1, mass_2)
    #logAxis.set_yscale('linear')
    #logAxis.set_ylim((0, 200));
    #ticks = [0, 100, 200, 300, 350, 400, 500, 600, 700, 805]

    #plt.xticks(range(10), ticks)

    plt.xlabel(xLabel + '[GeV]')
    plt.ylabel(yLabel + '[GeV]')
    #plt.title(figureTitle)
    #plt.MaxNLocator(3)
    cb = plt.colorbar(sc)
    cb.set_label('Significance')
    
    n = [ '%.2f' % elem for elem in Z_N_exp_list]
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')
    
    plt.savefig('significanceplots/significance_' + method_type + '_' + process_name + '_' + level_type + '.pdf')
    #plt.show()



############################
#    SlepSnu               #
############################

#Low:

X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_low.h5')
best_Z_N_exp, best_cut_low = getBestCut(X_test, scalefac, signal_low)
print('Best cut for low mass splittings is %.4f' %best_cut_low)
#signalDict = slepsnu_high
mass_1_low, mass_2_low, delta_m_low, Z_N_exp_list_low = calculateSignificances(X_test, scalefac, best_cut_low, signalDict_low)
#print(mass_1_low)
#Inter:
X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_inter.h5')
best_Z_N_exp, best_cut_inter = getBestCut(X_test, scalefac, signal_inter)
print('Best cut for intermediate mass splittings is %.4f' %best_cut_inter)
#signalDict = slepsnu_high
mass_1_inter, mass_2_inter, delta_m_inter, Z_N_exp_list_inter = calculateSignificances(X_test, scalefac, best_cut_inter, signalDict_inter)
#print(mass_1_inter)

#High:
X_test = pd.read_hdf('/storage/monande/significance/TestOutputFromML_' + method_type + '_' + process_name + '_' + level_type + '_high.h5')
best_Z_N_exp, best_cut_high = getBestCut(X_test, scalefac, signal_high)
print('Best cut for high mass splittings is %.4f' %best_cut_high)

#signalDict = slepsnu_high
mass_1_high, mass_2_high, delta_m_high, Z_N_exp_list_high = calculateSignificances(X_test, scalefac, best_cut_high, signalDict_high)
#print(mass_1_high)
#plotting:

#delta_m = abs(np.subtract(mass_1, mass_2))
#mass_2 = delta_m
#mass_1 = delta_m

#print(mass_1)

sig_type = 'all'
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

















    
    """
    mass_1 = mass_1_low + mass_1_inter + mass_1_high
    mass_2 = mass_2_low + mass_2_inter + mass_2_high
    delta_m = delta_m_low + delta_m_inter + delta_m_high
    Z_N_exp_list = Z_N_exp_list_low + Z_N_exp_list_inter + Z_N_exp_list_high 

    mass_1 = np.array([float(elem) for elem in mass_1])
    mass_2 = np.array([float(elem) for elem in mass_2])
    delta_m = np.array([float(elem) for elem in delta_m])
    Z_N_exp_list = [float(elem) for elem in Z_N_exp_list]
    figureTitle = r'%.2f for high, %.2f for intermediate and %.2f for low $\Delta$m' %(best_cut_high, best_cut_inter, best_cut_low)
    mass_x_low = []
    mass_y_low = []
    mass_x_high = []
    mass_y_high = []
    Z_N_low = []
    Z_N_high = []
    for i, j, k in zip(mass_1, mass_2, Z_N_exp_list):
        #print(i)
        if i < 300:
            mass_x_low.append(i)
            mass_y_low.append(j)
            Z_N_low.append(k)
        elif i>= 300:
            mass_x_high.append(i)
            mass_y_high.append(j)
            Z_N_high.append(k)

    sig_type = 'low'
    mass_1 = np.array(mass_x_low)
    mass_2 = np.array(mass_y_low)
    Z_N_exp_list = np.array(Z_N_low)
    

    figureTitle = r'%.2f for low $\Delta$m' %(best_cut_low)
    plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)


    sig_type = 'high'
    mass_1 = np.array(mass_x_high)
    mass_2 = np.array(mass_y_high)
    Z_N_exp_list = np.array(Z_N_high)
    

    figureTitle = r'%.2f for high and %.2f for intermediate $\Delta$m' %(best_cut_high, best_cut_inter)
    plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)
    """
    """
    sig_type = 'low'
    mass_1 = mass_1_low
    mass_2 = mass_2_low
    delta_m = delta_m_low
    Z_N_exp_list = Z_N_exp_list_low

    mass_1 = np.array([float(elem) for elem in mass_1])
    mass_2 = np.array([float(elem) for elem in mass_2])
    delta_m = np.array([float(elem) for elem in delta_m])
    Z_N_exp_list = [float(elem) for elem in Z_N_exp_list]
    figureTitle = r'%.2f for low $\Delta$m' %(best_cut_low)
    plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)

    sig_type = 'interAndHigh'
    mass_1 = mass_1_inter + mass_1_high
    mass_2 = mass_2_inter + mass_2_high
    delta_m = delta_m_inter + delta_m_high
    Z_N_exp_list = Z_N_exp_list_inter + Z_N_exp_list_high 

    mass_1 = np.array([float(elem) for elem in mass_1])
    mass_2 = np.array([float(elem) for elem in mass_2])
    delta_m = np.array([float(elem) for elem in delta_m])
    Z_N_exp_list = [float(elem) for elem in Z_N_exp_list]
    figureTitle = r'%.2f for high and %.2f for intermediate $\Delta$m' %(best_cut_high, best_cut_inter)
    plotSignificance(mass_1, mass_2, Z_N_exp_list, process_name, level_type, xLabel, yLabel, figureTitle, xAxis, yAxis, sig_type)

    """













"""
X_test = pd.read_hdf('/home/monaa/scratch2/significance/outputFromML_' + method_type + '_' + process_name + '_' + level_type + '_' + sig_type + '.h5')

print(X_test.columns)

#pd.read('outputFromML_monoZ_high_level_high.h5', 'testOutput')

cuts = np.linspace(0,1,21)
#print('X=', x)

cut = 0.05

bkg_exp = np.zeros(len(cuts)-1)
sig_exp = np.zeros(len(cuts)-1)
Z_N_exp_list = []#np.zeros(len(cuts)-1)
#bkg_exp = np.zeros(len(cuts)-1)
best_cut_list = []
datasetnumber_list = []
slep_mass = []
neutralino_mass = []

dictonary = process_name + '_' + sig_type
print(dictonary)

for i in slepslep_high.keys():
    best_Z_N_exp = -10
    print(slepslep_high[i])
    j = i.split(',')
    slep_mass.append(j[0])
    neutralino_mass.append(j[1])

    for cut in cuts[:-1]:
        cut_string = 'output > {:f}'.format(cut)
        #print(cut_string)

        bkg_exp = np.sum(3.34*X_test.query("ylabel == 0 & "+cut_string).loc[:,"eventweight"])
        sig_exp = np.sum(3.34*X_test.query("ylabel == 1 & DatasetNumber==" + slepslep_high[i] + "& "+cut_string).loc[:,"eventweight"])
        Z_N_exp = RooStats.NumberCountingUtils.BinomialExpZ(sig_exp, bkg_exp, 0.3)
        if Z_N_exp > best_Z_N_exp:
            best_Z_N_exp = Z_N_exp
            best_cut = cut
            #print(best_Z_N_exp)
    Z_N_exp_list.append(best_Z_N_exp) 
    best_cut_list.append(best_cut)
    datasetnumber_list.append(slepslep_high[i])
    #print(bkg_exp, sig_exp, Z_N_exp)

    print('The best expected significance is %f and is done by a cut on %.2f' %(best_Z_N_exp, best_cut))

#print(Z_N_exp_list)

best_cut_list = [ '%.2f' % elem for elem in best_cut_list ]


print(Z_N_exp_list)
print(best_cut_list)
print(datasetnumber_list)



fig, ax = plt.subplots()

# instanciate a figure and ax object
# annotate is a method that belongs to axes
#ax.plot(x, y, 'ro',markersize=23)

## controls the extent of the plot.


cm = plt.cm.get_cmap('plasma')
#xy = range(20)
#z = xy
sc = ax.scatter(slep_mass, neutralino_mass, c=Z_N_exp_list, s=500,marker = 's', cmap=cm)
plt.colorbar(sc)
#plt.show()

z = slep_mass
y = neutralino_mass
n = [ '%.2f' % elem for elem in Z_N_exp_list]
#xy = zip(x, y)
offset = 1.0 
#ax.set_xlim(min(x)-offset, max(x)+ offset)
#ax.set_ylim(min(y)-offset, max(y)+ offset)

# loop through each x,y pair
for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]),color = 'white', 
                fontsize="small", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')

#plt.contourf(slep_mass, neutralino_mass, Z_N_exp_list, 's')
plt.savefig('significance.pdf')
plt.show()

"""
