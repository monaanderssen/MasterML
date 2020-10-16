import uproot
import pandas as pd
import numpy as np
import argparse

#from samples import *
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



if args.slepslep:
    xLabel = r'm($\~l$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'slepslep'
    filename = 'direct_slepton_v1.6_significances.dat'

elif args.slepsnu:
    xLabel = r'm($\~\chi_1^{\pm}$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'slepsnu'
    filename = 'C1C1_SlepSnu_v1.6_significances.dat'

elif args.WW:
    xLabel = r'm($\~\chi_1^{\pm}$)'
    yLabel = r'm($\~\chi_1^{0}$)'
    process_name = 'WW'
    filename = 'C1C1_WW_v1.6_significances.dat'

elif args.monoZ:
    xLabel = r'm$_{V}$'
    yLabel = r'm$_{\chi}$'
    process_name = 'monoZ'
    filename = 'mono_Z_v1.6_significances.dat'




def plotSignificanceWithZoom(mass_1, mass_2, sign, process_name, xLabel, yLabel):
    """ Plot significance for direct slepton and mono-Z """
    fig, ax = plt.subplots()

    cm = plt.cm.get_cmap('brg')
    sc = ax.scatter(mass_1, mass_2, c=sign, s=100,marker = 's', cmap=cm, vmin = 0, vmax = 3.5)

    plt.xlabel(xLabel + '[GeV]')
    plt.ylabel(yLabel + '[GeV]')

    cb = plt.colorbar(sc)
    cb.set_label('Significance')
    if args.slepslep:
    	axins = inset_axes(ax, height = 1.5, width = '53%', loc=2) 
    	x1, x2, y1, y2 = 75, 210, -10, 170
    elif args.monoZ:
    	axins = inset_axes(ax, height = 1.5, width = '40%', loc=2) 
    	x1, x2, y1, y2 = -10, 170, -10, 60

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.scatter(mass_1, mass_2, c=sign, s=100,marker = 's', cmap=cm, vmin = 0, vmax = 3.5)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    
    for i, txt in enumerate(sign):
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
    plt.savefig('significanceplots/significanceCutandCount_' + process_name + '.pdf')
    
    plt.show()

def plotSignificance(mass_1, mass_2, sign, process_name, xLabel, yLabel):
    """ Plot signficance for chargino production """
    fig, ax = plt.subplots()

    cm = plt.cm.get_cmap('plasma')
    sc = ax.scatter(mass_1, mass_2, c=sign, s=100,marker = 's', cmap=cm)

    plt.xlabel(xLabel + '[GeV]')
    plt.ylabel(yLabel + '[GeV]')

    cb = plt.colorbar(sc)
    cb.set_label('Significance')
    
    for i, txt in enumerate(sign):
        ax.annotate(txt, (mass_1[i], mass_2[i]),color = 'white', 
                    fontsize=4, weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('significanceplots/significanceCutandCount_' + process_name + '.pdf')
    
    plt.show()


mass_1 = []
mass_2 = []

sign = []

infile = open(filename, 'r')
next(infile)
if not args.monoZ:
	for lines in infile:
		#lines.readlines() 
		line = lines.split(' ')
		sign.append(float(line[6]))
		a = line[0].split('_')
		b = a[2].split('p0')
		c = a[3].split('p0')
		mass_1.append(float(b[0]))
		mass_2.append(float(c[0]))


elif args.monoZ:
	for lines in infile:
		#lines.readlines() 
		line = lines.split(' ')
		sign.append(float(line[6]))
		a = line[0].split('_')
		b = a[2].split('DM')
		c = a[3].split('MM')
		mass_2.append(float(b[1]))
		mass_1.append(float(c[1]))
	
#print(mass_1, mass_2, sign)

if args.slepsnu or args.WW:
	plotSignificance(mass_1, mass_2, sign, process_name, xLabel, yLabel)

elif args.slepslep or args.monoZ:
	plotSignificanceWithZoom(mass_1, mass_2, sign, process_name, xLabel, yLabel)




