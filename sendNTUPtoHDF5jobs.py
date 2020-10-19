import pty
import shlex
import os
import time
import subprocess
import sys
import shutil
from ROOT import TChain

doRun  = ["triboson","lowMassDY","higgs","topOther"] #,"singleTop","data18","data15-16","data17"
doRun += ["diboson","Zjets","ttbar","Wjets","singleTop"]
doRun += ["data15-16","data17","data18"]

doRun = []

runSig = ["slepslep","slepsnu","WW","monoZ"]
runMassSplit = ["high","inter","low"]

#doRun = ["higgs"]

screenSessions = []
for key in doRun:
    cmdString = "screen -dmS dummy%sxc screen -S %sxc python main.py --prepare_hdf5 --%s %s --chunksize 1e5" %(key,key,"data" if "data" in key else "bkg",key)
    print(cmdString)
    #continue
    screenSessions.append("%sxc"%(key))
    (master, slave) = pty.openpty()
    cmdArgs = shlex.split(cmdString)
    p = subprocess.Popen(cmdArgs, close_fds = False, shell=False,
                         stdin=slave, stdout=slave, stderr=slave)
    
for key in runSig:
    for ms in runMassSplit:
        cmdString = "screen -dmS dummy%s%sxc screen -S %s%sxc python main.py --prepare_hdf5 --sig --%s --%s --chunksize 1e5" %(key,ms,key,ms,key,ms)
        print(cmdString)
        #continue
        screenSessions.append("%sxc"%(key))
        (master, slave) = pty.openpty()
        cmdArgs = shlex.split(cmdString)
        p = subprocess.Popen(cmdArgs, close_fds = False, shell=False,
                             stdin=slave, stdout=slave, stderr=slave)
print("Followig %i sessions were started:" %len(screenSessions))
for ss in screenSessions:
    print(ss)

