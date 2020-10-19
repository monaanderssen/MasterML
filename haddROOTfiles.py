import sys,os

rootdir = "/storage/eirikgr/NEWnTuples_EIRIK/"

years = ["18","17","1516"]

#for path, subdirs, files in os.walk(rootdir):
subdirs = [x[0] for x in os.walk(rootdir)]
for p in subdirs:
    if "Data" in p: continue
    if "SUSY" in p and ("18" in p):# or "1516" in p or "17" in p):
        if "18" in p: newdir = p.replace("18","")
        elif "17" in p: newdir = p.replace("17","")
        elif "1516" in p: newdir = p.replace("1516","")
        print(newdir)
        #sys.exit()
        if not os.path.isdir(newdir):
            print("%s does not exist"%(newdir))
            try:
                os.mkdir(newdir)
            except OSError:
                print ("Creation of the directory %s failed" % (newdir))
            else:
                print ("Successfully created the directory %s " % (newdir))
        print(p)
        for path, subdirs, files in os.walk(p):
            for f in files:
                if f.endswith("merged_processed.root"):
                    fullname = os.path.join(rootdir,p+"/"+f)
                    haddstr = "hadd "+newdir+"/"+f+" "
                    nhadd = 0
                    for y in years:
                        if os.path.isfile(fullname.replace("18",y)):
                            haddstr += fullname.replace("18",y)+" "
                            nhadd += 1
                    if not nhadd == 3 and nhadd > 0:
                        print("Could not find all years for %s"%f)
                    elif nhadd > 0:
                        #print(haddstr)
                        os.system(haddstr)
   # for name in files:
   #     print os.path.join(path, name)
