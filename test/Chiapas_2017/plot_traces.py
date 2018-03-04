# Ext modules
import numpy as np
import os,sys,shutil
import time
import h5py
from glob import glob
from scipy import signal

# Int modules
import sacpy
import cmt
#from Sampler import simpleSampler

from Arguments import *

assert os.path.exists(o_cmt_file), 'INPUT CMTSOLUTION %s not found'%(o_cmt_file)
    
# Prepare data (data is should already be filtered)
cmtp = cmt.cmtproblem()
cmtp.preparedata(i_sac_lst)
npts = []
for chan_id in cmtp.chan_ids:
    npts.append(cmtp.data[chan_id].npts)
npts = np.array(npts)
dobs = cmtp.D

# Build Green GF_names dictionary for each sub-source
GF_names = []
s = sacpy.sac()
GF_names = {}
for dkey in cmtp.data:
    GF_names[dkey] = {}
    data = cmtp.data[dkey]
    sac_file = 'GF.%s.%s.LH%s.SAC.filter.conv.sac'%(data.kstnm,data.knetwk,data.kcmpnm[-1])
    for j in range(6):
        MTnm = cmtp.cmt.MTnm[j]
        dir_name = os.path.join(GF_DIR,'%s'%(MTnm.upper()))
        GF_names[dkey][MTnm] = os.path.join(dir_name,sac_file)

# Compute Greens
cmtp.cmt.rcmtfile(o_cmt_file,scale=1./GF_M0)
cmtp.preparekernels(GF_names,delay=cmtp.cmt.ts)

# Predict
cmtp.calcsynt()

# Save traces
if os.path.exists(o_trace_dir):
    shutil.rmtree(o_trace_dir)
os.mkdir(o_trace_dir)
for chan_id in cmtp.chan_ids:    
    cmtp.synt[chan_id].write(os.path.join(o_trace_dir,cmtp.synt[chan_id].id+'.sac'))

# Station locations
staloc = []
for chan_id in cmtp.chan_ids:
    sacdata = cmtp.data[chan_id].copy()
    staloc.append([sacdata.stla,sacdata.stlo,sacdata.az,sacdata.dist])
staloc = np.array(staloc)

# Traces
cmtp.setTimeWindow(wpwin=wpwin,swwin=swwin,dcwin=dcwin)
cmtp.traces(length=1400,show_win=True,variable_xlim=True,staloc=staloc,rasterize=False,yfactor=1.2,ofile='traces.pdf')
