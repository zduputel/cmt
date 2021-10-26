# Ext modules
import numpy as np
import os
import time
import h5py
from glob import glob
from scipy import signal

# Int modules
import sacpy
import cmt
#from Sampler import simpleSampler

from Arguments import *

# Prepare data (data is should already be filtered)
cmtp = cmt.cmtproblem()
cmtp.preparedata(i_sac_lst,wpwin=wpwin,swwin=swwin,dcwin=dcwin)
cmtp.buildD()
npts = []
for chan_id in cmtp.chan_ids:
    npts.append(cmtp.data[chan_id].npts)
npts = np.array(npts)
dobs = cmtp.D

# Build Green GF_names dictionary for each sub-source
GF_names = []
s = sacpy.Sac()
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
cmtp.preparekernels(GF_names,delay=time_shift)
cmtp.buildG()
cmtp.cmt.rcmtfile(i_cmt_file)
cmtp.cmt.ts = time_shift

# Invert
cmtp.cmtinv()
cmtp.cmt.wcmtfile(o_cmt_file,scale=GF_M0)

# Predict
cmtp.calcsynt()

