'''
Simple class for CMT inversion problems

Written by Z. Duputel and L. Rivera, May 2015
'''

# Externals
from scipy import signal
from scipy import linalg
from copy  import deepcopy
from multiprocessing import Pool, cpu_count
import numpy as np
import os,sys

# Personals
from sacpy  import sac
from .cmt   import cmt
from .force import force

TRACES_PLOTPARAMS = {'backend': 'pdf', 'axes.labelsize': 10,
                     'pdf.fonttype': 42, 'ps.fonttype':42,
                     'font.size': 10,
                     'xtick.labelsize': 10,
                     'ytick.labelsize': 10,
                     'legend.fontsize': 10,
                     'lines.markersize': 6,
                     'font.size': 10,
                     'savefig.dpi': 200,
                     'keymap.all_axes': 'a',
                     'keymap.back': ['left', 'c', 'backspace'],
                     'keymap.forward': ['right', 'v'],
                     'keymap.fullscreen': 'f',
                     'keymap.grid': 'g',
                     'keymap.home': ['h', 'r', 'home'],
                     'keymap.pan': 'p',
                     'keymap.save': 's',
                     'keymap.xscale': ['k', 'L'],
                     'keymap.yscale': 'l',
                     'keymap.zoom': 'o',                  
                     'path.snap': True,
                     'savefig.format': 'pdf',
                     'pdf.compression': 9,
                     'figure.figsize': [11.69,8.270]}

def nextpow2(i): 
    n = 2 
    while n < i: 
        n = n * 2 
    return n

def Pplus(h,tvec,stfdur):
    stf = h.copy()
    for i in range(len(tvec)):
        #if tvec[i]<0.0 or tvec[i]>stfdur or h[i]<0.0:
        if tvec[i]>stfdur or h[i]<0.0:
            stf[i]=0.0
    return stf

def conv_by_sin_stf(sac_in,delay,half_width):
    '''
    Convolve by a Sinusoid STF with 1 negative and 1 positive lobes (could be appropriate for a single force)
    '''
    # Copy sac
    sac_out = sac_in.copy()

    # Build half-sinusoidal STF
    lh = int(np.floor(half_width/sac_in.delta + 0.5))
    LH = lh + 1
    h = np.sin(np.pi*np.arange(LH)/lh)

    # Weighted average
    sac_out.depvar *= 0.
    for i in range(sac_in.npts):
        il  = i
        ir  = i
        cum = h[0] * sac_in.depvar[i]
        for j in range(1,LH):
            il -= 1
            ir += 1
            if il < 0.:
                xl = 0.
            else:
                xl = sac_in.depvar[il]
            if ir >= sac_in.npts:
                xr = 0.
            else:
                xr = sac_in.depvar[ir]
            cum += h[j] * (xr - xl)
        sac_out.depvar[i] = cum

    # Delay trace
    sac_out.b += delay

    # All done
    return sac_out

def conv_by_tri_stf(sac_in,delay,half_width):
    '''
    Convolve by a triangular STF
    '''
    # Copy sac
    sac_out = sac_in.copy()

    # Build triangle
    lh = int(np.floor(half_width/sac_in.delta + 0.5))
    LH = lh + 1
    h = np.zeros((LH,),dtype='float32')
    for i in range(LH):
        h[i] = 1. - np.float32(i)/np.float32(lh)
    al0 = 2. * h.sum() - h[0]
    h /= al0

    # Weighted average
    sac_out.depvar *= 0.
    for i in range(sac_in.npts):
        il  = i
        ir  = i
        cum = h[0] * sac_in.depvar[i]
        for j in range(1,LH):
            il -= 1
            ir += 1
            if il < 0.:
                xl = 0.
            else:
                xl = sac_in.depvar[il]
            if ir >= sac_in.npts:
                xr = 0.
            else:
                xr = sac_in.depvar[ir]
            cum += h[j] * (xl + xr)
        sac_out.depvar[i] = cum

    # Delay trace
    sac_out.b += delay

    # All done
    return sac_out

def ts_hd_misfit(inputs):
    '''
    Sub-function for time-shift and half-duration grid-search
    Args:
        * inputs: tuple including input parameters with two options:
        - 4 params: ts_index, ts_val, zero_trace 
          (do inversion for a single time-shift)
           When 4 parameters are used, Green's functions are already 
           convolved with a STF
        - 5 params: hd_index, hd_val, ts_search, zero_trace 
          (do inversion for a series of time-shifts)
           When 5 parameters are used, Green's functions are convolved 
           with a triangular STF of a specified hd_value
    ''' 
    # Get cmtproblem object
    cmtp = inputs[0]
    cmtp_hd = cmtp.copy() 
     
    # Parse search parameters
    if len(inputs)==4:   # Do inversion for a single time-shift
        start = inputs[1]
        ts_search = [inputs[2]]
        if cmtp.force_flag:
            vertical_force = inputs[3]
        else:
            zero_trace = inputs[3]
    elif len(inputs)==5: # Half-duration grid-search
        start = 0
        j = inputs[1]
        hd = inputs[2]
        ts_search = inputs[3]
        # Convolve with triangle of half-duration hd
        cmtp_hd.preparekernels(delay=0.,stf=hd,read_from_file=False,windowing=False)
        if cmtp.force_flag:
            vertical_force = inputs[4]
        else:
            zero_trace = inputs[4]
        
    # Time-shift search
    i = np.arange(start,start+len(ts_search))
    rms  = np.zeros((len(ts_search),))
    rmsn = np.zeros((len(ts_search),))
    for k,ts in enumerate(ts_search):
        cmtp_ts = cmtp_hd.copy()
        # Prepare kernels
        cmtp_ts.preparekernels(delay=ts,stf=None,read_from_file=False)
        cmtp_ts.buildG()
        cmtp_ts.cmt.ts = ts
         
        # Invert
        if cmtp_ts.force_flag:
            cmtp_ts.forceinv(vertical_force=vertical_force)
        else:
            cmtp_ts.cmtinv(zero_trace=zero_trace)
         
        # Get RMS misfit
        res,nD,nS = cmtp_ts.global_rms
        rms[k] = res/np.sqrt(float(cmtp_ts.D.size))
        rmsn[k] = res/nD
     
    # All done
    if len(inputs)==4:
        return i,ts_search,rms,rmsn
    else:
        return i,j,ts_search,hd,rms,rmsn

def warningconditioning(rank,eigvals,cond_threshold):
    out = sys.stderr
    out.write("### Warning: ill-conditioned matrix ###\n")
    out.write("Condition number: %e\n"%(eigvals[0]/eigvals[-1]))
    out.write("Condition threshold: %e\n"%(cond_threshold))
    out.write("(%d eigvals out of %d)\n"%(rank,len(eigvals)))
    out.write("#######################################\n")

def getICDG(G):
    '''
    Decomposition using Kawakatsu (1996) decomposition for diagonal elements
    Args:
        * G matrix (columns ordered as Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    Returns:
        * G matrix with diagonal elements reordered as I (isotropic)
          C (CLVD) and D (difference)
    '''
    GICD = np.zeros_like(G)
    GICD[:,0] = 1./3. * (G[:,0] + G[:,1] + G[:,2])    # Isotropic
    GICD[:,1] = 1./3. * (G[:,1] + G[:,2] - 2.*G[:,0]) # CLVD
    GICD[:,2] = 1./2. * (G[:,1] - G[:,2])             # Difference
    GICD[:,3] = G[:,3]
    GICD[:,4] = G[:,4]
    GICD[:,5] = G[:,5]
    # All done
    return GICD

def getMTfromICD(MICD):
    '''
    Recombine ICD inverted moment tensor components into standard
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp components
    '''
    MT = np.zeros_like(MICD)
    MT[0] = MICD[0] - MICD[1] # Isotropic - CLVD
    MT[1] = MICD[0]+0.5*MICD[1]+MICD[2]
    MT[2] = MT[1]-2*MICD[2]
    MT[3] = MICD[3]
    MT[4] = MICD[4]
    MT[5] = MICD[5]
    # All done
    return MT

def getICDfromMT(MT):
    '''
    Get ICD moment tensor for standard Mrr, Mtt, Mpp, Mrt, Mrp, Mtp components
    '''
    MICD = np.zeros_like(MT)
    MICD[0] = 1./3. * (MT[0] + MT[1] + MT[2])    # Isotropic
    MICD[1] = 1./3. * (MT[1] + MT[2] - 2.*MT[0]) # CLVD
    MICD[2] = 1./2. * (MT[1] - MT[2])             # Difference
    MICD[3] = MT[3]
    MICD[4] = MT[4]
    MICD[5] = MT[5]
    # All done
    return MICD

class parseConfigError(Exception):
    """
    Raised if the config file is incorrect
    """
    pass


def parseConfig(cfg_file):
    '''
    Parse my config files and returns a config dictionary
    Args:
         cfg_file: configuration filename
    '''

    # Check if cfg_file exists
    assert os.path.exists(cfg_file), 'Cannot read %s: No such file'%(cfg_file)

    # Fill the config dictionary
    config = {}
    try:
        config_lines = open(cfg_file, 'r').readlines()
        for line in config_lines:
            if line.find('#')==0:
                continue
            if line.rstrip():
                key,value = line.strip().split(':')
                key   = key.strip()
                value = value.strip()
                if key in config:
                    config[key].append(value)
                else:
                    config[key]=value
    except:
        raise parseConfigError('Incorrect format in %s!\n'%cfg_file)

    # All done
    return config


# MAIN CLASS
class cmtproblem(object):

    def __init__(self,cmt_flag=False,force_flag=False):
        '''
        Constructor
        Args: 
            * cmt_flag: if True (default), invert for cmt parameters
            * force_flag: if True, invert for a force (default=False)
        Remarks:
        - If cmt_flag and force_flag are True, invert for a complex source including 
          cmt and single force parameters (see cmtforceinv). 
        - If cmt_flag and force_flag are both False, will invert for cmt parameters
          only.
        '''

        # Setting cmt_flag and force_flag flags
        self.force_flag = force_flag
        if cmt_flag is False and force_flag is False:
            self.cmt_flag = True
        else:
            self.cmt_flag   = cmt_flag

        # List of channel ids
        self.chan_ids = []

        # Data and Green's functions
        self.data = None  # data dictionary
        self.synt = None  # synthetic dictionary
        self.gf   = None  # Green's functions sac object      
        self.D = None     # Data vector
        self.G = None     # Green's function matrix
        self.delta = None # Data sampling step
        self.twin = {}    # Time-window dictionary

        # RMS misfit
        self.global_rms = None # Global RMS misfit
        self.rms  = None       # RMS misfit per station

        # CMT/FORCE objects
        self.cmt = cmt()
        if self.force_flag:
            self.force = force()

        # Deconvolution
        self.duration    = None

        # Tapering
        self.taper       = False # Taper flag
        self.taper_n     = None  # Taper width 
        self.taper_left  = None  # Taper (left part)
        self.taper_right = None  # Taper (right part)
        
        # All done
        return

    def cmtinv(self,zero_trace=True,MT=None,scale=1.,rcond=1e-4,ICD=False,get_Cm=False,get_BIC=False):
        '''
        Perform CMTinversion (stored in cmtproblem.cmt.MT)
        Args:
            * zero_trace: if True impose zero trace
            * MT: optional, constrain the focal mechanism (only invert M0)
            * scale: M0 scale
            * rcond: Cut-off ratio  small singular values (default: 1e-4)
            * ICD: recombine MT components into I, C, D, Mrt, Mrp and Mtp
            * get_Cm: if True, return the posterior covariance matrix
            * get_BIC: if True, return the Bayesian information criterion
        '''

        assert self.D is not None, 'D must be assigned before cmtinv'
        assert self.G is not None, 'G must be assigned before cmtinv'
        assert self.cmt_flag is True, 'cmtproblem not properly initialized (self.cmt_flag != True)'
        assert self.G.shape[1] == 6, 'cmtproblem not properly initialized for cmtinv (%d collumns in self.G)'%(self.G.shape[1])

        # Constraints
        if MT is not None: # Fixing the focal mechanism (only invert M0)
            G = self.G.dot(MT)
        elif zero_trace: # Zero trace
            G = np.empty((self.D.size,5))
            if ICD:
                G = getICDG(self.G)
                G = G[:,1:]
            else:
                for i in range(2):
                    G[:,i] = self.G[:,i] - self.G[:,2]
                for i in range(3):
                    G[:,i+2] = self.G[:,i+3]
        else:
            if ICD:
                G = getICDG(self.G)
            else: 
                G = self.G

        # Moment tensor inversion
        if MT is not None:
            m = (G.T.dot(self.D))/(G.T.dot(G))
        else:
            m,res,rank,s = np.linalg.lstsq(G,self.D,rcond=rcond)
            if rank < G.shape[1]:
                warningconditioning(rank,s,1./rcond)
        
        # Fill-out the global RMS attribute
        S  = G.dot(m)
        res = self.D - S
        res = np.sqrt(res.dot(res)/res.size)
        nD  = np.sqrt(self.D.dot(self.D)/self.D.size)
        nS  = np.sqrt(S.dot(S)/S.size)
        self.global_rms = [res,nD,nS]
        
        # Scale moment tensor
        m *= scale

        # Populate cmt attribute
        self.cmt.MT = np.zeros((6,))
        if MT is not None:
            self.cmt.MT = m * MT.copy()
        elif zero_trace:
            if ICD:
                MICD = np.zeros((6,))
                MICD[1:] = m[:]
                self.cmt.MT = getMTfromICD(MICD)
            else:
                self.cmt.MT[5] = m[4]
                self.cmt.MT[4] = m[3]
                self.cmt.MT[3] = m[2]
                self.cmt.MT[0] = m[0]
                self.cmt.MT[1] = m[1]
                self.cmt.MT[2] = -m[0] -m[1]
        else:
            if ICD:
                self.cmt.MT = getMTfromICD(m)
            else:
                self.cmt.MT = m.copy()

        # Return the posterior covariance matrix and BIC
        out = None
        if get_Cm:
            out = np.linalg.inv(G.T.dot(G))
        if get_BIC:
            BIC = self.getBIC(G,m)
            if out is None:
                return BIC
            else:
                out = [out,BIC]

        # All done
        return out


    def forceinv(self,vertical_force=False,F=None,scale=1.,rcond=1e-4,get_Cm=False, get_BIC=False):
        '''
        Perform inversion for a single force (stored in cmtproblem.cmt.force)
        Args:
            * vertical_force: if True impose a pure vertical force
            * F: optional, constrain the force orientation (only invert M0)
            * scale: M0 scale
            * rcond: Cut-off ratio  small singular values (default: 1e-4)
            * get_Cm: if True, return the posterior covariance matrix
            * get_BIC: if True, return the Bayesian information criterion
        '''

        # Check things before doing inversion
        assert self.D is not None, 'D must be assigned before forceinv'
        assert self.G is not None, 'G must be assigned before forceinv'
        assert self.force_flag is True, 'cmtproblem not properly initialized (self.force_flag != True)'
        assert self.G.shape[1] == 3, 'cmtproblem not properly initialized for single force (%d collumns in self.G)'%(self.G.shape[1])

        # Get the Green's functions
        if vertical_force:
            G = self.G[:,-1]
        elif F is not None:
            G = self.G.dot(F)
        else:
            G = self.G
        
        # Inversion
        if vertical_force or (F is not None):
            m = (G.T.dot(self.D))/(G.T.dot(G))
        else:
            m,res,rank,s = np.linalg.lstsq(G,self.D,rcond=rcond)
            if rank < G.shape[1]:
                warningconditioning(rank,s,1./rcond)
    
        # Fill-out global rms values
        S  = G.dot(m)
        res = self.D - S
        res = np.sqrt(res.dot(res)/res.size)
        nD  = np.sqrt(self.D.dot(self.D)/self.D.size)
        nS  = np.sqrt(S.dot(S)/S.size)
        self.global_rms = [res,nD,nS]

        # Scale force vector
        m *= scale
        
        # Populate force attributes
        if vertical_force:
            self.force.F = np.zeros((3,))
            self.force.F[-1] = m.copy()
        elif F is not None:
            self.force.F = m * F.copy()
        else:
            self.force.F = m.copy()

        # Return the posterior covariance matrix and BIC
        out = None
        if get_Cm:
            out = np.linalg.inv(G.T.dot(G))
        if get_BIC:
            BIC = self.getBIC(G,m)
            if out is None:
                return BIC
            else:
                out = [out,BIC]

        # All done
        return out


    def cmtforceinv(self,zero_trace=True,MT=None,vertical_force=False,F=None,scale=1.,rcond=1e-4,get_Cm=False, get_BIC=False):
        '''
        Perform inversion for both CMT and single force paramerters (stored in cmtproblem.cmt.MT and cmtproblem.cmt.force)
        Args:
            * zero_trace: if True impose zero trace
            * MT: optional, constrain the focal mechanism (only invert M0)
            * scale: M0 scale
            * rcond: Cut-off ratio  small singular values (default: 1e-4)
            * get_Cm: if True, return the posterior covariance matrix
            * get_BIC: if True, return the Bayesian information criterion
        '''

        assert self.D is not None, 'D must be assigned before cmtinv'
        assert self.G is not None, 'G must be assigned before cmtinv'
        assert self.G.shape[1] == 9, 'cmtproblem not properly initialized for cmtforceinv (%d collumns in self.G)'%(self.G.shape[1])

        # Get Green's functions for CMT parameters
        if MT is not None: # Fixing the focal mechanism (only invert M0)
            Gcmt = (self.G[:,:6].dot(MT)).reshape(self.D.size,1)
        elif zero_trace: # Zero trace
            Gcmt = np.empty((self.D.size,5))
            for i in range(2):
                Gcmt[:,i] = self.G[:,i] - self.G[:,2]
            for i in range(3):
                Gcmt[:,i+2] = self.G[:,i+3]
        else:
            Gcmt = self.G[:,:6].copy()

        # Get the Green's functions for a single force
        if vertical_force:
            Gforce = (self.G[:,-1]).reshape(self.D.size,1)
        elif F is not None:
            Gforce = (self.G[:,6:].dot(F)).reshape(self.D.size,1)
        else:
            Gforce = self.G[:,6:].copy()

        # Building the G matrix 
        G = np.append(Gcmt,Gforce,axis=1)
        
        # Moment tensor inversion
        m,res,rank,s = np.linalg.lstsq(G,self.D,rcond=rcond)
        if rank < G.shape[1]:
            warningconditioning(rank,s,1./rcond)

        # Fill-out the global RMS attribute
        S  = G.dot(m)
        res = self.D - S
        res = np.sqrt(res.dot(res)/res.size)
        nD  = np.sqrt(self.D.dot(self.D)/self.D.size)
        nS  = np.sqrt(S.dot(S)/S.size)
        self.global_rms = [res,nD,nS]
        
        # Scale moment tensor
        m *= scale

        # Populate cmt attributes
        self.cmt.MT = np.zeros((6,))
        if MT is not None:
            self.cmt.MT = m[0] * MT.copy()
        elif zero_trace:
            self.cmt.MT[5] = m[4]
            self.cmt.MT[4] = m[3]
            self.cmt.MT[3] = m[2]
            self.cmt.MT[0] = m[0]
            self.cmt.MT[1] = m[1]
            self.cmt.MT[2] = -m[0] -m[1]
        else:
            self.cmt.MT = m[:6].copy()

        # Populate force attributes
        if vertical_force:
            self.force.F = np.zeros((3,))
            self.force.F[-1] = m[-1]
        elif F is not None:
            self.force.F = m[-1] * F.copy()
        else:
            self.force.F = m[:6:].copy()

        # Return the posterior covariance matrix
        out = None
        if get_Cm:
            out = np.linalg.inv(G.T.dot(G))
        if get_BIC:
            BIC = self.getBIC(G,m)
            if out is None:
                return BIC
            else:
                out = [out,BIC]

        # All done
        return out

    def getBIC(self,G,m,Cd=None):
        '''
        Get the Bayesian Information Criterion (e.g., Bishop, 2006)
        (assuming a broad prior)
        Args:
            * G: Green's function matrix
            * m: model vector (MAP)
            * Cd: data covariance matrix
        '''
        
        # Check the inputs
        assert self.D is not None, 'self.D should exist before using getBIC'
        d = self.D
        assert d.size == G.shape[0], 'incorrect Green function or data size'
        assert m.size == G.shape[1], 'incorrect Green function or model size'
        N = float(d.size)
        M = float(m.size)
        if Cd is None:
            Cdi = np.eye(d.size)
            Cd_det = 1.
        else:
            assert Cd.shape[0] == s.size, 'Incorrect size for Cd'
            Cdi = np.linalg.inv(Cd)
            Cd_det = np.linalg.det(Cd)
        
        # Log-likelihood 
        llk = 0.5 * (d - G.dot(m)).T.dot(Cdi).dot(d - G.dot(m)) 
        llk -= 0.5*N*np.log(2.*np.pi) + 0.5*np.log(Cd_det)
        BIC = llk - 0.5*M*np.log(N)

        # All done
        return BIC
        
    def buildD(self):
        '''
        Build D matrices from data dictionary
        '''

        self.D = []
        for chan_id in self.chan_ids:
            self.D.extend(self.data[chan_id].depvar)
        self.D = np.array(self.D)

        # All done
        return

    def buildG(self):
        '''
        Build G matrices from data dictionary
        '''

        # Initialize G matrix
        self.G = []
        
        # CMT parameters
        if self.cmt_flag:
            for mt in self.cmt.MTnm:
                self.G.append([])
                for chan_id in self.chan_ids:
                    self.G[-1].extend(self.gf[chan_id][mt].depvar)
        
        # FORCE parameters
        if self.force_flag:
            for f in self.force.Fnm:
                self.G.append([])
                for chan_id in self.chan_ids:
                    self.G[-1].extend(self.gf[chan_id][f].depvar)

        # Final touch
        self.G = np.array(self.G).T
        
        # All done
        return
        
    def preparedata(self,i_sac_lst,filter_freq=None,filter_order=4,filter_btype='bandpass',wpwin=False,
                    swwin=None,dcwin={},taper_width=None,derivate=False,o_dir=None,o_sac_lst=None):
        '''
        Prepare Data before cmt inversion
        Args:
            * i_sac_lst: list of input data sac file 
            * filter_freq (optional): filter corner frequencies (see sacpy.filter)
            * filter_order (optional): default is 4 (see sacpy.filter)
            * filter_btype (optional): default is 'bandpass' (see sacpy.filter)
            * wpwin: if True, use W-phase window
            * swwin: surface wave windowing (optional):
               - if wpwin=False: Surface wave type windowing
                     Time window from dist/swwin[0] to dist/swwin[1]
                     if swwin[2] exists, it is a minimum window length
               - if wpwin=True: W-phase time windowing
                     Time window from P arrival time (Ptt) + swwin*gcarc
            * dcwin: dictionnary of time-windows for individual channels (optional):
                       channel keys are defined following the format of sacpy.sac.id
                       For each channel:
               - if wpwin=False: [tbeg,tend] list with respect to origin time
               - if wpwin=True:  [tbeg,tend] with respect to P arrival time
            * taper_width: Apply taper to data
            * derivate: if True, will differenciate the data
            * o_dir: Output directory for filtered data (optional)
            * o_sac_lst: output sac list (if o_dir is not None, default='o_sac_lst')
        '''
        
        # Read sac file list
        L = open(i_sac_lst).readlines()
        ifiles = []
        for l in L:
            # Skip commented lines
            if l[0]=='#':
                continue            
            ifiles.append(l.strip().split()[0])
        
        if o_dir is not None:
            if o_sac_lst is None:
                o_sac_lst=os.path.join(o_dir,'o_sac_lst')
            o_lst = open(o_sac_lst,'wt')            
        
        # Get delta (should be identical for all channels)
        data_sac = sac(ifiles[0])
        self.delta = data_sac.delta

        # Instantiate data dict
        self.data = {}
        self.chan_ids = []
        
        # Instantiate sacpy.sac
        data_sac = sac()        
        
        # Instanciate time window dictionary
        self.twin = {}

        # Hanning taper
        self.taper = False
        if taper_width is not None:
            self.taper = True
            self.taper_n = int(taper_width/self.delta)
            H  = np.hanning(2*self.taper_n)
            zeros = np.zeros((self.taper_n))
            self.taper_left  = np.append(zeros,H[:self.taper_n])
            self.taper_right = np.append(H[self.taper_n:],zeros)
            self.taper_n *= 2
        
        # Loop over sac_files
        for ifile in ifiles:
            
            # Read sac file            
            data_sac.read(ifile)
            assert np.round(data_sac.delta,3)==self.delta, 'data should be sampled at 1sps'
            
            # Filter
            if filter_freq is not None:
                data_sac.filter(freq=filter_freq,order=filter_order,btype=filter_btype)

            # Differentiate
            if derivate:
                data_sac.derivate()            

            # Output directory for unwindowed filtered sac data
            if o_dir is not None:
                ofile=os.path.join(o_dir,os.path.basename(ifile)+'.filt')
                o_lst.write('%s %s\n'%(ofile,data_sac.id))
                data_sac.wsac(ofile)
                
            # Time-window
            chan_id = data_sac.id
            if wpwin:
                assert data_sac.gcarc >= 0., 'gcarc must be assigned in sac data header'
                tbeg = data_sac.t[0]-data_sac.o
                if swwin is not None:
                    tend = tbeg + swwin * data_sac.gcarc
                else:
                    tend = tbeg + 15.0  * data_sac.gcarc
                
            elif swwin is not None:
                assert len(swwin)>=2,    'swwin must be [V_window, Window_width]'
                assert swwin[0]>swwin[1], 'vbeg must be larger than vend'
                assert data_sac.dist >= 0., 'dist must be assigned in sac data header'
                tbeg = data_sac.dist/swwin[0]
                tend = data_sac.dist/swwin[1]
                if len(swwin)>2 and (tend-tbeg) < swwin[2]:
                    tend = tbeg + swwin[2]
                if len(swwin)>3:
                    tbeg -= swwin[3]
                    if (tend-tbeg) < swwin[2]:
                        tend = tbeg + swwin[2]

            if chan_id in dcwin:                
                if wpwin:
                    tbeg = dcwin[chan_id][0] + data_sac.t[0]-data_sac.o
                    tend = dcwin[chan_id][1] + data_sac.t[0]-data_sac.o
                else:
                    tbeg = dcwin[chan_id][0] 
                    tend = dcwin[chan_id][1] 

            if wpwin or dcwin or swwin is not None:
                self.twin[chan_id] = [tbeg,tend]              
                ib = int((tbeg+data_sac.o-data_sac.b)/data_sac.delta)
                ie = ib+int((tend-tbeg)/data_sac.delta)
                t    = np.arange(data_sac.npts)*data_sac.delta+data_sac.b-data_sac.o
                if ib<0 or ie>data_sac.npts:
                        sys.stderr.write('Warning: Incomplete data for %s (ib<0 or ie>npts): Rejected\n'%(ifile))
                        continue
                if self.taper:
                    ib -= self.taper_n
                    ie += self.taper_n
                data_sac.depvar = data_sac.depvar[ib:ie+1].copy()
                if self.taper:
                    data_sac.depvar[:self.taper_n]  *= self.taper_left
                    data_sac.depvar[-self.taper_n:] *= self.taper_right
                data_sac.t[0] = tbeg+data_sac.o
                data_sac.b    = t[ib]+data_sac.o
                data_sac.e    = t[ie]+data_sac.o
                data_sac.npts = len(data_sac.depvar)

            # Populate the dictionary
            self.data[data_sac.id] = data_sac.copy()
            self.chan_ids.append(data_sac.id)

        if o_dir is not None:
            o_lst.close()
        
        # All done            
        return

    def setTimeWindow(self,wpwin=False,swwin=None,dcwin={}):
        '''
        Set time-window parameters in self.twin
        '''
        self.twin = {}
        for chan_id in self.data:
            # Data file
            data_sac = self.data[chan_id]
            
            # W-phase time-window
            if wpwin:
                assert data_sac.gcarc >= 0., 'gcarc must be assigned in sac data header'
                tbeg = data_sac.t[0]-data_sac.o
                if swwin is not None:
                    tend = tbeg + swwin * data_sac.gcarc
                else:
                    tend = tbeg + 15.0  * data_sac.gcarc

            # Surface time-window    
            elif swwin is not None:
                assert len(swwin)>=2,    'swwin must be [V_window, Window_width]'
                assert swwin[0]>swwin[1], 'vbeg must be larger than vend'
                assert data_sac.dist >= 0., 'dist must be assigned in sac data header'
                tbeg = data_sac.dist/swwin[0]
                tend = data_sac.dist/swwin[1]
                if len(swwin)>2 and (tend-tbeg) < swwin[2]:
                    tend = tbeg + swwin[2]
                if len(swwin)>3:
                    tbeg -= swwin[3]
                    if (tend-tbeg) < swwin[2]:
                        tend = tbeg + swwin[2]
            else:
                tbeg = data_sac.b - data_sac.o
                tend = data_sac.e - data_sac.e                            
        
            # Time-window defined for individual channels
            if chan_id in dcwin:                
                if wpwin:
                    tbeg = dcwin[chan_id][0] + data_sac.t[0]-data_sac.o
                    tend = dcwin[chan_id][1] + data_sac.t[0]-data_sac.o
                else:
                    tbeg = dcwin[chan_id][0] 
                    tend = dcwin[chan_id][1] 

            self.twin[chan_id] = [tbeg,tend]
            
        # All done
        return

    def preparekernels(self,GF_names=None,stf=None,delay=0.,filter_freq=None,filter_order=4,filter_btype='bandpass',
                       baseline=0,left_taper=False,wpwin=False,derivate=False,scale=1.,read_from_file=True,
                       windowing=True):
        '''
        Prepare Green's functions (stored in self.gf dictionary as sacpy instances)
        Args:
            * GF_names : dictionary of GFs names (i.e., path to sac files)
                Must be structured as follows:
                GF_names = {'channel_id1': {'MTcmp': gf_sac_file_name, ...}, ...}
                where:
                    - channel_id1 is the channel id in the sacpy format "knetwk_kstnm_khole_kcmpnm"
                    -  indicates the moment tensor component (i.e., 'rr', 'tt', 'pp', 'rt', 'rp', 'tp')
            * stf : moment rate function (optionnal, default=None)
                - can be a scalar giving a triangular (cmt) or sinusoidal (force) STF half-duration
                - can be a list or array of len==2 with triangular and sinusoidal hald-furation when 
                  inverting for both cmt and force parameters
                - can be a single array used for all stations
                - can be a dictionary with one stf per channel id (apparent STFs)
            * delay: time-shift (in sec, optional)
                - can be a single value used for all stations
                - can be a list or array of len==2 when inverting for both force and cmt parameters
                - can be a dictionary with one delay per channel id
            * filter_freq (optional): filter corner frequencies (see sacpy.filter)
            * filter_order (optional): default is 4 (see sacpy.filter)
            * filter_btype (optional): default is 'bandpass' (see sacpy.filter)
            * baseline : number of samples to remove baseline (default: no baseline)
            * left_taper: if True, apply left taper over baseline (optional)
            * wpwin: Deprecated
            * derivate: if True, will differentiate the Green's functions
            * scale: scaling factor for all GFs (optional)
            * read_from_file: option to read the GF database
                - if True, load the Green's functions from the sac files in the GF database
                - if False, use the Green's functions that were previously stored in self.gf                        
        '''
        
        # sacpy.sac instantiation
        gf_sac = sac()
        
        # gf dictionary
        if self.gf is None or read_from_file:
            self.gf = {}
            assert GF_names is not None, 'GF_names must be specified'
            chan_ids = GF_names.keys()
        else:
            chan_ids = self.chan_ids
        
        # Assign cmt delay
        triangular_stf = False
        sinusoidal_stf = False
        if not isinstance(delay,dict): # Not a delay dictionary
            if isinstance(delay,float) or isinstance(delay,int):
                self.cmt.ts = delay
                if self.force_flag:
                    self.force.ts = delay
            else:
                assert len(delay)==2, 'Incorrect input delay'
                assert self.cmt_flag is True and self.force_flag is True, 'Incorrect delay in preparekernels'
                self.cmt.ts = delay[0]
                self.force.ts = delay[1]
            if stf is not None:
                if isinstance(stf,float) or isinstance(stf,int): # Triangular stf (cmt) or sinusoidal (force)
                    triangular_stf = True
                    self.cmt.hd    = float(stf)
                    if self.force_flag:
                        self.force.hd = float(stf)
                        triangular_stf = False
                        sinusoidal_stf = True
                elif len(stf) == 2: # half-durations for both force and cmt parameters
                    assert self.cmt_flag is True and self.force_flag is True, 'Incorrect stf in preparekernels'
                    triangular_stf = True
                    sinusoidal_stf = True
                    self.cmt.hd    = float(stf[0])
                    self.force.hd  = float(stf[1])
                else:               # half duration is half the len of stf
                    self.cmt.hd = (len(stf)-1)*0.5
                    if self.force_flag:
                        self.force.hd = (len(stf)-1)*0.5
                        
            else:
                if isinstance(delay,float) or isinstance(delay,int):
                    self.cmt.hd = delay
                    if self.force_flag:
                        self.force.hd = delay
                else:
                    assert len(delay)==2, 'Incorrect input delay'
                    self.cmt.hd   = delay[0]
                    self.force.hd = delay[1]
        else:
            if isinstance(stf,float) or isinstance(stf,int): # Triangular/Sinusoidal stf
                triangular_stf = True
                self.cmt.hd    = float(stf)
                if self.force_flag:
                    self.force.hd    = float(stf)
                    triangular_stf = False
                    sinusoidal_stf = True
            elif len(stf) == 2: # half-durations for both force and cmt parameters
                triangular_stf = True
                sinusoidal_stf = True
                self.cmt.hd    = float(stf[0])
                self.force.hd  = float(stf[1])
             
        # Loop over channel ids
        for chan_id in chan_ids:

            read_GF = False
            if chan_id not in self.gf or read_from_file:
                self.gf[chan_id] = {}
                assert chan_id in GF_names, 'missing channel id (%s)'%(chan_id)
                read_GF = True

            # Loop over moment-tensor/force components
            nms = []
            if self.cmt_flag:
                nms += self.cmt.MTnm
            if self.force_flag:
                nms += self.force.Fnm
            for m in nms:

                # Read Green's functions
                if read_GF: # Read GF sac file
                    gf_sac.read(GF_names[chan_id][m])
                else:       # Get GF from the self.gf dictionary
                    gf_sac = self.gf[chan_id][m].copy()
                 
                # Check the sampling rate (to be improved)
                assert np.round(gf_sac.delta,3)==self.delta, 'GFs should be sampled with data sampling step (%.3f s)'%(self.delta)
                 
                # Remove baseline
                if baseline>0:
                    ibase = int(baseline/gf_sac.delta)
                    av = gf_sac.depvar[:ibase].mean()
                    gf_sac.depvar -= av                    
                 
                # Left taper
                if left_taper:
                    assert baseline>0., 'Baseline must be >0. for left taper' 
                    ibase = int(baseline/gf_sac.delta)                    
                    ang = np.pi / baseline
                    inds = np.arange(ibase)
                    gf_sac.depvar[inds] *= (1.0 - np.cos(inds*ang))/2.
                 
                # Scale GFs
                gf_sac.depvar *= scale
                 
                # Convolve with STF(s)
                if stf is not None:
                    if m in self.cmt.MTnm and triangular_stf: # Convolve with a triangular stf
                        gf_sac = conv_by_tri_stf(gf_sac,0.,self.cmt.hd)

                    elif m in self.force.Fnm and sinusoidal_stf:
                        gf_sac = conv_by_sin_stf(gf_sac,0.,self.force.hd)
                            
                    elif isinstance(stf,np.ndarray) or isinstance(stf,list): 
                        gf_sac.depvar=np.convolve(gf_sac.depvar,stf,mode='same')
                        
                    else:
                        assert chan_id in stf, 'No channel id %s in stf'%(chan_id)
                        gf_sac.depvar=np.convolve(gf_sac.depvar,stf[chan_id],mode='same')

                # Time-shift
                if isinstance(delay,dict):
                    assert chan_id in delay, 'No channel id %s in delay'%(chan_id)
                    gf_sac.b += delay[chan_id]                    
                elif isinstance(delay,float) or isinstance(delay,int):
                    gf_sac.b += delay
                else:
                    assert len(delay)==2, 'Incorrect input delay'
                    assert (m in self.cmt.MTnm) or (m in self.force.Fnm), 'Incorrect MT/FORCE component'
                    if m in self.cmt.MTnm:
                        gf_sac.b += delay[0]
                    else:
                        gf_sac.b += delay[1]

                # Filter
                if filter_freq is not None:
                    gf_sac.filter(freq=filter_freq,order=filter_order,btype=filter_btype)                

                if derivate:
                    gf_sac.derivate()            

                # Time-window matching data
                if self.data is not None and windowing is not False:
                    assert chan_id in self.data, 'No channel id %s in data'%(chan_id)
                    data_sac = self.data[chan_id]                    
                    b    = data_sac.b - data_sac.o
                    npts = data_sac.npts                    
                    t    = np.arange(gf_sac.npts)*gf_sac.delta+gf_sac.b-gf_sac.o
                    if wpwin:
                        t0 = data_sac.t[0]-data_sac.o
                    else:
                        t0 = data_sac.b-data_sac.o

                    if chan_id in self.twin:
                        t0 = self.twin[chan_id][0]
                    
                    ib = int((t0-gf_sac.b)/gf_sac.delta)
                    ie = ib+data_sac.npts                    

                    if ib<0:
                        gf_sac.depvar = np.append(np.zeros((-ib,)),gf_sac.depvar)
                        ib = 0
                    assert ib>=0, 'Incomplete GF (ie<0)'
                    assert ie<=gf_sac.npts,'Incomplete GF (ie>npts)'                    

                    gf_sac.depvar = gf_sac.depvar[ib:ib+npts]
                    if self.taper:
                        gf_sac.depvar[:self.taper_n]  *= self.taper_left
                        gf_sac.depvar[-self.taper_n:] *= self.taper_right                    
                    gf_sac.kstnm  = data_sac.kstnm
                    gf_sac.kcmpnm = data_sac.kcmpnm
                    gf_sac.knetwk = data_sac.knetwk
                    gf_sac.khole  = data_sac.khole
                    gf_sac.id     = data_sac.id
                    gf_sac.stlo   = data_sac.stlo
                    gf_sac.stla   = data_sac.stla
                    gf_sac.npts   = npts
                    gf_sac.b      = t[ib]+gf_sac.o

                # Populate the dictionary
                self.gf[chan_id][m] = gf_sac.copy()

        # All done
        return
    
    def ts_hd_gridsearch(self,ts_search,hd_search,GF_names,filter_freq=None,filter_order=4,
                        filter_btype='bandpass',derivate=False,zero_trace=True,vertical_force=False,ncpu=None):
        '''
        Performs a grid-search to get optimum centroid time-shifts and half-duration
        Args:
            * ts_search: time-shift values to be explored (list or ndarray)
            * hd_search: half-duration values to be explored (list or ndarray)
            * GF_names : dictionary of GFs names (see preparekernels)
            * filter_freq (optional): filter corner frequencies (see sacpy.filter)
            * filter_order (optional): default is 4 (see sacpy.filter)
            * filter_btype (optional): default is 'bandpass' (see sacpy.filter) 
            * derivate (optional): default is False (do not derivate green's functions)
            * zero_trace (optional) : default is True (impose zero trace)
            * vertical_force (optional) : default is False (if True impose vertical force for single force inversion)
            * ncpu (optional): number of cpus (default is the number of CPUs in the system)
        '''
         
        # Number of cores
        if ncpu is None:
            ncpu = cpu_count()
         
        # Initialize rms arrays
        rms  = np.zeros((len(ts_search),len(hd_search)))
        rmsn = np.zeros((len(ts_search),len(hd_search)))
         
        # constrain (zero_trace or vertical_force)
        constraint = zero_trace
        if self.force_flag:
            constraint = vertical_force        

        # Initialize the grid-search
        todo = []
        if len(hd_search)==1: # Parallelism is done with respect to ts_search
            # Prepare the kernels
            self.preparekernels(GF_names,delay=0.,stf=hd_search[0],filter_freq=filter_freq,filter_order=filter_order,
                                filter_btype=filter_btype,derivate=derivate,windowing=False)
            # Todo list
            for i,ts in enumerate(ts_search):
                todo.append([self,i,ts,constraint])
        else: # Parallelism is done with respect to hd_search
            # Prepare the kernels
            self.preparekernels(GF_names,delay=0.,stf=None,filter_freq=filter_freq,filter_order=filter_order,
                                filter_btype=filter_btype,derivate=derivate,windowing=False)
            # Todo list
            for j,hd in enumerate(hd_search):
                todo.append([self,j,hd,ts_search,constraint])
                
        # Do the grid-search
        pool = Pool(ncpu)
        outputs = pool.map(ts_hd_misfit,todo)
         
        # Fill out the rms matrices
        rms  = np.zeros((len(ts_search),len(hd_search)))
        rmsn = np.zeros((len(ts_search),len(hd_search)))
        for output in outputs:
            # Parse outputs
            if len(hd_search)==1:
                j  = 0
                i  = output[0]
                r  = output[2]
                rn = output[3]
            else:
                i  = output[0]
                j  = output[1]
                r  = output[4]
                rn = output[5]
            # Fill out rms
            rms[i,j]  = r
            rmsn[i,j] = rn
        # All done
        return rms, rmsn
        

    def calcsynt(self,scale=1.,stf=None):
        '''
        Compute synthetics. If data exists, will also compute rms
        '''

        # Check that gf and cmt are assigned
        assert self.gf is not None, 'Green s function must be computed'
        if self.force_flag:
            assert self.force.F[0] is not None, 'Force must be assigned'
        else:
            assert self.cmt.MT[0] is not None, 'Moment tensor must be assigned'

        # Assign synt
        self.synt = {}

        # Assign rms
        if self.data is not None:
            self.rms    = {}
            
        # Loop over channel ids
        for chan_id in self.chan_ids:
            self.synt[chan_id] = self.data[chan_id].copy()
            self.synt[chan_id].depvar *= 0.

            # Loop over moment tensor components 
            if self.cmt_flag:
                for m in range(6):
                    MTnm=self.cmt.MTnm[m]
                    self.synt[chan_id].depvar += self.cmt.MT[m]*self.gf[chan_id][MTnm].depvar*scale
                
            # Loop over force components
            if self.force_flag:
                for m in range(3):
                    Fnm=self.force.Fnm[m]
                    self.synt[chan_id].depvar += self.force.F[m]*self.gf[chan_id][Fnm].depvar*scale

            # STF convolution
            if stf is not None:
                npts_fft = nextpow2(self.synt[chan_id].npts)
                fsynt    = np.fft.fft(self.synt[chan_id].depvar,n=npts_fft)
                freqs= np.fft.fftfreq(n=npts_fft)
                if isinstance(stf,np.ndarray) or isinstance(stf,list):
                     fstf = np.fft.fft(stf,n=npts_fft)
                else:
                     assert chan_id in stf, 'No channel id %s in stf'%(chan_id)
                     fstf = np.fft.fft(stf[chan_id],n=npts_fft)
                self.synt[chan_id].depvar = np.real(np.fft.ifft(fsynt*fstf))[:self.synt[chan_id].npts]
                #import matplotlib.pyplot as plt                                     
                #plt.plot(self.data[chan_id].depvar,'k-')
                #plt.plot(self.synt[chan_id].depvar,'b-')
                #plt.title('%s %.2f'%(chan_id,self.synt[chan_id].az))
                #plt.show()

            # RMS calculation
            if self.data is not None:
                res = self.synt[chan_id].depvar - self.data[chan_id].depvar
                res = np.sqrt(res.dot(res)/float(self.data[chan_id].npts))
                nsynt = self.synt[chan_id].depvar.dot(self.synt[chan_id].depvar)
                nsynt = np.sqrt(nsynt/float(self.synt[chan_id].npts))
                ndata = self.data[chan_id].depvar.dot(self.data[chan_id].depvar)
                ndata = np.sqrt(ndata/float(self.data[chan_id].npts))                
                self.rms[chan_id] = [res,ndata,nsynt]
                

        # All done
        return

    def deconv_projlandweber(self,duration=None,nit=1000,nit_min=100,gauss_std=6.,gauss_n=50,fwahm=None):
        '''
        
        '''
        # Check that data and synt are available
        assert self.data is not None, 'data not available'
        assert self.synt is not None, 'synt not available'

        # Smoothing gaussian
        if fwahm is not None:
            sigma_avg   = fwahm/(2.0*np.sqrt(2.0*np.log(2)))

        # Main loop
        self.stf = {}
        stf_duration = None
        for chan_id in self.chan_ids:

            # Number of samples for fft
            npts_fft = nextpow2(self.synt[chan_id].npts)
            
            # Get data and synthetics
            data = self.data[chan_id]
            synt = self.synt[chan_id]
            npts = data.npts            
            assert npts==synt.npts, 'data and synt must have the same number of samples'
            assert np.round(data.delta,4)==np.round(synt.delta,4), 'data and synt must have the same sampling frequency'

            # Pre-convolution with a Gaussian (cf., Vallee et al., 2010)
            #G  = signal.gaussian(32,std=4.4)
            G  = signal.gaussian(gauss_n,std=gauss_std)
            iG = G.sum()*data.delta
            G /= iG
            fG = np.fft.fft(G,n=npts_fft)
            fdata       = np.fft.fft(data.depvar,n=npts_fft)
            fdata       = fdata*fG
            data.depvar = np.real(np.fft.ifft(fdata))[:self.synt[chan_id].npts]

            #import matplotlib.pyplot as plt
            #plt.subplot(211)
            #plt.plot(G)
            #plt.subplot(212)
            #plt.semilogy(np.fft.fftfreq(npts_fft),np.abs(fG))
            #plt.show()
            
            
            # FFT synthetics
            npts_fft = nextpow2(npts)
            fsynt = np.fft.fft(synt.depvar,n=npts_fft)
            fsynt_max = np.absolute(fsynt).max()
            data_norm = data.depvar.dot(data.depvar)            

            # Step
            tau   = 1.0/(fsynt_max*fsynt_max)

            # STF vectors
            fstf  = np.zeros((npts_fft,))      # Freq. domain
            stf   = np.zeros_like(data.depvar) # Time domain
            t_stf = np.arange(npts)*data.delta # Time vector

            if duration is not None:
                stf_duration = duration
            elif self.duration is not None:
                if chan_id in self.duration:
                    stf_duration = self.duration[chan_id]
                else:
                    stf_duration = t_stf.max()                        
            else:
                stf_duration = t_stf.max()
            
                
            # Projected Landweber
            stf_p = stf.copy()
            eps_p = None
            for it in range(nit):
                # Update STF
                g    = fstf + tau * np.conjugate(fsynt)*(fdata-fsynt*fstf)      
                stf  = np.real(np.fft.ifft(g,n=npts_fft))[:npts]
                stf  = Pplus(stf,t_stf,stf_duration)
                # Compute predictions
                fstf = np.fft.fft(stf,n=npts_fft) 
                pred = np.real(np.fft.ifft(fsynt*fstf))[:npts]
                # Exit conditions
                eps = np.sqrt(np.sum((data.depvar-pred)*(data.depvar-pred))/data_norm) # Misfit
                stfo = stf.copy()
                if eps_p is not None:
                    # delta(stf)                
                    delta = np.sqrt(np.sum((stf-stf_p)*(stf-stf_p))/np.sum(stf_p*stf_p))
                    if eps>eps_p:
                        print(eps_p)
                        stf = stf_p.copy()
                        break                    
                    if it>=nit_min and (np.absolute(eps_p-eps)/eps_p<0.00001 or delta<0.00001):
                        break
                
                # Update stf_p and eps_p
                stf_p = stf.copy()
                eps_p = eps

            # Convolve with gaussian
            if fwahm is not None:
                gaussw  = signal.gaussian(npts,std=sigma_avg)
                igaussw = gaussw.sum()*data.delta
                gaussw /= igaussw
                stf = np.convolve(stf,gaussw,mode='same')
            
            # Fill out dictionary
            self.stf[chan_id]=stf.copy()
            #import matplotlib.pyplot as plt
            #fstf = np.fft.fft(stf,n=npts_fft) 
            #pred = np.real(np.fft.ifft(fsynt*fstf))[:npts]
            ##pred = np.real(np.fft.ifft(fsynt))[:npts]
            #plt.plot(data.depvar)
            #plt.plot(pred)
            #plt.title(chan_id)
            #plt.show()

        # All done
        return

    def traces(self,length=3000,i_sac_lst=None,show_win=False,swwin=None,wpwin=None,t0delay=150.,
               variable_xlim=False,rasterize=True,staloc=None,ofile='traces.pdf',yfactor=1.1):
        '''
        Plot data / synt traces
        '''

        import matplotlib
        matplotlib.use('PDF')
        matplotlib.rcParams.update(TRACES_PLOTPARAMS)
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap        

        # Use input sac list instead of self.data        
        if i_sac_lst is not None:
            assert os.path.exists(i_sac_lst),'%s not found'%(i_sac_lst)
            sacdata = sac()
            i_sac = {}
            L = open(i_sac_lst).readlines()
            for l in L:
                if l[0]=='#':
                    continue
                items = l.strip().split()
                i_sac[items[1]] = items[0]
            
        # Create figure
        fig = plt.figure()
        fig.subplots_adjust(bottom=0.06,top=0.87,left=0.06,right=0.95,wspace=0.25,hspace=0.35)        
        pp = matplotlib.backends.backend_pdf.PdfPages(ofile)
        nc = 3
        nl = 5
        perpage = nc * nl

        if staloc is None:
            coords = []
            for chan_id in self.chan_ids:
                sacdata = self.data[chan_id].copy()
                coords.append([sacdata.stla,sacdata.stlo,sacdata.az,sacdata.dist])
            coords = np.array(coords)
        else:
            coords = staloc.copy()       
        
        # Loop over channel ids
        ntot   = len(self.chan_ids)
        npages = np.ceil(float(ntot)/float(perpage))
        nchan = 1
        count = 1        
        pages = 1
        for chan_id in self.chan_ids:
            if i_sac_lst is not None:                
                sacdata.read(i_sac[chan_id])
            else:
                sacdata = self.data[chan_id].copy()
            sacsynt = self.synt[chan_id].copy()
            if count > perpage:
                plt.suptitle('CMT3D,   p %d/%d'%(pages,npages), fontsize=16, y=0.95)
                ofic = 'page_W_%02d.pdf'%(pages)
                print(ofic)
                fig.set_rasterized(rasterize)
                pp.savefig(orientation='landscape')
                plt.close()
                pages += 1
                count = 1
                fig = plt.figure()
                fig.subplots_adjust(bottom=0.06,top=0.87,left=0.06,right=0.95,wspace=0.25,hspace=0.35)
            # Time window
            t1 = np.arange(sacdata.npts,dtype='double')*sacdata.delta + sacdata.b - sacdata.o
            t2 = np.arange(sacsynt.npts,dtype='double')*sacsynt.delta + sacsynt.b - sacsynt.o
            # Plot trace
            ax = plt.subplot(nl,nc,count)
            plt.plot(t1,sacdata.depvar*1000.,'k-')
            plt.plot(t2,sacsynt.depvar*1000.,'r-')
            # Axes limits
            #plt.xlim([t1[0],t1[-1]+(t1[-1]-t1[0])*0.4])
            t0 = t1[0] - t0delay
            if t0<0.:
                t0 = 0.
            plt.xlim([t0,t0+length])
            a    = np.absolute(sacdata.depvar).max()*1000.
            ymin = -yfactor*a
            ymax =  yfactor*a
            if show_win:
                sacdata = self.data[chan_id]
                if wpwin:
                    assert sacdata.gcarc >= 0., 'gcarc must be assigned in sac data header'
                    tbeg = sacdata.t[0] - sacdata.o
                    if swwin is not None:
                        tend = tbeg + swwin * sacdata.gcarc
                    else:
                        tend = tbeg + 15.0  * sacdata.gcarc
                
                elif swwin is not None:
                    assert len(swwin)>=2,    'swwin must be [V_window, Window_width]'
                    assert swwin[0]>swwin[1], 'vbeg must be larger than vend'
                    assert sacdata.dist >= 0., 'dist must be assigned in sac data header'
                    tbeg = sacdata.dist/swwin[0]
                    tend = sacdata.dist/swwin[1]
                    if len(swwin)>2 and (tend-tbeg) < swwin[2]:
                        tend = tbeg + swwin[2]
                elif chan_id in self.twin:
                    tbeg,tend = self.twin[chan_id]
                else:                    
                    tbeg = self.data[chan_id].b - self.data[chan_id].o
                    tend = self.data[chan_id].e - self.data[chan_id].o
                plt.plot([tbeg,tend],[0,0],'ro')
                ib = int((tbeg+sacdata.o-sacdata.b)/sacdata.delta)
                ie = ib+int((tend-tbeg)/sacdata.delta)
                if ib<0:
                    ib = 0
                if ie>sacdata.npts:
                    ie = sacdata.npts
                a    = np.absolute(sacdata.depvar[ib:ie]).max()*1000.
                ymin = -yfactor*a
                ymax =  yfactor*a                
                if variable_xlim:
                    plt.xlim([tbeg - t0delay,tend+length])
            ylims = [ymin,ymax]
            plt.ylim(ylims)                    
            # Annotations
            if sacdata.kcmpnm[2] == 'Z':
                label = r'%s %s %s %s $(\phi,\Delta) = %6.1f^{\circ}, %6.1f^{\circ}$'%(
                    sacdata.knetwk,sacdata.kstnm, sacdata.kcmpnm, sacdata.khole,
                    sacdata.az, sacdata.gcarc)
            else:
                label  = r'%s %s %s %s $(\phi,\Delta,\alpha) = %6.1f^{\circ},'
                label += '%6.1f^{\circ}, %6.1f^{\circ}$'
                label  = label%(sacdata.knetwk,sacdata.kstnm, sacdata.kcmpnm, sacdata.khole,
                                sacdata.az, sacdata.gcarc, sacdata.cmpaz)
            label = label#+' %.1f'%(self.rms[chan_id][0]/self.rms[chan_id][2])
            plt.title(label,fontsize=10.0,va='center',ha='center')
            if not (count-1)%nc:
                plt.ylabel('mm',fontsize=10)
            if (count-1)/nc == nl-1 or nchan+nc > ntot:
                plt.xlabel('time, sec',fontsize=10) 
            plt.grid()
            
            # Map
            m = Basemap(projection='ortho',lat_0=sacdata.evla,lon_0=sacdata.evlo,resolution='c')
            pos  = ax.get_position().get_points()
            W  = pos[1][0]-pos[0][0] ; H  = pos[1][1]-pos[0][1] ;        
            ax2 = plt.axes([pos[1][0]-W*0.38,pos[0][1]+H*0.01,H*1.08,H*1.00])
            m.drawcoastlines(linewidth=0.5,zorder=900)
            m.fillcontinents(color='0.75',lake_color=None)
            m.drawparallels(np.arange(-60,90,30.0),linewidth=0.2)
            m.drawmeridians(np.arange(0,420,60.0),linewidth=0.2)
            m.drawmapboundary(fill_color='w')
            xs,ys = m(coords[:,1],coords[:,0])
            xr,yr = m(sacdata.stlo,sacdata.stla)
            xc,yc = m(sacdata.evlo,sacdata.evla)
            m.plot(xs,ys,'o',color=(1.00000,  0.74706,  0.00000),ms=4.0,alpha=1.0,zorder=1000)
            m.plot([xr],[yr],'o',color=(1,.27,0),ms=8,alpha=1.0,zorder=1001)
            m.scatter([xc],[yc],c='b',marker=(5,1,0),s=120,zorder=1002)                
            
            plt.axes(ax)
            count += 1
            nchan += 1
        ofic = 'page_W_%02d.pdf'%(pages)
        print(ofic)
        fig.set_rasterized(rasterize)
        plt.suptitle('CMT3D,    p %d/%d'%(pages,npages), fontsize=16, y=0.95)
        pp.savefig(orientation='landscape')
        plt.close()
        pp.close()       

        # All done
        return

    def rmsscreening(self,th=5.0):
        '''
        RMS screening
        '''

        # Check rms is assigned
        assert self.rms is not None, 'rms must be assigned'

        # Perform screening
        chan_ids = deepcopy(self.chan_ids)
        for chan_id in chan_ids:
            if self.rms[chan_id][0]/self.rms[chan_id][2] >= th:
                del self.data[chan_id]
                del self.gf[chan_id]
                self.chan_ids.remove(chan_id)

        # All done
        return

    def wchanidlst(self,f_name):
        '''
        Write list of channels in f_name
        '''
        fid = open(f_name,'wt')
        for chan_id in self.chan_ids:
            fid.write('%s\n'%(chan_id))
        fid.write
        # All done
        return


    def copy(self):
        '''
        Returns a copy of the sac object
        '''
        # All done
        return deepcopy(self)               
