'''
Simple class for CMT inversion problems

Written by Z. Duputel and L. Rivera, May 2015
'''


# Externals
from scipy import signal
from scipy import linalg
from copy  import deepcopy
import numpy as np
import os

# Personals
from sacpy import sac
from .cmt  import cmt

TRACES_PLOTPARAMS = {'backend': 'pdf', 'axes.labelsize': 10,
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
        
class cmtproblem(object):

    def __init__(self):
        '''
        Constructor
        '''

        # List of channel ids
        self.chan_ids = []
        # Data and Green's functions
        self.data = None  # data dictionary
        self.synt = None  # synthetic dictionary
        self.gf   = None  # Green's functions sac object      
        self.D = None     # Data vector
        self.G = None     # Green's function matrix
        # RMS misfit
        self.global_rms = None # Global RMS misfit
        self.rms  = None       # RMS misfit per station
        # CMT object
        self.cmt = cmt()
        # Deconvolution
        self.duration    = None
        # Tapering
        self.taper       = False # Taper flag
        self.taper_n     = None  # Taper width 
        self.taper_left  = None  # Taper (left part)
        self.taper_right = None  # Taper (right part)
        # All done

    def cmtinv(self,zero_trace=True,scale=1.):
        '''
        Perform CMTinversion
        Model will be 
        '''

        assert self.D is not None, 'D must be assigned before cmtinv'
        assert self.G is not None, 'G must be assigned before cmtinv'

        # Zero trace
        if zero_trace:
            G = np.empty((self.D.size,5))
            for i in range(2):
                G[:,i] = self.G[:,i] - self.G[:,2]
            for i in range(3):
                G[:,i+2] = self.G[:,i+3]
        else:
            G = self.G

        # Compute GtG
        GtG = G.T.dot(G)

        # Compute Cm
        Cm = linalg.inv(GtG)
        
        # Moment tensor
        Dp = G.T.dot(self.D)
        m  = Cm.dot(Dp)

        # Scale moment tensor
        m *= scale

        # Populate cmt attribute
        self.cmt.MT = np.zeros((6,))
        if zero_trace:
            self.cmt.MT[5] = m[4]
            self.cmt.MT[4] = m[3]
            self.cmt.MT[3] = m[2]
            self.cmt.MT[0] = m[0]
            self.cmt.MT[1] = m[1]
            self.cmt.MT[2] = -m[0] -m[1]
        else:
            self.cmt.MT = m.copy()

        S  = G.dot(m)
        S /= scale
        res = self.D - S
        res = np.sqrt(res.dot(res)/res.size)
        nD  = np.sqrt(self.D.dot(self.D)/self.D.size)
        nS  = np.sqrt(S.dot(S)/S.size)
        self.global_rms = [res,nD,nS]
        
        # All done
        
    def buildD(self):
        '''
        Build D matrices from data dictionary
        '''

        self.D = []
        for chan_id in self.chan_ids:
            self.D.extend(self.data[chan_id].depvar)
        self.D = np.array(self.D)

        # All done

    def buildG(self):
        '''
        Build G matrices from data dictionary
        '''
        
        self.G = []
        for mt in self.cmt.MTnm:
            self.G.append([])
            for chan_id in self.chan_ids:
                self.G[-1].extend(self.gf[chan_id][mt].depvar)
        self.G = np.array(self.G).T
        
        # All done
            
        
    def preparedata(self,i_sac_lst,filter_coef=None,wpwin=False,swwin=None,taper_width=None,
                    o_dir=None,o_sac_lst=None):
        '''
        Prepare Data
        Args:
            * i_sac_lst: list of input data sac file
            * filter_coef: filter coefficients (optional)
            * wpwin: if True, use W-phase window
            * win_param: surface wave windowing (optional):
               - if wpwin=False: Surface wave type windowing
                     Time window from dist/swwin[0] to dist/swwin[1]
                     if swwin[2] exists, it is a minimum window length
               - if wpwin=True: W-phase time windowing
                     Time window from P arrival time (Ptt) + swwin*gcarc
            * taper_width: Apply taper to data
            * o_dir: Output directory for filtered data (optional)
        '''

        # We assume delta=1
        delta = 1.
        
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

        # Hanning taper
        self.taper = False
        if taper_width is not None:
            self.taper   = True
            self.taper_n = int(taper_width/delta)
            H  = np.hanning(2*self.taper_n)
            zeros = np.zeros((self.taper_n))
            self.taper_left  = np.append(zeros,H[:self.taper_n])
            self.taper_right = np.append(H[self.taper_n:],zeros)
            self.taper_n *= 2
                     
        # Instantiate data dict
        self.data = {}
        self.chan_ids = []

        # Instantiate sacpy.sac
        data_sac = sac()        

        # Loop over sac_files
        for ifile in ifiles:

            # Read sac file            
            data_sac.rsac(ifile)
            assert np.round(data_sac.delta,3)==delta, 'data should be sampled at 1sps'

            # Filter
            if filter_coef is not None:
                assert len(filter_coef)==2, 'Incorrect filter_coef, must include [b,a]'
                b,a = filter_coef
                data_sac.depvar = signal.lfilter(b,a,data_sac.depvar)

            # Output directory for unwindowed filtered sac data
            if o_dir is not None:
                ofile=os.path.join(o_dir,os.path.basename(ifile)+'.filt')
                o_lst.write('%s %s\n'%(ofile,data_sac.id))
                data_sac.wsac(ofile)
                
            # Time-window            
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
                    
            if wpwin or swwin is not None:
                ib = int((tbeg+data_sac.o-data_sac.b)/data_sac.delta)
                ie = ib+int((tend-tbeg)/data_sac.delta)
                t    = np.arange(data_sac.npts)*data_sac.delta+data_sac.b-data_sac.o
                assert ib>=0, 'Incomplete data for %s (ie<0)'%(ifile)                
                assert ie<=data_sac.npts,'Incomplete data for %s (ie>npts)'%(ifile)
                if self.taper:
                    ib -= self.taper_n
                    ie += self.taper_n
                data_sac.depvar = data_sac.depvar[ib:ie+1].copy()
                if self.taper:
                    data_sac.depvar[:self.taper_n]  *= self.taper_left
                    data_sac.depvar[-self.taper_n:] *= self.taper_right
                data_sac.t[0]   = tbeg+data_sac.o
                data_sac.b      = t[ib]+data_sac.o
                data_sac.e      = t[ie]+data_sac.o
                data_sac.npts   = len(data_sac.depvar)

            # Populate the dictionary
            self.data[data_sac.id] = data_sac.copy()
            self.chan_ids.append(data_sac.id)

        if o_dir is not None:
            o_lst.close()
        
        # All done            
                
        
    def preparekernels(self,GF_names,stf=None,delay=0.,filter_coef=None,baseline=0,left_taper=False,wpwin=False,scale=1.):
        '''
        Prepare Green's functions
        Args:
            * GF_names : dictionary of GFs names
            * stf : moment rate function (optionnal)
                - can be a single array used for all stations
                - can be a dictionary with one stf per channel id
            * delay: time-shift (in sec, optional)
                - can be a single value used for all stations
                - can be a dictionary with one delay per channel id
            * filter_coef: filter coefficient (for scipy.signal.lfilter, optional)
            * baseline : number of samples to remove baseline (default: no baseline)
            * left_taper: if True, apply left taper over baseline (optional)
            * scale: scaling factor for all GFs (optional)
        '''

        # sacpy.sac instantiation
        gf_sac = sac()
        
        # gf dictionary
        self.gf = {}

        # Assign cmt delay
        if not isinstance(delay,dict):
            self.cmt.ts = delay
            if stf is not None:
                self.cmt.hd = (len(stf)-1)*0.5
            else:
                self.cmt.hd = delay
            
        # Loop over channel ids
        for chan_id in GF_names.keys():
            self.gf[chan_id] = {}
        
            # Loop over moment tensor components
            for m in self.cmt.MTnm:

                # Read GF sac file
                gf_sac.rsac(GF_names[chan_id][m])
                assert np.round(gf_sac.delta,3)==1.0, 'GFs should be sampled at 1sps'

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
                    if isinstance(stf,np.ndarray) or isinstance(stf,list): 
                        gf_sac.depvar=np.convolve(gf_sac.depvar,stf,mode='same')
                        
                    else:
                        assert chan_id in stf, 'No channel id %s in stf'%(chan_id)
                        gf_sac.depvar=np.convolve(gf_sac.depvar,stf[chan_id],mode='same')

                # Time-shift
                if isinstance(delay,dict):
                    assert chan_id in delay, 'No channel id %s in delay'%(chan_id)
                    gf_sac.b += delay[chan_id]                    
                else:
                    gf_sac.b += delay
                        
                # Filter 
                if filter_coef is not None:
                    assert len(filter_coef)==2, 'Incorrect filter_coef, must include [b,a]'
                    b,a = filter_coef
                    gf_sac.depvar = signal.lfilter(b,a,gf_sac.depvar)

                # Time-window matching data
                if self.data is not None:
                    assert chan_id in self.data, 'No channel id %s in data'%(chan_id)
                    data_sac = self.data[chan_id]                    
                    b    = data_sac.b - data_sac.o
                    npts = data_sac.npts                    
                    assert np.round(data_sac.delta,3)==1.0, 'data should be sampled at 1sps'

                    t    = np.arange(gf_sac.npts)*gf_sac.delta+gf_sac.b-gf_sac.o
                    if wpwin:
                        t0 = data_sac.t[0]-data_sac.o
                    else:
                        t0 = data_sac.b-data_sac.o
                    ib = int((t0-gf_sac.b)/gf_sac.delta)
                    ie = ib+data_sac.npts                    

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

    def calcsynt(self,scale=1.,stf=None):
        '''
        Compute synthetics. If data exists, will also compute rms
        '''

        # Check that gf and cmt are assigned
        assert self.gf is not None, 'Green s function must be computed'
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
            for m in range(6):
                MTnm=self.cmt.MTnm[m]
                self.synt[chan_id].depvar += self.cmt.MT[m]*self.gf[chan_id][MTnm].depvar*scale
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

    def deconv_projlandweber(self,duration=None,nit=1000,nit_min=100,fwahm=5.):
        '''
        
        '''
        # Check that data and synt are available
        assert self.data is not None, 'data not available'
        assert self.synt is not None, 'synt not available'

        # Smoothing gaussian
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
            G  = signal.gaussian(32,std=6.0)
            iG = G.sum()*data.delta
            G /= iG
            fG = np.fft.fft(G,n=npts_fft)
            fdata       = np.fft.fft(data.depvar,n=npts_fft)
            fdata       = fdata*fG
            data.depvar = np.real(np.fft.ifft(fdata))[:self.synt[chan_id].npts]
            
            
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
                    if it>=nit_min and (np.absolute(eps_p-eps)/eps_p<0.0001 or delta<0.0001):
                        break
                
                # Update stf_p and eps_p
                stf_p = stf.copy()
                eps_p = eps

            ## Convolve with gaussian
            #gaussw  = signal.gaussian(npts,std=sigma_avg)
            #igaussw = gaussw.sum()*data.delta
            #gaussw /= igaussw
            #stf = np.convolve(stf,gaussw,mode='same')
            
            # Fill out dictionary
            self.stf[chan_id]=stf.copy()

            
        # All done

    def traces(self,length=3000,i_sac_lst=None,show_win=False):
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
        pp = matplotlib.backends.backend_pdf.PdfPages('traces.pdf')        
        nc = 3
        nl = 5
        perpage = nc * nl

        coords = []
        for chan_id in self.chan_ids:
            sacdata = self.data[chan_id].copy()
            coords.append([sacdata.stla,sacdata.stlo,sacdata.az,sacdata.dist])
        coords = np.array(coords) 
        
        # Loop over channel ids
        ntot   = len(self.chan_ids)
        npages = np.ceil(float(ntot)/float(perpage))
        nchan = 1
        count = 1        
        pages = 1
        for chan_id in self.chan_ids:
            if i_sac_lst is not None:                
                sacdata.rsac(i_sac[chan_id])
            else:
                sacdata = self.data[chan_id].copy()
            sacsynt = self.synt[chan_id].copy()
            if count > perpage:
                plt.suptitle('CMT3D,   p %d/%d'%(pages,npages), fontsize=16, y=0.95)
                ofic = 'page_W_%02d.pdf'%(pages)
                print(ofic)
                fig.set_rasterized(True)
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
            plt.plot(t1,sacdata.depvar*1000.,'k')
            plt.plot(t2,sacsynt.depvar*1000.,'r-')
            # Axes limits
            #plt.xlim([t1[0],t1[-1]+(t1[-1]-t1[0])*0.4])
            t0 = t1[0] - 150.0
            if t0<0.:
                t0 = 0.
            plt.xlim([t0,t0+length*sacdata.delta])
            a    = np.absolute(sacsynt.depvar).max()*1000.
            ymin = -1.1*a
            ymax =  1.1*a
            ylims = [ymin,ymax]
            plt.ylim(ylims)
            if show_win:
                tbeg = self.data[chan_id].b - self.data[chan_id].o
                tend = self.data[chan_id].e - self.data[chan_id].o
                plt.plot([tbeg,tend],[0,0],'ro')
                t3 = np.arange(self.data[chan_id].npts,dtype='double')*self.data[chan_id].delta + self.data[chan_id].b - self.data[chan_id].o
                #plt.plot(t3,self.data[chan_id].depvar*1000.,'b-')

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
            label = label+' %.1f'%(self.rms[chan_id][0]/self.rms[chan_id][2])
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
        fig.set_rasterized(True)
        plt.suptitle('CMT3D,    p %d/%d'%(pages,npages), fontsize=16, y=0.95)
        pp.savefig(orientation='landscape')
        plt.close()
        pp.close()       

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

    def wchanidlst(self,f_name):
        '''
        Write list of channels in f_name
        '''
        fid = open(f_name,'wt')
        for chan_id in self.chan_ids:
            fid.write('%s\n'%(chan_id))
        fid.write
        # All done


    def copy(self):
        '''
        Returns a copy of the sac object
        '''
        # All done
        return deepcopy(self)               
