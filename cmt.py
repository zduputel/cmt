'''
Simple classes for CMT inversion problems

Written by Z. Duputel and L. Rivera, May 2015
'''

# Externals
import numpy as np

class cmt(object):
    '''
    A simple cmt class
    '''
    def __init__(self,pdeline=None,evid=None,ts=None,hd=None,lon=None,lat=None,dep=None,MT=None):
        '''
        Constructor
        Args:
            * pdeline: PDE line
            * evid: event id
            * ts: time-shift
            * hd: half-duration
            * lon: centroid longitude
            * lat: centroid latitude
            * dep: centroid depth
            * MT: moment tensor
        '''
        self.pdeline = pdeline
        self.evid = evid
        self.MTnm = ['rr','tt','pp','rt','rp','tp']
        self.MT  = None
        self.ts  = ts
        self.hd  = hd
        self.lon = lon
        self.lat = lat
        self.dep = dep
        # All done
        
    def rcmtfile(self,cmtfil):
        '''
        Reads CMTSOLUTION file
        Args:
           * cmtfil: CMTSOLUTION filename
        '''
        L  = open(cmtfil).readlines()
        self.pdeline = L[0].strip('\n')
        self.evid  = L[1].strip().split(':')[1]
        self.ts    = float(L[2].strip().split(':')[1])
        self.hd    = float(L[3].strip().split(':')[1])
        self.lat   = float(L[4].strip().split(':')[1])
        self.lon   = float(L[5].strip().split(':')[1])
        self.dep   = float(L[6].strip().split(':')[1])        
        if len(L)>=13:
            self.MT = np.zeros((6,))
            for i in range(6):
                self.MT[i]=float(L[i+7].strip().split(':')[1])
        # All done

    def wcmtfile(self,cmtfil):
        '''
        Writes CMTSOLUTION file
        Args:
           * cmtfil: CMTSOLUTION filename
        '''
        fid = open(cmtfil, 'wt')
        fid.write('%s\n'%self.pdeline)
        fid.write('event name:%15s\n' % self.evid)
        fid.write('time shift:%12.4f\n'   % self.ts)
        fid.write('half duration:%9.4f\n' % self.hd)
        fid.write('latitude:%14.4f\n'     % self.lat)
        fid.write('longitude:%13.4f\n'    % self.lon)
        fid.write('depth:%17.4f\n'        % self.dep)
        if not np.equal(self.MT,None).any():
            for i in range(6):
                fid.write('M%s: %18.6e\n'%(self.MTnm[i],self.MT[i]))
        fid.close()
        # All done

    def plot(self,npx=250,colors=[[1.,0.,0.],[1.,1.,1.]],ax=None):
        '''
        Compute mechanism
        Args:
           * npx: number of points
           * colors: RGB colors for compressional and tensional quadrants
        '''
        
        # Mecanism
        meca = np.zeros((npx,npx))
    
        # x,y coordinates
        X,Y = np.meshgrid(np.linspace(-1,1,npx),np.linspace(-1,1,npx))
    
        # distance, angles
        r  = np.sqrt(X*X+Y*Y)
        ta = 2. * np.arcsin(r/np.sqrt(2))
        az = np.arctan2(X,Y)
        
        # gamma vector
        ga = np.zeros((3,npx,npx))
        ga[0] = -np.cos(ta)
        ga[1] = -np.sin(ta)*np.cos(az)
        ga[2] =  np.sin(ta)*np.sin(az)

        # P-wave amplitude
        MT = self.MT
        amp = MT[0]*ga[0]*ga[0] + MT[1]*ga[1]*ga[1] + MT[2]*ga[2]*ga[2]
        amp += 2*(MT[3]*ga[0]*ga[1] + MT[4]*ga[0]*ga[2] + MT[5]*ga[1]*ga[2])

        # Polarity
        i0,j0 = np.where(amp> 0.)
        i1,j1 = np.where(amp<=0.)
        i2,j2 = np.where(r>=0.98)
        i3,j3 = np.where(r>1.)
        pol = np.zeros((npx,npx,3))
        pol[i0,j0] = colors[0]
        pol[i1,j1] = colors[1]
        pol[i2,j2] = [0.,0.,0.]
        pol[i3,j3] = [1.,1.,1.]
        pol=np.ma.masked_where(pol>=2.,pol)[::-1,:]
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax  = fig.add_subplot(111)
        ax.imshow(pol)
        ax.set_axis_off()
        #return x,y,r,amp,pol
