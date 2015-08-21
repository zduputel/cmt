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

        
    def M0(self):
        '''
        Calculate M0 (using Dahlen and Tromp convention)
        '''
        assert self.MT is not None, 'MT must be assigned'
        MT = self.MT
        M0 =  MT[0]*MT[0] + MT[1]*MT[1] + MT[2]*MT[2]
        M0 += 2*(MT[3]*MT[3] + MT[4]*MT[4] + MT[5]*MT[5])
        M0 = np.sqrt(M0/2.)
        # All done
        return M0

    
    def Mw(self):
        '''
        Calculate Mw
        '''
        M0 = self.M0()
        Mw = 2./3.*(np.log10(M0)-16.1)
        # All done        
        return Mw

    
    def fullMT(self):
        '''
        Returns the full moment tensor
        '''

        # Check that MT is assigned
        assert self.MT is not None, 'MT must be assigned'

        # Create and fill the 3x3 array
        TM = np.zeros((3,3))
        TM[0,0] = self.MT[0]
        TM[1,1] = self.MT[1]
        TM[2,2] = self.MT[2]
        TM[0,1] = self.MT[3]
        TM[0,2] = self.MT[4]
        TM[1,2] = self.MT[5]
        TM[1,0] = TM[0,1]
        TM[2,0] = TM[0,2]
        TM[2,1] = TM[1,2]

        # All done
        return TM

    
    def v2str(self,vn,vs,tol=0.001):
        '''
        Returns strike dip rake of the plane with normal vn and slip vector vs
        '''

        # Check that normal is upward
        if vn[0]<0.:
            vn *= -1.
            vs *= -1.

        # Compute strike, dip, rake
        if vn[0] > 1. - tol:       # Horizontal plane
            strike = 0.
            dip    = 0.
            rake   = np.arctan2(-vs[2],-vs[1])
        elif vn[0] < tol:          # Vectical plane
            strike = np.arctan2(vn[1],vn[2])
            dip    = np.pi/2.
            rake   = np.arctan2(vs[0], -vs[1]*vn[2] + vs[2]*vn[1])
        else:                      # Oblique plane
            strike = np.arctan2(vn[1], vn[2])
            dip    = np.arccos(vn[0])
            rake   = np.arctan2((-vs[1]*vn[1] - vs[2]*vn[2]),
                                (-vs[1]*vn[2] + vs[2]*vn[1])*vn[0])

        # Rad 2 Deg
        rad2deg = 180./np.pi
        strike *= rad2deg
        if (strike<0.):
            strike += 360.
        dip  *= rad2deg
        rake *= rad2deg

        # All done
        return strike,dip,rake

    
    def nodalplanes(self):
        '''
        Returns strike, dip, rake of each plane
        '''

        # Get the full moment tensor
        MT = self.fullMT()

        # Compute eigenvalues/eigenvectors and sort them
        [di,vi] = np.linalg.eig(MT)
        i  = np.argsort(di)[::-1]
        di = di[i]
        vi = vi[:,i]

        # Normal/Slip vectors
        v1 = (vi[:,0] + vi[:,2])/np.sqrt(2.)
        v2 = (vi[:,0] - vi[:,2])/np.sqrt(2.)

        # Nodal planes
        strike1,dip1,rake1 = self.v2str(v1,v2)
        strike2,dip2,rake2 = self.v2str(v2,v1)

        # All done
        return [[strike1,dip1,rake1],[strike2,dip2,rake2]]


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
