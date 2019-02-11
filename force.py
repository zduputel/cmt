'''
Simple classes for CMT inversion problems

Written by Z. Duputel and L. Rivera, May 2015
'''

# Externals
import numpy as np
from copy     import deepcopy
from datetime import datetime

class force(object):
    '''
    A simple cmt class
    '''
    def __init__(self,pdeline=None,evid=None,ts=None,hd=None,lon=None,lat=None,dep=None,F=None,filename=None):
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
            * F: moment tensor
            * filename: if provided, will read parameters from a cmtsolution file
        '''
        # Assign cmt parameters
        self.pdeline = pdeline
        self.evid = evid
        self.Fnm = ['E','N','U']
        self.F   = F
        self.ts  = ts
        self.hd  = hd
        self.lon = lon
        self.lat = lat
        self.dep = dep
        
        # Read CMTSOLUTION file (if filename is provided)
        if filename is not None:
            self.rforcefile(filename)
            
        # All done
        return
        
    def rforcefile(self,forcefil,scale=1.):
        '''
        Reads FORCESOLUTION file
        Args:
           * forcefil: FORCESOLUTION filename
        '''
        L  = open(forcefil).readlines()
        self.pdeline = L[0].strip('\n')
        self.evid  = L[1].strip().split(':')[1]
        self.ts    = float(L[2].strip().split(':')[1])
        self.hd    = float(L[3].strip().split(':')[1])
        self.lat   = float(L[4].strip().split(':')[1])
        self.lon   = float(L[5].strip().split(':')[1])
        self.dep   = float(L[6].strip().split(':')[1])        
        if len(L)>=10:
            self.F = np.zeros((3,))
            for i in range(3):
                self.F[i]=float(L[i+7].strip().split(':')[1])*scale
        # All done

    def set_hypo(self,otime,hypo_lat,hypo_lon,hypo_dep):
        '''
        Set hypocenter and origin time
        Args:
            * otime: origin time (datetime.datetime object)
            * hypo_lat: hypocenter latitude (float)
            * hypo_lon: hypocenter longitude (float)
            * hypo_dep: hypocenter depth (float)
        '''
        # Set otime
        self.otime = otime
        # Set hypocenter
        self.hypo_lat=hypo_lat
        self.hypo_lon=hypo_lon
        self.hypo_dep=hypo_dep
        # All done
        return

    def set_hypofromPDE(self):
        '''
        Parses origin time and hypocenter coordinates from self.pdeline
        '''
        # Origin time
        items = self.pdeline[5:].strip().split()
        oyear  = int(items[0])
        omonth = int(items[1])
        oday   = int(items[2])
        ohour  = int(items[3])
        omin   = int(items[4])
        osec   = int(float(items[5]))
        omsec  = int((float(items[5])-float(osec))*1.0e6)
        otime = datetime(oyear,omonth,oday,ohour,omin,osec,omsec)
        # Hypocenter coordinates
        lat = float(items[6])
        lon = float(items[7])
        dep = float(items[8])
        # Set attributes
        self.set_hypo(otime,lat,lon,dep)
        # All done
        return

    def set_PDEfromhypo(self,mag=None):
        '''
        Build a pde line from hypocenter coordinates and origin time
        Args:
            * mag: magnitude to indicate in the PDE_line 
              (by default we estimate magnitude from MT components
               or set mag=0.0 if selt.MT is none)
        '''
        # Check inputs
        assert self.otime is not None, 'otime is not set'
        assert self.hypo_lat is not None, 'pde lat is not set'
        assert self.hypo_lon is not None, 'pde lon is not set'
        assert self.hypo_dep is not None, 'pde dep is not set'
        # Magnitude
        if mag is None:
            if self.MT is not None:
                mag = self.Mw()
            else:
                mag = 0.0 
        # Origin time
        o = self.otime
        s = o.second + o.microsecond * 1e-6
        ot = '%4d %2d %2d %2d %2d %5.2f'%(o.year,o.month,o.day,o.hour,o.minute,s)
        # Hypocenter
        lat = self.hypo_lat
        lon = self.hypo_lon
        dep = self.hypo_dep
        # Set the pde line
        hp = '%9.4f %9.4f %5.1f %3.1f %3.1f '%(lat,lon,dep,mag,mag)
        pl = ' PDE ' + ot + hp + self.pdeline[61:]
        self.pdeline = pl
        # All done
        return
        
    def wforcefile(self,forcefil,scale=1.):
        '''
        Writes FORCESOLUTION file
        Args:
           * forcefil: FORCESOLUTION filename
        '''
        fid = open(forcefil, 'wt')
        fid.write('%s\n'%self.pdeline)
        fid.write('event name:%15s\n' % self.evid)
        fid.write('time shift:%12.4f\n'   % self.ts)
        fid.write('half duration:%9.4f\n' % self.hd)
        fid.write('latitude:%14.4f\n'     % self.lat)
        fid.write('longitude:%13.4f\n'    % self.lon)
        fid.write('depth:%17.4f\n'        % self.dep)
        if not np.equal(self.F,None).any():
            for i in range(3):
                fid.write('M%s: %18.6e\n'%(self.Fnm[i],self.F[i]*scale))
        fid.close()
        # All done

    def Mcsf(self):
        '''
        Calculate Mcsf (using Dahlen and Tromp convention)
        '''
        assert self.F is not None, 'MT must be assigned'
        # Maximum force amplitude 
        Fmax = np.sqrt(self.F.dot(self.F))
        # Get Mcsf (see Kawakatsu, 1989)
        Mcsf=Fmax*self.hd*self.hd/np.pi*(2-np.sin(2*np.pi)/np.pi)
        # All done
        return Mcsf
        
    def copy(self):
        '''
        Returns a copy of the cmt object
        '''
        # All done
        return deepcopy(self)
