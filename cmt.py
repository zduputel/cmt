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

