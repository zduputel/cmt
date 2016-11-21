'''
Simple classes for CMT inversion problems

Written by Z. Duputel and L. Rivera, May 2015
'''


# Externals
from scipy import signal
import numpy as np
import os

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



def filtcoeffromimaster(i_master='./i_master',delta=1.):
    '''
    Read filter parameters in i_master
    '''
    
    iconfig = parseConfig(i_master)
    bp = np.array([float(iconfig['filt_cf1']),float(iconfig['filt_cf2'])])
    order = int(iconfig['filt_order'])
    bfilter,afilter = signal.butter(order,bp*2*delta,'bandpass')
    # All done
    return [bfilter,afilter]

        
