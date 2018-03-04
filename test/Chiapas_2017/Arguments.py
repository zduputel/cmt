import numpy as np


# CMT Model parameters
time_shift = 22.0 # Centroid time-shift
i_cmt_file = 'CMTSOLUTION'
o_cmt_file = 'o_CMTSOLUTION'

# Green's function directory (Green's functions are here filtered and convolved with an STF)
GF_DIR = './GFs/H050.0'
GF_M0  = 1.0e28 # Scalar moment used to compute Green's functions

# Data parameters (waveforms are here filtered in the same passband as Green's functions
i_sac_lst = 'i_sac_lst.txt'
wpwin = True # if True, time-window from P-travel time to P-travel time + swwin*Delta 
swwin = 35.  # where Delta is the epicentral distance in degrees (gcarc in data sac headers)
             # P-travel time should be defined as t0 in data sac headers

# Time-windows different from the default (set above)
dcwin = {'CU_TGUH_00_BHZ': [0.,310.], 'II_JTS_00_BHZ':  [0.,600.], 'MX_ZAIG_--_BHZ': [0.,240.],
         'MX_ZAIG_--_BHN': [0.,240.], 'MX_ZAIG_--_BHE': [0.,240.], 'IU_TUC_00_BHN':  [0.,500.],
         'IU_TUC_00_BHE':  [0.,400.], 'MX_HPIG_--_BHZ': [0.,400.], 'IU_SDV_10_BHZ':  [0.,600.],
         'IU_SDV_10_BHZ':  [0.,600.], 'CI_GSC_--_BHE':  [0.,750.], 'CI_SNCC_--_BHE': [0.,750.],         
         'CI_ISA_--_BHZ':  [0.,750.], 'IU_RSSD_10_BHE': [0.,550.], 'IU_RSSD_10_BHN': [0.,800.],
         'BK_SAO_00_BHE':  [0.,750.], 'BK_SAO_00_BHE':  [0.,600.], 'BK_MCCM_00_BHZ': [0.,800.],
         'IU_COR_00_BHE':  [0.,850.], 'II_FFC_00_BHN':  [0.,1000.]}


# plot_traces arguments
o_trace_dir = 'SYNTH_SEM'
