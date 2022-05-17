import numpy as np

def ea2rm(E_angles):  
    '''
    Convert Euler angles to rotation matrix (using Goldstein convention)
    '''
    R = np.zeros((3,3))
    phi, theta, psi = E_angles
    R[0,0] =  np.cos(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.sin(psi)
    R[0,1] =  np.cos(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.sin(psi)
    R[0,2] =  np.sin(psi)*np.sin(theta)
    R[1,0] = -np.sin(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.cos(psi)
    R[1,1] = -np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.cos(psi)
    R[1,2] =  np.cos(psi)*np.sin(theta)
    R[2,0] =  np.sin(theta)*np.sin(phi)
    R[2,1] = -np.sin(theta)*np.cos(phi)
    R[2,2] =  np.cos(theta)
    # All done
    return R

def f_ea(x): 
    '''
    Non-normalized homogeneous PDF in SO(3) for
    the Euler angles 
    '''
    return np.sin(x[1])

def random_ea(Nrand,avoid_duplicates=True):
    '''
    Random generator of euler angles using a simple metropolis rule
    Arg:
        Nrand: Number of random euler angles
    '''
    # Limit of euler angles 
    lims = [2.*np.pi,np.pi,2.*np.pi]
    v = [np.random.rand(3)*lims]; probi = f_ea(v[-1])
    count = 0
    while count<Nrand:
        x = np.random.rand(3)*lims # Simple uniform proposal PDF
        probj  = f_ea(x) # PDF to sample
        accept = False
        if probj > probi: accept = True
        else:
            if np.random.rand() < probj/probi: accept = True
        if accept: # Accept
            v.append(x)
            probi = probj
            count += 1
        elif avoid_duplicates: # Reject: regenerate until Accept
            continue
        else:      # Reject: duplicate
            v.append(v[-1])
            count += 1
    return np.array(v)

def random_rm(Nrand,avoid_duplicates=True):
    '''
    Randomly generate rotation matrices for Focal mechanism generation
    Arg:
        Nrand: Number of random euler angles
        avoid_duplicates: if True (default) will avoid duplicates in the random generation
    '''
    # Random generation of euler angles
    ea = random_ea(Nrand,avoid_duplicates=avoid_duplicates)
    # Conversion to random matrices
    rm = []
    for i in range(Nrand): rm.append(ea2rm(ea[i]))
    # All done
    return rm


