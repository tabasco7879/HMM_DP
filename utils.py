#! /usr/bin/python

''' several useful functions '''
import numpy as np
from scipy import special as sp

def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def log_sum(log_a, log_b):
    ''' we know log(a) and log(b), compute log(a+b) '''
    v = 0.0;
    if (log_a < log_b):
        v = log_b+np.log(1 + np.exp(log_a-log_b))
    else:
        v = log_a+np.log(1 + np.exp(log_b-log_a))
    return v


def argmax(x):
    ''' find the index of maximum value '''
    n = len(x)
    val_max = x[0]
    idx_max = 0

    for i in range(1, n):
        if x[i]>val_max:
            val_max = x[i]
            idx_max = i

    return idx_max

def expect_log_dirichlet(alpha, ignore_zero=False):
    if not ignore_zero:
        if (len(alpha.shape) == 1):
            return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
        return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])
    else:
        elog_dir = np.zeros(alpha.shape)
        if (len(alpha.shape) == 1):
            elog_dir[1:] = (sp.psi(alpha[1:]) - sp.psi(np.sum(alpha[1:])))
        else:
            elog_dir[:,1:] = (sp.psi(alpha[:,1:]) - sp.psi(np.sum(alpha[:,1:], 1))[:, np.newaxis])
        return elog_dir

def expect_log_sticks(sticks):
    '''
        beta(1, \alpha_1)
        sticks is np.array([\alpha_i_1,...], [\alpha_i_2]]) where the first row
        is the first parameter of beta and the second row is the second parameter
        for beta. Each column is a pair of parameters for beta.
        notice that there is p-1 sticks input and return p log sticks
    '''
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n - 1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks