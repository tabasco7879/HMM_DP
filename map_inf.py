import numpy as np
import scipy.special as sp
from utils import log_normalize, expect_log_sticks
from i2b2line import i2b2Line
from hmm_dp import var_inf_line

def print_elbo_table(elbo_table):
    for i in range(len(elbo_table)):
        for j in range(len(elbo_table[i])):
            for k in range(len(elbo_table[i][j])):
                if not(elbo_table[i][j][k] == None):                
                    print i, j, k, elbo_table[i][j][k][0], elbo_table[i][j][k][1]

def map_inf(line, hParam, vParam):
    '''
        keep 2 token history m n^4
    '''
    #[length][s0][s1]=[elbo, seq, c]
    elbo_table = [[[None for k in range(hParam.num_of_tags())]
                 for j in range(hParam.num_of_tags())]
                    for i in range(len(line.X))]
    
    line.S = [0] * len(line.X)
    # initialization length 1
    line.Length = 2
    for s in range(1,hParam.num_of_tags()):
        line.S[line.Length - 1] = s
        elbo = var_inf_line(line, hParam, vParam, None, True)
        elbo_table[line.Length - 1][0][s] = [elbo, [0, s]]

    # initialization length 2
    line.Length = 3
    for s0 in range(1,hParam.num_of_tags()):
        for s1 in range(1,hParam.num_of_tags()):
            line.S[line.Length - 2] = s0
            line.S[line.Length - 1] = s1            
            elbo = var_inf_line(line, hParam, vParam, None, True)
            elbo_table[line.Length - 1][s0][s1] = [elbo, [0, s0, s1]]
       
    for l in range(3,len(line.X)):
        line.Length = l + 1        
        #by adding one
        for s2 in range(1,hParam.num_of_tags()):
            line.S[l] = s2
            for s0 in range(1,hParam.num_of_tags()):
                for s1 in range(1,hParam.num_of_tags()):                    
                    r = elbo_table[l - 1][s0][s1]                          
                    #print l-1, r[1], s0, s1
                    line.S[:l] = r[1]                                     
                    elbo = var_inf_line(line, hParam, vParam, None, True)
                    if (elbo_table[l][s1][s2] == None or elbo_table[l][s1][s2][0] < elbo):
                        seq = r[1] + [s2]
                        elbo_table[l][s1][s2] = [elbo, seq]
                        assert l + 1 == len(seq)
        #by adding two
        for s2 in range(1,hParam.num_of_tags()):
            for s1 in range(1,hParam.num_of_tags()):
                line.S[l] = s2
                line.S[l - 1] = s1
                for s_minus_one in range(0,hParam.num_of_tags()):
                    for s0 in range(1,hParam.num_of_tags()):
                        r = elbo_table[l - 2][s_minus_one][s0]
                        if not(r == None):
                            line.S[:l - 1] = r[1]
                            elbo = var_inf_line(line, hParam, vParam, None, True)
                            if (elbo_table[l][s1][s2] == None or elbo_table[l][s1][s2][0] < elbo):
                                seq = r[1] + [s1,s2]
                                elbo_table[l][s1][s2] = [elbo, seq]
                                assert l + 1 == len(seq)        

    max_elbo = -1e100
    l = len(line.X) - 1
    for s1 in range(1,hParam.num_of_tags()):
        for s2 in range(1,hParam.num_of_tags()):
            if (elbo_table[l][s1][s2][0] > max_elbo):
                max_elbo = elbo_table[l][s1][s2][0]
                max_seq = elbo_table[l][s1][s2][1]
    
    print_elbo_table(elbo_table)
    return max_elbo, max_seq

def map_inf_simple(line, hParam, vParam):
    '''
        keep 1 token history m n^2
    '''
    #[length][s0][s1]=[elbo, seq, c]
    elbo_table = [[None for k in range(hParam.num_of_tags())]                 
                    for i in range(len(line.X))]
    
    line.S = [0] * len(line.X)
    # initialization length 1
    line.Length = 2
    for s in range(1,hParam.num_of_tags()):
        x = line.X[line.Length - 1]
        # only compute if the state has ever emit the token
        if s==1 or np.amax(vParam.phi_e1_sz[s][:,x]) > 1.05:
            line.S[line.Length - 1] = s
            elbo = var_inf_line(line, hParam, vParam, None, True)
            elbo_table[line.Length - 1][s] = [elbo, [0, s]]
            
    if (len(line.X) > 2):
        for l in range(2,len(line.X)):
            line.Length = l + 1        
            #by adding one
            for s1 in range(1,hParam.num_of_tags()):
                x = line.X[l]
                # only compute if the state has ever emit the token
                if s1==1 or np.amax(vParam.phi_e1_sz[s1][:,x]) > 1.05:
                    line.S[l] = s1
                    for s0 in range(1,hParam.num_of_tags()):             
                        r = elbo_table[l - 1][s0]
                        if r != None:
                            line.S[:l] = r[1]
                            elbo = var_inf_line(line, hParam, vParam, None, True)
                            if (elbo_table[l][s1] == None or elbo_table[l][s1][0] < elbo):
                                seq = r[1] + [s1]
                                elbo_table[l][s1] = [elbo, seq]
                                assert l + 1 == len(seq)
    max_elbo = -1e100
    l = len(line.X) - 1
    for s in range(1,hParam.num_of_tags()):        
        if (elbo_table[l][s] != None and elbo_table[l][s][0] > max_elbo):
            max_elbo = elbo_table[l][s][0]
            max_seq = elbo_table[l][s][1]
    
    #print_elbo_table(elbo_table)
    return max_elbo, max_seq

def map_inf_bf(line, hParam, vParam):
    '''
        brute force method to make inference, n^m
    '''
    line.S = [0] * len(line.X)
    (max_elbo, max_seq) = map_inf_bf2(line, hParam, vParam, 1)
    return max_elbo, max_seq

def map_inf_bf2(line, hParam, vParam, length):
    max_elbo = -1e100
    for s in range(1, hParam.num_of_tags()):
        x = line.X[length]
        if s==1 or np.amax(vParam.phi_e1_sz[s][:,x]) > 1.05:
            line.S[length] = s
            if length == len(line.X) - 1:
                elbo = var_inf_line(line, hParam, vParam, None, True)
                seq = list(line.S)
            else:
                (elbo, seq) = map_inf_bf2(line, hParam, vParam, length + 1)
            if (elbo > max_elbo):
                max_elbo = elbo
                max_seq = seq
    return max_elbo, max_seq