from hmm_dp import var_inf_line, var_inf_tag
from i2b2line import i2b2Line, i2b2Corpus
from param import HyperParam, VarParam
from map_inf import map_inf, map_inf_bf, map_inf_simple
import numpy as np

def test_case_1():
    tag_num=3
    vocab_num=10    
    hParam = HyperParam(vocab_num, tag_num, 2)
    vParam=VarParam(hParam, 'gamma')
    X=[0,1,1,1,2,1,6,4,8,9]
    S=[0,1,1,1,1,1,2,1,1,1]
    l1=i2b2Line(X, S, None, None, None, None)
    X=[0,1,1,1,3,1,6,4,8,9]
    S=[0,1,1,1,1,1,1,1,1,1]
    l2=i2b2Line(X, S, None, None, None, None)    
    lines=[l1,l2]
    (vParam, elbo) = var_inf_tag(lines, hParam, vParam)
    s1=map_inf_bf(l1, hParam, vParam)    
    s2=map_inf_bf(l2, hParam, vParam)
    print s1
    print s2

test_case_1()