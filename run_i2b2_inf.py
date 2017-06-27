from hmm_dp import var_inf_tag
from i2b2line import i2b2Line, i2b2Corpus
from param import HyperParam, VarParam
from map_inf import map_inf_simple, map_inf_bf
import time
import numpy as np

def run_i2b2_var_inf():
    tag_num=9
    vocab_num=11937
    corpus=i2b2Corpus('i2b2_data.txt')
    hParam = HyperParam(vocab_num, tag_num, len(corpus.Lines))
    
    best_elbo=-1.0e100
    
    for i in range(1):
        vParam = VarParam(hParam, 'gamma')
        #vParam.save(hParam, "i2b2_param_init.txt")
        #vParam=VarParam(hParam, 'file', 'i2b2_param_init.txt')
        lines = corpus.Lines
        (vParam, elbo) = var_inf_tag(lines, hParam, vParam)
        vParam.save(hParam, "i2b2_param.txt", elbo)
        if (elbo>best_elbo):
            best_elbo=elbo
            best_vParam=vParam

    #best_vParam.save(hParam, "i2b2_param_best.txt", best_elbo)

def run_i2b2_map_inf_valid():
    tag_num=9
    vocab_num=10494
    corpus=i2b2Corpus('test.txt')
    hParam = HyperParam(vocab_num, tag_num, len(corpus.Lines))
    vParam=VarParam(hParam, 'file', 'i2b2_param.txt')
    
    tp=[0]*tag_num; fp=[0]*tag_num; fn=[0]*tag_num

    with open('valid_log.txt', 'w') as f:
        for line in corpus.Lines:       
            seq0=line.S
            start_time=time.clock();
            if (check_complexity(line, hParam, vParam)<200):
                (elbo1, seq1)=map_inf_bf(line, hParam, vParam)
            else:
                (elbo1, seq1)=map_inf_simple(line, hParam, vParam)
            time_spent=time.clock() - start_time
            #line.S=seq0
            #elbo0=var_inf_only_line(line, hParam, vParam)
            #print 'elbo1 (%f) elbo0 (%f)'%(elbo1, elbo0)            
            error=0
            log='file(%s) line(%s)\n'%(line.File_ID, line.Line_ID)
            for i in range(len(line.X)):
                s0=seq0[i]
                s1=seq1[i]
                x=line.X[i]
                if s0!=s1:
                    fn[s0]+=1
                    fp[s1]+=1
                    error+=1                    
                else:
                    tp[s0]+=1

            if error>0:
                f.write(log)
                log='\t'.join([str(x) for x in line.X])
                f.write(log+'\n')
                log='\t'.join([str(x) for x in seq0])
                f.write(log+'\n')
                log='\t'.join([str(x) for x in seq1])
                f.write(log+'\n')
                f.flush()
            print 'file(%s) line(%s) time:(%f) error:(%d)'%(line.File_ID, line.Line_ID, time_spent, error)

        line='\t'.join([str(x) for x in tp])
        f.write(line+'\n')
        line='\t'.join([str(x) for x in fp])
        f.write(line+'\n')
        line='\t'.join([str(x) for x in fn])
        f.write(line+'\n')

def run_i2b2_map_inf():
    tag_num=9
    vocab_num=10494
    corpus=i2b2Corpus('test.txt')
    hParam = HyperParam(vocab_num, tag_num, len(corpus.Lines))
    vParam=VarParam(hParam, 'file', 'i2b2_param.txt')

    with open('i2b2_result.txt', 'w') as f:            
        for line in corpus.Lines:       
            seq0=line.S
            start_time=time.clock();
            if (check_complexity(line, hParam, vParam)<200):
                (elbo1, seq1)=map_inf_bf(line, hParam, vParam)
            else:
                (elbo1, seq1)=map_inf_simple(line, hParam, vParam)
            time_spent=time.clock() - start_time
            seq2=[]
            for s1 in seq1:
                if (s1==3): # name
                    s2=4 # doctor
                elif (s1==4): # age
                    s2=6 
                elif (s1==5): # location
                    s2=9 # street
                elif (s1==6): # profession
                    s2=15
                elif (s1==7): # contact
                    s2=16 # phone
                elif (s1==8): # id
                    s2=22 # idnum
                else:
                    s2=s1
                seq2.append(s2)
            result_line='%s %s'%(line.File_ID, line.Line_ID)
            for (x, s1, s2, s, e) in zip(line.X, seq1, seq2, line.Start, line.End):
                if (x==-1): continue
                result_line+=' %d:%d:%d:%d:%d'%(x,s1,s2,s,e)
            f.write(result_line+'\n')
            f.flush()
            print 'file(%s) line(%s) time:(%f)'%(line.File_ID, line.Line_ID, time_spent)

        

def tag_word_list():
    tag_num=9
    vocab_num=7684    
    hParam = HyperParam(vocab_num, tag_num, 1)
    vParam=VarParam(hParam, 'file', 'i2b2_param.txt')

    for i in range(hParam.num_of_sub_tags(1)):
        if(np.amax(vParam.phi_e1_sz[1][i,:])>1):
            top10=np.argsort(vParam.phi_e1_sz[1][i,:])[-10:][::-1]
            print top10

def check_complexity(line, hParam, vParam):
    complexity = 1
    for x in line.X[1:]:
        m=1
        for s in range(2, hParam.num_of_tags()):
            if np.amax(vParam.phi_e1_sz[s][:,x]) > 1.05:
                m+=1;
        complexity*=m
    return complexity

run_i2b2_var_inf()
#run_i2b2_map_inf()