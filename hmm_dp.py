import sys
import numpy as np
import scipy.special as sp
from utils import log_normalize, log_sum, argmax, expect_log_dirichlet, expect_log_sticks
from param import VarParam

def var_inf_line(line, hParam, vParam, dParam, inf_only=False):
             
    now_c = [np.ones(hParam.num_of_atoms(s)) \
        * (1.0 / hParam.num_of_atoms(s)) for s in line.S]    
    now_z = [np.zeros(hParam.num_of_sub_tags(s)) for s in line.S]
    
    eta_szs = [[np.ones((hParam.num_of_sub_tags(s0), \
            hParam.num_of_atoms(s1), \
            hParam.num_of_sub_tags(s1)))* (1.0/hParam.num_of_sub_tags(s1))
            for s1 in range(hParam.num_of_tags())] 
            for s0 in range(hParam.num_of_tags())]
      
    v_szs = [[np.zeros((hParam.num_of_sub_tags(s0), \
            2, hParam.num_of_atoms(s1) - 1))
            for s1 in range(hParam.num_of_tags())] 
            for s0 in range(hParam.num_of_tags())]

    for s0 in range(hParam.num_of_tags()):
        for s1 in range(hParam.num_of_tags()):
            if (s0 == 0 and s1 == 0): # from s_{-1} to s_{0}
                v_szs[s0][s1][0, 0] = 1
                v_szs[s0][s1][0, 1] = hParam.alpha_szs(s1)
            elif (s1>0):
                for z in range(hParam.num_of_sub_tags(s0)):
                    v_szs[s0][s1][z, 0] = 1
                    v_szs[s0][s1][z, 1] = hParam.alpha_szs(s1)

    log_sticks_v_szs = [[np.zeros((hParam.num_of_sub_tags(s0), \
            hParam.num_of_atoms(s1)))
            for s1 in range(hParam.num_of_tags())]
            for s0 in range(hParam.num_of_tags())]
    
    old_elbo = -1.0e100 
    iter = 0
    converge = 1.0

    d_eta_s = None
    d_v_szs_minus_one = None
    d_v_szs = None

    seq2=set()    
    for i in range(line.Length):
        s_minus_1=0 if i==0 else line.S[i-1]
        s0=line.S[i]        
        seq2.add((s_minus_1, s0))        
        
    while iter < hParam.max_iter_line() and (converge < 0.0 or converge > hParam.var_convergence()):        
        iter+=1

        # pre-computing        
        for (s_minus_1, s0) in seq2:
            if (s_minus_1 == 0 and s0 == 0):
                log_sticks_v_szs[s_minus_1][s0][0] = \
                    expect_log_sticks(v_szs[s_minus_1][s0][0])
            else:
                for z in range(hParam.num_of_sub_tags(s_minus_1)):
                    log_sticks_v_szs[s_minus_1][s0][z] = \
                        expect_log_sticks(v_szs[s_minus_1][s0][z])
        
        d_eta_szs = [[np.zeros((hParam.num_of_sub_tags(s0), \
            hParam.num_of_atoms(s1), \
            hParam.num_of_sub_tags(s1)))
            for s1 in range(hParam.num_of_tags())] 
            for s0 in range(hParam.num_of_tags())]
        
        d_v_szs = [[np.zeros((hParam.num_of_sub_tags(s0), \
            2, hParam.num_of_atoms(s1) - 1))
            for s1 in range(hParam.num_of_tags())] 
            for s0 in range(hParam.num_of_tags())]

        new_elbo = 0
        for i in range(line.Length):
            s0 = line.S[i]
            if i == 0:
                s_minus_1 = 0
                s1 = line.S[i + 1]
                # update c
                # compute E_q[logP(c_0)]
                elog_c0 = log_sticks_v_szs[s_minus_1][s0][0]
                            
                # compute E_q[logP(s_1)]
                elog_s1 = eta_szs[s_minus_1][s0][0].dot(vParam.log_dir_phi_t_sz[s0][:, s1]) # (px1)=(pxz)x(zx1)
                
                # compute E_q[logP(c_1)]
                elog_c1 = eta_szs[s_minus_1][s0][0].dot((log_sticks_v_szs[s0][s1].dot(now_c[i + 1]))) # (p_0xz_0)x((z_0xp_1)x(p_1x1))
                
                log_now_c, _ = log_normalize(elog_c0 + elog_s1 + elog_c1)               
                now_c[i] = np.exp(log_now_c)
                now_z[i] = now_c[i].dot(eta_szs[s_minus_1][s0][0])

                # computer elbo
                new_elbo+=np.sum(now_c[i] * (elog_c0 - log_now_c))
                new_elbo+=np.sum(now_c[i] * elog_s1)

                # update eta
                elog_s1 = np.outer(now_c[i], vParam.log_dir_phi_t_sz[s0][:,s1]) # (pxz)=(px1)x(1xz)
                d_eta_szs[s_minus_1][s0][0]+=elog_s1

                #for z in range(hParam.num_of_sub_tags(s0)):
                elog_c1 = np.outer(now_c[i], log_sticks_v_szs[s0][s1].dot(now_c[i + 1]))
                d_eta_szs[s_minus_1][s0][0]+=elog_c1

                # update v_szs_minus_one
                d_v_szs[s_minus_1][s0][0,0] = now_c[i][:-1]
                d_v_szs[s_minus_1][s0][0,1] = np.flipud(np.cumsum(np.flipud(now_c[i][1:])))

            else:
                s_minus_2 = 0 if i - 2 < 0 else line.S[i - 2]
                s_minus_1 = line.S[i - 1]                
                if i == line.Length - 1:
                    s1 = -1
                else:
                    s1 = line.S[i + 1]
                
                # compute E_q[logP(c_n)]
                if i-2<0:
                    eta_minus_1=eta_szs[s_minus_2][s_minus_1][0]
                else:
                    eta_minus_1=eta_szs[s_minus_2][s_minus_1].T.dot(now_z[i-2]).T

                eta_0=eta_szs[s_minus_1][s0].T.dot(now_z[i-1]).T

                elog_c0 = log_sticks_v_szs[s_minus_1][s0].T.dot(eta_minus_1.T.dot(now_c[i - 1]))

                if not(s1 < 0): # not the last one                 
                    # compute E_q[logP(s_{n+1})]
                    elog_s1 = eta_0.dot(vParam.log_dir_phi_t_sz[s0][:, s1]) # (px1)=(pxz)x(zx1)
                
                    # compute E_q[logP(c_{n+1})]
                    elog_c1 = eta_0.dot(log_sticks_v_szs[s0][s1].dot(now_c[i + 1]))
                
                    # compute E_q[log(x_n)]
                    elog_x0 = eta_0.dot(vParam.log_dir_phi_e1_sz[s0][:, line.X[i]])

                    log_now_c, _ = log_normalize(elog_c0 + elog_s1 + elog_c1 + elog_x0)
                else:
                    # compute E_q[log(x_n)]
                    elog_x0 = eta_0.dot(vParam.log_dir_phi_e1_sz[s0][:, line.X[i]])
                    log_now_c, _ = log_normalize(elog_c0 + elog_x0)
                
                now_c[i] = np.exp(log_now_c)
                now_z[i] = now_c[i].dot(eta_0)

                # compute elbo
                new_elbo+=np.sum(now_c[i] * (elog_c0 - log_now_c))
                new_elbo+=np.sum(now_c[i] * elog_x0)
                if not(s1 < 0):
                    new_elbo+=np.sum(now_c[i] * elog_s1)                 

                # update eta
                elog_x0 = np.outer(now_c[i], vParam.log_dir_phi_e1_sz[s0][:, line.X[i]])
                elog_x0 = np.tile(elog_x0, (len(now_z[i-1]),1,1)).T.dot(now_z[i-1]).T
                if not s1 < 0:
                    d_eta_szs[s_minus_1][s0]+=elog_x0

                    elog_s1 = np.outer(now_c[i], vParam.log_dir_phi_t_sz[s0][:,s1]) #(pxz)=(px1)x(1xz)
                    elog_s1 = np.tile(elog_s1, (len(now_z[i-1]),1,1)).T.dot(now_z[i-1]).T
                    d_eta_szs[s_minus_1][s0]+=elog_s1

                    elog_c1 = np.outer(now_c[i], log_sticks_v_szs[s0][s1].dot(now_c[i + 1]))
                    elog_c1 = np.tile(elog_c1, (len(now_z[i-1]),1,1)).T.dot(now_z[i-1]).T                    
                    d_eta_szs[s_minus_1][s0]+=elog_c1
                else:
                    d_eta_szs[s_minus_1][s0]+=elog_x0

                # update v_szs
                d_v_szs_0 = np.flipud(np.cumsum(np.flipud(now_c[i][1:])))
                d_v_szs[s_minus_1][s0][:,0]+=(now_c[i][:-1] * now_z[i - 1][:,np.newaxis])
                d_v_szs[s_minus_1][s0][:,1]+=(d_v_szs_0 * now_z[i - 1][:,np.newaxis])
                            
        # update eta_szs, v_szs and elbo        
        for (s_minus_1, s0) in seq2:
            eta_0 = vParam.log_sticks_v_s[s0] + d_eta_szs[s_minus_1][s0]
            for i in range(eta_0.shape[0]):
                log_eta_0, _ = log_normalize(eta_0[i])
                eta_szs[s_minus_1][s0][i] = np.exp(log_eta_0)
                new_elbo+=np.sum(np.multiply(eta_szs[s_minus_1][s0][i], vParam.log_sticks_v_s[s0] - log_eta_0))

            if (s_minus_1==0 and s0==0):
                v_szs[s_minus_1][s0][0] = np.array([1, hParam.alpha_szs(s0)])[:, np.newaxis] \
                    + d_v_szs[s_minus_1][s0][0]
                new_elbo+=(hParam.num_of_atoms(s0) - 1) * np.log(hParam.alpha_szs(s0))
                dig_sum = sp.psi(np.sum(v_szs[s_minus_1][s0][0],0))
                new_elbo+=np.sum((np.array([1, hParam.alpha_szs(s0)])[:,np.newaxis] \
                    - v_szs[s_minus_1][s0][0]) \
                        * (sp.psi(v_szs[s_minus_1][s0][0]) - dig_sum))
                new_elbo-=np.sum(sp.gammaln(np.sum(v_szs[s_minus_1][s0][0], 0))) - np.sum(sp.gammaln(v_szs[s_minus_1][s0][0]))
            else:
                for z in range(hParam.num_of_sub_tags(s_minus_1)):
                    v_szs[s_minus_1][s0][z] = np.array([1, hParam.alpha_szs(s1)])[:, np.newaxis] \
                        + d_v_szs[s_minus_1][s0][z]

                    # compute elbo
                    new_elbo+=(hParam.num_of_atoms(s0) - 1) * np.log(hParam.alpha_szs(s_minus_1))
                    dig_sum = sp.psi(np.sum(v_szs[s_minus_1][s0][z],0))
                    new_elbo+=np.sum((np.array([1, hParam.alpha_szs(s_minus_1)])[:,np.newaxis] \
                        - v_szs[s_minus_1][s0][z]) \
                            * (sp.psi(v_szs[s_minus_1][s0][z]) - dig_sum))
                    new_elbo-=np.sum(sp.gammaln(np.sum(v_szs[s_minus_1][s0][z],0))) - np.sum(sp.gammaln(v_szs[s_minus_1][s0][z]))

        converge = (new_elbo - old_elbo) / abs(old_elbo)        
        old_elbo = new_elbo
        
    if not(inf_only):
        # update tag level parameters
        for i in range(line.Length):
            s0 = line.S[i]
            if i == len(line.S) - 1:
                s1 = -1
            else:
                s1 = line.S[i + 1]
            if i == 0:
                s_minus_1 = 0    
                #update \phi^T_{sz}
                dParam.phi_t_sz[s0][:,s1]+=now_c[i].dot(eta_szs[s_minus_1][s0][0]) #(zx1)=(1xp)x(pxz)
            else:
                s_minus_1 = line.S[i - 1]
                eta_0=eta_szs[s_minus_1][s0].T.dot(now_z[i-1]).T
                if not(s1 < 0):
                    #update \phi^T_{sz}
                    dParam.phi_t_sz[s0][:,s1]+=now_c[i].dot(eta_0)

                #update \phi^E_{sz}
                dParam.phi_e1_sz[s0][:,line.X[i]]+=now_c[i].dot(eta_0)

        #update v_s        
        for (s_minus_1, s0) in seq2:
            dParam.v_s[s0][0]+=np.sum(eta_szs[s_minus_1][s0],(0,1))[:-1]
            dParam.v_s[s0][1]+=np.flipud(np.cumsum(np.flipud(np.sum(eta_szs[s_minus_1][s0],(0,1))[1:])))
    return new_elbo

def var_inf_tag(lines, hParam, vParam):
    old_elbo = -1.0e100
    iter = 0
    converge = 1.0    
    
    while True:
        iter+=1
        elbo_param = elbo_tag(hParam, vParam)
        dParam = VarParam(hParam)
        elbo_lines = 0
        line_counter = 0
        for line in lines:
            elbo_line = var_inf_line(line, hParam, vParam, dParam)
            elbo_lines+=elbo_line
            line_counter+=1
            sys.stdout.write('\riter(%d) (%d/%d)' % (iter, line_counter, len(lines)))
            sys.stdout.flush()

        new_elbo = elbo_param + elbo_lines
        converge = (new_elbo - old_elbo) / abs(old_elbo)
        print "\r----- iter(%d) elbo: (%f) convergence: (%f) -----" % (iter, new_elbo, converge)
        if not (iter < hParam.max_iter() and (converge < 0.0 or converge > hParam.var_convergence())):
            break
        old_elbo = new_elbo

        #update variational params
        for s in range(hParam.num_of_tags()):
            vParam.phi_e1_sz[s] = hParam.alpha_e1(s) + dParam.phi_e1_sz[s]            
            vParam.phi_t_sz[s][:,1:] = hParam.alpha_t(s) + dParam.phi_t_sz[s][:,1:]        

        #update v_s
        for s in range(hParam.num_of_tags()):
            vParam.v_s[s] = np.array([1, hParam.alpha_s(s)])[:, np.newaxis] + dParam.v_s[s]

        vParam.init_calc(hParam)
        
    return vParam, new_elbo

def elbo_tag(hParam, vParam):
    elbo = 0
    # compute E_q[logP(phi_e1_sz|)] - E_q[log q(phi_e1_sz)]
    for s in range(hParam.num_of_tags()):
        elog_phi_e1_sz = vParam.log_dir_phi_e1_sz[s]        
        elbo+=np.sum((hParam.alpha_e1(s) - vParam.phi_e1_sz[s]) * elog_phi_e1_sz)
        elbo+=np.sum(sp.gammaln(vParam.phi_e1_sz[s]) - sp.gammaln(hParam.alpha_e1(s)))        
        elbo+=np.sum(sp.gammaln(hParam.num_of_words() * hParam.alpha_e1(s)) - sp.gammaln(np.sum(vParam.phi_e1_sz[s], 1)))
    
    # compute E_q[logP(phi_t_sz|)] - E_q[log q(phi_t_sz)]
    for s in range(hParam.num_of_tags()):
        elog_phi_t_sz = vParam.log_dir_phi_t_sz[s][:,1:]
        elbo+=np.sum((hParam.alpha_t(s) - vParam.phi_t_sz[s][:,1:]) * elog_phi_t_sz)
        elbo+=np.sum(sp.gammaln(vParam.phi_t_sz[s][:,1:]) - sp.gammaln(hParam.alpha_t(s)))
        elbo+=np.sum(sp.gammaln((hParam.num_of_tags() - 1) * hParam.alpha_t(s)) - sp.gammaln(np.sum(vParam.phi_t_sz[s][:,1:], 1)))

    # compute E_q[logP(v_s)] - E_q[log q(v_s)]
    for s in range(hParam.num_of_tags()):
        elbo+=np.log(hParam.alpha_s(s)) * (hParam.num_of_sub_tags(s) - 1)
        elbo-=np.sum(sp.gammaln(np.sum(vParam.v_s[s],0))) - np.sum(sp.gammaln(vParam.v_s[s]))
        digsum = sp.psi(np.sum(vParam.v_s[s],0))
        digamma = sp.psi(vParam.v_s[s]) - digsum
        elbo+=np.sum((np.array([1, hParam.alpha_s(s)])[:,np.newaxis] - vParam.v_s[s]) * digamma)

    return elbo