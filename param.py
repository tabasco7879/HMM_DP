import numpy as np
from utils import expect_log_dirichlet, expect_log_sticks

class HyperParam:
    def __init__(self, vocab_sz, tag_sz, l_sz):
        self._vocabulary_size=vocab_sz        
        self._tag_size=tag_sz

        # LINE_START, TEXT, DATE, NAME (PATIENT, DOCTOR, USERNAME), AGE, 
        # LOCATION(COUNTRY, ORGANIZATION, STREET, CITY, STATE, ZIP, HOSPITAL, LOCATION_OTHER),
        # PROFESSION, CONTACT(PHONE, FAX, EMAIL, URL, IPADDRESS),
        # IDS(MEDICALRECORD, IDNUM, DEVICE, BIOID, HEALTHPLAN)
        
        self._sub_tags_num = [4, 20, 4, 12, 4,
                              12,
                              8, 4,
                              6]
        self._atoms_num = [2, 10, 2, 6, 2,
                           6,
                           4, 2,
                           3]

        self._sub_tags_num = [2, 10, 4]
        self._atoms_num = [4, 5, 2]

        self._alpha_e1=.1       
        self._alpha_t=.1

        self._alpha_s=1
        self._alpha_szs_minus_one=.1
        self._alpha_szs=.1

        self._line_num=l_sz
    
    def num_of_sub_tags(self, s):
        return self._sub_tags_num[s]
    
    def num_of_atoms(self, s):
        return self._atoms_num[s]

    def num_of_tags(self):
        return self._tag_size

    def num_of_words(self):
        return self._vocabulary_size

    def num_of_lines(self):
        return self._line_num

    def alpha_e1(self, s):
        return self._alpha_e1
   
    def alpha_t(self, s):
        return self._alpha_t
    
    def alpha_s(self, s):
        return self._alpha_s

    def alpha_szs_minus_one(self):
        return self._alpha_szs_minus_one

    def alpha_szs(self, s):
        return self._alpha_szs

    def var_convergence(self):
        return 1e-5

    def max_iter(self):
        return 200

    def max_iter_line(self):
        return 100

class VarParam:
    def __init__(self, hParam, init_type='zero', init_file_name=None):
        if init_type=='zero':
            self.init_zero(hParam)
        elif init_type=='gamma':
            self.init_uniform(hParam)
            self.init_gamma(hParam)
            self.init_calc(hParam)
        elif init_type=='file':
            self.init_file(hParam, init_file_name)
            self.init_calc(hParam)
        elif init_type=='uniform':
            self.init_uniform(hParam)
            self.init_calc(hParam)
    
    def init_uniform(self, hParam):
        self.phi_e1_sz = [np.ones((hParam.num_of_sub_tags(s), \
            hParam.num_of_words())) \
            * hParam.alpha_e1(s)
            for s in range(hParam.num_of_tags())]                 
        
        self.phi_t_sz = [np.ones((hParam.num_of_sub_tags(s), \
            hParam.num_of_tags())) \
            * hParam.alpha_t(s)
            for s in range(hParam.num_of_tags())]

        for s in range(hParam.num_of_tags()):
            self.phi_t_sz[s][:,0]=0
            
        self.v_s = [np.ones((2, hParam.num_of_sub_tags(s)-1))
            for s in range(hParam.num_of_tags())]

        for s in range(hParam.num_of_tags()):
            self.v_s[s][1]=hParam.alpha_s(s)                                          

    def init_gamma(self, hParam, phi_e1_sz=True, phi_t_sz=True):               
        for s in range(hParam.num_of_tags()):
            if (phi_e1_sz):
                self.phi_e1_sz[s]=(hParam.alpha_e1(s)+
                    np.random.gamma(1,1,(hParam.num_of_sub_tags(s), hParam.num_of_words()))
                        *hParam.num_of_lines()*20.0/(hParam.num_of_sub_tags(s)*hParam.num_of_words())
                        )            
            if (phi_t_sz):
                self.phi_t_sz[s][:,1:]=(hParam.alpha_t(s)+
                    np.random.gamma(1,1,(hParam.num_of_sub_tags(s), hParam.num_of_tags()-1))
                        *hParam.num_of_lines()*20.0/(hParam.num_of_sub_tags(s)*hParam.num_of_tags())
                        )
                
    def init_calc(self, hParam):
        self.log_sticks_v_s=[np.zeros(hParam.num_of_sub_tags(s)) 
            for s in range(hParam.num_of_tags())]

        self.log_dir_phi_e1_sz=[np.zeros((hParam.num_of_sub_tags(s), \
            hParam.num_of_words()))
            for s in range(hParam.num_of_tags())]                

        self.log_dir_phi_t_sz = [np.zeros((hParam.num_of_sub_tags(s), \
            hParam.num_of_tags()))
            for s in range(hParam.num_of_tags())]        
        
        for s in range(hParam.num_of_tags()):
            self.log_sticks_v_s[s]=expect_log_sticks(self.v_s[s])
            self.log_dir_phi_e1_sz[s]=expect_log_dirichlet(self.phi_e1_sz[s])            
            self.log_dir_phi_t_sz[s]=expect_log_dirichlet(self.phi_t_sz[s], True)                                    

    def init_zero(self, hParam):
        self.phi_e1_sz = [np.zeros((hParam.num_of_sub_tags(s), \
            hParam.num_of_words()))
            for s in range(hParam.num_of_tags())]

        self.phi_t_sz = [np.zeros((hParam.num_of_sub_tags(s), \
            hParam.num_of_tags()))
            for s in range(hParam.num_of_tags())]

        self.v_s = [np.zeros((2, hParam.num_of_sub_tags(s)-1))
            for s in range(hParam.num_of_tags())]        
   
    def init_file(self, hParam, file_name):
        self.init_zero(hParam)
        var_param_dict={
            'phi_e1_sz': self.phi_e1_sz, 
            'phi_t_sz': self.phi_t_sz,            
            'v_s': self.v_s
            }
        with open(file_name) as f:
            option=f.readline()
            while (len(option)>0):
                option=option.strip()
                if (option=='phi_e1_sz'                     
                    or option=='phi_t_sz'):
                    for s in range(hParam.num_of_tags()):
                        for r in range(hParam.num_of_sub_tags(s)):
                            var_param_dict[option][s][r]= \
                                    [float(w) for w in f.readline().split()]
                if (option=='v_s'):
                    for s in range(hParam.num_of_tags()):
                        for r in range(2):
                            var_param_dict[option][s][r]= \
                                    [float(w) for w in f.readline().split()]
                option=f.readline()
            
    def save(self, hParam, file_name, elbo=-1e100):
        with open(file_name, 'w') as f:
            f.write("elbo: %f\n"%elbo)
            f.write("phi_e1_sz\n")
            for s in range(hParam.num_of_tags()):
                for r in self.phi_e1_sz[s]:
                    line=' '.join([str(x) for x in r])
                    f.write(line+'\n')            

            f.write("phi_t_sz\n")
            for s in range(hParam.num_of_tags()):
                for r in self.phi_t_sz[s]:
                    line=' '.join([str(x) for x in r])
                    f.write(line+'\n')
            
            f.write("v_s\n")
            for s in range(hParam.num_of_tags()):
                for r in self.v_s[s]:
                    line=' '.join([str(x) for x in r])
                    f.write(line+'\n')