import logging
import os
import numpy as np
logger = logging.getLogger(__name__)
from glob import glob
from scipy.io.matlab import loadmat
import numpy as np
import pandas as pd
import re
import warnings

class ABCDDataset: 
    def __init__(self, data_dir):
        self.data_root = data_dir
        self.ds_root   = os.path.join(data_dir, "external", "Julich")
        self.ds_external = os.path.join(data_dir, "external")
        
        
    def list_subjects(self):
        d = os.path.join(self.ds_root)
        return [subj for subj in os.listdir(d) if not subj.startswith(".")]

    
    def load_sc(self, subj, log10=False):
        
        d = os.path.join(self.ds_root)
        
        separator          = ''
        file_julich        = separator.join([d,'/',subj,'/ses-1/SC/',subj,'_SC_Schaefer_7NW100p.txt']) 
        file_julich_no_log = separator.join([d,'/',subj,'/ses-1/SC/',subj,'_SC_Schaefer7NW100p_nolog10.txt']) 
        SC                 = np.loadtxt(file_julich)
        SC_nolog           = np.loadtxt(file_julich_no_log)
        
        if log10:
            return SC
        else:
            return SC_nolog
        
    def load_bold(self,subj): #원래 VAB github에서는 이름이 `load_subject_fc_100`이다 (fc가 아니라 bold라서 bold로 이름 바꿈
        
        d = os.path.join(self.ds_root)
        
        separator   = ''
        file_julich = separator.join([d,'/',subj,'/ses-1/FC/',subj,'_meanTS_GS_bptf_Schaefer100_7NW.txt'])
        bold        = np.loadtxt(file_julich)
        
        return bold
    
    def parcellation_100(self):
        """
        lists the parcellation region names
        """
        d = os.path.join(self.ds_external)
        separator      = ''
        parce_frame    = pd.read_csv(separator.join([d,'/Schaefer2018_100Parcels_7Networks_tab.txt']), delimiter="\t",header=None)
        parce_list     = parce_frame[1].tolist()
        return parce_list

"""

밑의 것들 : 아직 implement안한 것들이고 그냥 그대로 VAB github에서 복붙 한것들 
* cognitive score : 이거는 내가 cognitive score csv파일을 따로 만들어서 data의 method중 하나로 만들어야 할듯 

"""

class metadata():
    def __init__(self):
        data_root = "/storage/bigdata/ABCD/TVB/ABCD_Relesae4.0_tabular_dataset.csv" #cumulative 해서 release 4만 쓰면 됨
        self.total_data = pd.read_csv(data_root)
        
        ##subject name에 _가 들어간 경우가 있어서, 제거해줘야함
        new_subjectkey = self.total_data['subjectkey'].str.replace("_","")
        self.total_data['subjectkey'] = new_subjectkey


        ###우리가 쓸 measure들의 집합의 columns들을 저장해놓자####
        self.demography_columns = ['subjectkey','eventname','sex','race_g','married','high_educ','high_educ2','income','foreign_born','religion_prefer','gay_parent','gay_youth','race_ethnicity','age','family_adversity','height','weight','abcd_site','family_id','vol','bmi','total_ratio','history_ratio','parent_identity','demo_brthdat_v2','demo_sex_v2','gender_identity','parent_age','foreign_born_family']
        self.nihbx_columns = self.total_data.columns[2:12] 
        self.cbcl_columns = self.total_data.columns[329:349]
        #self.demo_columns =  정의하기!
        
        #event index (특정 timeopint(event)일때의 index 값들만 따로 빼놓자) (boolearn sereis)
        self.baseline_idx = self.total_data["eventname"] =="baseline_year_1_arm_1" #나중에 쓰임, baseline의 그것을 알려주낟
        self.release_names = ["baseline_year_1_arm_1", "1_year_follow_up_y_arm_1","2_year_follow_up_y_arm_1","3_year_follow_up_y_arm_1"]
        
        """
        others should be added to, if they are ot be used (ex : KSADS and so on)
        Teams General Learning에 있는 정훈쌤이 올리신 release 4의 readme 읽으면 어떤 것 쓰면 될지 나옴
        """
    """
    self.columns만 저장해놨기에, cbcl_totprob = self.total_data['cbcl_totprob'].dropna() 이런식으로만 불러오면 된다!
    """

    def load_demo(self, release = 2):
        #load demogrpahy
        #baseline year를 쓸 것이다!
        warnings.warn("must implmemet so that some info that don't change but is only written in baseline should be extracted too")
        
        #new : release맞춰서 뽑기
        mask = (self.total_data['eventname']== self.release_names[release-1])
        
        demo = self.total_data[mask][self.demography_columns]
        
        #before : baseline그것에 한정해서 생각했다
        ##extract columns
        #demo = self.total_data[self.demography_columns]
        ##extract rows (only baseline)
        #demo = demo.loc[self.baseline_idx]
        
        return demo
    

    def extract_normal(self, data = None, release = [1,2], CBCL = 65, NIHTBX = 2):
        """
        CBCL < 65, |IQ| < 2td 인 subject들만 보겠다 (KSADS도 옵션으로 넣을 수 있을 듯?)
        #total 값들만 해서 함
        #주의 : NIHTBX는 release 1,3에만 있다
        INPUTS
        * release : 어느 release사용할지
            * 1 : baseline, 2 : 1 year followup, and so on
        * CBC, NIHTBX : threshold값들
        * data ; input data (which will be extracted and outputed)
            *안주어지면, 그냥 subject key만 나옴
        
        """
        warnings.warn("will do only based on CBCL and NIH toolboxes.. also, NIH toolbox only has release 1 and 3")
        
        remain_sub = set(self.total_data['subjectkey']) #i.e. set where we'll keep track of the stuff
        
        for when in release: #stage : what time stage was used
            follow_up = when - 1 #i.e. 1=> 0 (baseline)이 되고 그런다
            
            #in releases where NIH toolbox was done:
            
            mask_set = [] #where we'll store various masks for this release iteration
            
            #IQ mask
            if when == 1 or 3: #i.e. in releases where NIHTBX was measured
                nihtbx_data = self.total_data[self.total_data['eventname'] == "baseline_year_1_arm_1"]['nihtbx_totalcomp_uncorrected'].dropna()
                mu = nihtbx_data.mean()
                std = nihtbx_data.mean()
                mask_nihtbx = (self.total_data['eventname'] == "baseline_year_1_arm_1") & (abs(self.total_data['nihtbx_totalcomp_uncorrected']-mu) < NIHTBX*std ) 
                
                mask_set.append(mask_nihtbx)

            #CBCL
            mask_cbcl = (self.total_data['eventname']== self.release_names[follow_up]) & (self.total_data['cbcl_totprob']<CBCL) #i.e. 두 조건을 동시에 만족하는 mask만들기
            
            mask_set.append(mask_cbcl)
            
            filtered_sub_set = [set(self.total_data[mask]['subjectkey']) for mask in mask_set]
            
            for subs in filtered_sub_set:
                remain_sub= remain_sub & subs #iteratively filter out things

        revive_mask = self.total_data['subjectkey'].isin(remain_sub)
        
        #data주어졌으면 그것 filtering, 아니면 subjectkey 만 주기
        if data is not None:
            return data[revive_mask]
        else:
            return self.total_data[revive_mask]['subjectkey']
        
        
    """
    same subject key가진 row가 여러개 (b/c mulitple releases)이니, row number 을 무조건 keep하고, index를 subjectkey같은 걸로 바꾸지 말기!
    
    """
    
#밑 : 저번에 HBP에 맞춰서 해놓은 건데, 이제는 버려도 될듯? (만약에 reference를 위해 써야한다면 밑의 것을 쓰기)
#def metadata(data_dir):
#    data_root = data_dir
#    ds_external = os.path.join(data_root, "external")
#    patient = pd.read_csv(os.path.join(ds_external, 'HBP_Descriptives.csv'),delimiter=";",decimal = ',')
#    patient.set_index("ID_HBP", inplace=True)
#    return patient
        
     #def cognitive_score(self):
     #   
     #   d              = os.path.join(self.ds_external)
     #   separator      = ''
     #   score_frame    = pd.read_csv(separator.join([d,'/1000BD_cognitivescore_1visit.csv']),delimiter=";",decimal = ',')
     #   
     #   subj_age_CS       = score_frame["Age"].tolist()
     #   gender_CS         = score_frame["Sex"].tolist()
     #   subj_ID_CS        = score_frame["ID"].tolist()
     #   visit_CS          = score_frame["Visit"].tolist()
     #   processing_speed  = score_frame["TMT_ARW(ProcessingSpeed)"].tolist()
     #   concept_shifting  = score_frame["TMT_BA(ConceptShifting)"].tolist()
     #   working_memory    = score_frame["LPS_RRW(ProblemSolving)"].tolist()
     #   PhonematicFluency = score_frame["RWT_PB2RW(PhonematicFluency)"].tolist()
     #   SemanticFluency   = score_frame["RWT_SB2RW(SemanticFluency)"].tolist()
     #   PhonematicFluency_Switch  = score_frame["RWT_PGR2RW(PhonematicFluency_Switch)"].tolist()
     #   SemanticFluency_Switch    = score_frame["RWT_SSF2RW(SemanticFluency_Switch)"].tolist()
     #   Vocabulary        = score_frame["AWST03P(Vocabulary)"].tolist()
     #   
     #   return subj_age_CS,gender_CS,subj_ID_CS,visit_CS,processing_speed,concept_shifting,working_memory,PhonematicFluency,SemanticFluency,PhonematicFluency_Switch,SemanticFluency_Switch,Vocabulary   
    ####ADD METADATA TO HERE? (AS IN THE VAB GITHUB) => not sure yet.. let's see


