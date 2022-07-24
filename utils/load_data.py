import logging
import os
import numpy as np
logger = logging.getLogger(__name__)
from glob import glob
from scipy.io.matlab import loadmat
import numpy as np
import pandas as pd
import re

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


