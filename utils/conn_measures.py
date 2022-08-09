import bct
import pandas as pd
import warnings
import numpy as np

"""
things to add, when compared to 정우쌤 코드들은 다음과 같다
* strength based / absolute streamline count based thresholding => 일단은 density based로만 thresholding하기로 함
* use decorators???

def calcul_closeness_centrality(distance_mat, n_node): => 이거는 정우쌤이 직접 code로 구현하기는 했음. 일단은 무시

"""

class compute_bct_UW():
    def __init__(self, mat, threshold = None): #mat : matrix (SC or FC)
        #self.mat = mat
        self.threshold = threshold
        self.mat_original = mat
        self.mat = mat/mat.max() #0~1로 min/maxing
        
        #thresholding여부를 threshold값이 주어진 여부로 결정
        if threshold == None:
            warnings.warn("no thresholding will be used please check if this is what you want :)!")
            print("no thresholding will be used please check if this is what you want :)!")
        else :
            self.mat = bct.threshold_proportional(self.mat, threshold) #thresholded matrix를 쓴다       
            
        self.n_node = (self.mat).shape[0] #used later
        
        ######CHECK USABILITY, IF FAIL, RAISE EXCEPTION######
        if self.usability_check_density_based() == None:
            raise Exception("usability test FAILED")
        
        self.conn_len_mat = bct.weight_conversion(self.mat, 'lengths')
        self.dist_mat, self.NOE_in_SP = bct.distance_wei(self.conn_len_mat)
        #have to do "calcul_module_and_modularity_louvain" (modular structure is used on some other measures)
        
        ####bct inputs, modular structure and modulariteies
        count = 1
        modular_structures = np.zeros(5)
        while modular_structures.max() != 5:
            modular_structures, modularities = bct.community_louvain(self.mat)
            count+=1
            if count > 1000: #MUST BE CHANGED
                if (np.max(modular_structures) == 6) or (np.max(modular_structures)==4):
                    break
                    
        self.modular_structures = modular_structures
        self.modularities = modularities
        """
        다 implement한후에, "여러 output형태 가진 것들은 따로 def로 묶은 후, 써주기)
        """    
    """
    __init__에서 modular에서 5로 한 이유와, 조금더 correct한 approach가 뭔지 : Modular slowing of resting-state dynamic functional connectivity as a marker of cognitive dysfunction induced by sleep deprivation => 이 논문에서, bct.community_louvain쓸때 어떤 것을 써야하는지 나오게 함 (stochastic한 algorithm이니, 2000번 돌려서 community 갯수가 5개가 되게 되는 robust한 gamma값을 써야한다고 논문에서 나옴.. 일단은 그냥 1로 쓰되, fitting을 해야할 수도 있을듯) 
    """
    def scalar_properties(self):
        bcp = bct.charpath(self.dist_mat)
        data_dict = {
            "transivity" : bct.transitivity_wu(self.mat), #transitivity
            "local_efficiency" : bct.efficiency_wei(self.mat), #local efficiency
            "assortativity" : bct.assortativity_wei(self.mat,flag=0), #flag=0 because WU
            "pos_strength_sum" : bct.strengths_und_sign(self.mat)[2],
            "char_path_len" : bcp[0], 
            "global_efficiency" : bcp[1], 
            "graph_radius" : bcp[3], 
            "graph_diameter" : bcp[4], #float, float, vec, float, float
            "max_modularity_mertric_gam0_1" : bct.modularity_und(self.mat, 0.1)[1],
            "max_modularity_mertric_gam1" : bct.modularity_und(self.mat, 1)[1],
            "max_modularity_mertric_gam10" : bct.modularity_und(self.mat, 10)[1],
        }
        return data_dict, data_dict.keys()

    def vector_properties(self):
        s_core_index, s_core_size = self.s_core_computation()
        
        data_dict = {
            "degrees" : bct.degrees_und(self.mat),
            "clustering_coef" : bct.clustering_coef_wu(self.mat), ##better help documentation on the sign version
            "local_assortativity" : bct.local_assortativity_wu_sign(self.mat)[0], #becayse [1] is NaN becuase all values are positive 
            "betweenness_centrality" : bct.betweenness_wei(self.mat), 
            "eigenvector_centrality" : bct.eigenvector_centrality_und(self.mat),
            "pos_strength" : bct.strengths_und_sign(self.mat)[0],
            "vertex_eccentricity": bct.charpath(self.dist_mat)[2],
            "nodal_btw_vec" : bct.edge_betweenness_wei(self.mat)[1],
            "s_core_index_0to1" : s_core_index,
            "s_core_size_0to1" : s_core_size,
            "k_core_index" : bct.kcoreness_centrality_bu(np.sign(self.mat))[0],
            "within_module_deg_z_score" : bct.module_degree_zscore(self.mat, self.modular_structures),
            "participation_coef" : bct.participation_coef(self.mat, self.modular_structures),
            "rich_club_coef" : bct.rich_club_wu(self.mat), #klevel은 default로 되도록 함(maximujm degree로 자동 setting)
            "opt_community_struct_gam0_1" : bct.modularity_und(self.mat, 0.1)[0],
            "opt_community_struct_gam1" : bct.modularity_und(self.mat, 1)[0],
            "opt_community_struct_gam10" : bct.modularity_und(self.mat, 10)[0],
        }
        return data_dict, data_dict.keys()
    
    def matrix_properties(self):
        data_dict = {
            "matrix_itself_norm_thresh" : self.mat,
            "agreement" : bct.agreement(self.mat), #근데 이거 SC를 input으로 넣는게 맞느지 모르겠따... (일단은 돌려봄)
            "mean_first_passage_time" : bct.mean_first_passage_time(self.mat),
            "dist_mat" : self.dist_mat,
            "num_edge_shortest_path" : self.NOE_in_SP,
            "shortest_path_len" : bct.distance_wei_floyd(self.mat)[0],
            "num_edges_shortest_path":  bct.distance_wei_floyd(self.mat)[1],
            "Pmat":  bct.distance_wei_floyd(self.mat)[2],
            "edge_btw_mat" : bct.edge_betweenness_wei(self.mat)[0],
        }
        return data_dict, data_dict.keys()
    
    def usability_check_density_based(self):
        """
        returns True : CAN be used (opposite of its name haha)(rename?)
        returns None : CANNOT be used (does not pass the usability test)
        """
        density = bct.density_und(self.mat_original)[0]
        #density-based outlier removal was successful
        if self.threshold == None:
            warnings.warn( "threshold was not provided, will exit with None")
            return None
        elif density < self.threshold :
            warnings.warn ("density based outlier detected, will exit with None")
            return None
        else:
            #fragmentation outlier removal (since we're sure fragmentation didn't occur
            n_comp = len(bct.get_components(self.mat)[1]) #n_comp : component의 갯수
            if n_comp != 1:
                warnings.warn("n_comp > 1, will exit with None")
                return None
                #raise error 하지 않은 이유 : 여러번 돌릴 꺼라서 error뜨면 안됨 
            else :
                #print("subject passed the test!")
                return True 
    
    def s_core_computation(self):
        """
        do things only for s-core hting
        """
        s_core_index = np.zeros(self.n_node)
        s_core_size = np.zeros(100)
        for i,s_core in enumerate(np.linspace(0,1,100)):
            ss, s_core_size[i] = bct.score_wu(self.mat, s_core)
            cond = ss.sum(axis=1) != 0 #i.e. True if not zero (meaningful), (i.e. the thing is not zero things)
            s_core_index[cond] = s_core #어떤 score이었을때 되었는지를 keep track
            
        return s_core_index, s_core_size
    
"""
* 밑의 것들은  모두 additional paraemters를 정해줘야해서 하지 않았다 
* only exception : bct.modularity_und는 다양한 gamma값들을 넣어서 (0.1, 1, 10) 돌렸다! (with no starting community struture input)
* 나중에 class안에 method로 넣어도 될듯? (with additional  inputs to its methods)
"""

#########below dont do because parameter GAMMA needed###############
#bct.core_periphery_dir(sample_sc) #vec, scalar #don't do it becuase parameter needed
#help(bct.core_periphery_dir)

#bct.modularity_und(sample_sc) #vec, scalar #implemented in compute_BCT_UW, wit set q values (0.1, 1, 10) (with no starting community structure)
#help(bct.modularity_und)

#bct.community_louvain(sample_sc) #vec, scalar
#help(bct.community_louvain)
####################################################################




##################below : other properties that need other parameters############
#bct.agreement_weighted(sample_sc) #이거는 input으로 weight가 따로 들어가야함
#help(bct.agreement_weighted)

#bct.consensus_und(sample_sc)  #tau, reps, seed필요 
#help(bct.consensus_und)

#bct.retrieve_shortest_pathr(sample_sc)  #have to put in certain parameters (source node, output node and so on), and output is not some thing
#help(bct.retrieve_shortest_path)

#bct.pagerank_centrality(sample_sc) #need dampning factor, d as input
#help(bct.pagerank_centrality)
################################################################################


################the things below need ci(community affiliation vector) as input###################
#bct.module_degree_zscore(sample_sc)         => implemented
#bct.participation_coef(sample_sc)        => implemented
#bct.participation_coef_sign(sample_sc)        => implemented
#bct.gateway_coef_sign(sample_sc)
#bct.diversity_coef_sign(sample_sc)                     
######################################################################


##############other quirky properties that couldn't be used for whatever reason
#bct.link_communities(sample_sc) ##type clustering = "single" or "complete"  두개 있다! #오래걸림!~
## 일단은 오래 걸려서 commented out

##vector but varying length (depending on the SC property)
#bct.rich_club_wu(sample_sc).shape  => implemented (따로 klevel input을 안줘서, 자동으로 maximum degrees를 쓰도록 
#help(bct.rich_club_wu)



