import bct
import pandas as pd

class compute_bct_UW():
    def __init__(self, mat): #mat : matrix (SC or FC)
        self.mat = mat
    
    def scalar_properties(self):
        data_dict = {
            "transivity" : bct.transitivity_wu(self.mat), #transitivity
            "local_efficiency" : bct.efficiency_wei(self.mat), #local efficiency
            "assortativity" : bct.assortativity_wei(self.mat,flag=0), #flag=0 because WU
            "pos_strength_sum" : bct.strengths_und_sign(self.mat)[2],
            "char_path_len" : bct.charpath(self.mat)[0], 
            "global_efficiency" : bct.charpath(self.mat)[1], 
            "graph_radius" : bct.charpath(self.mat)[3], 
            "graph_diameter" : bct.charpath(self.mat)[4], #float, float, vec, float, float
            "max_modularity_mertric_gam0_1" : bct.modularity_und(self.mat, 0.1)[1],
            "max_modularity_mertric_gam1" : bct.modularity_und(self.mat, 1)[1],
            "max_modularity_mertric_gam10" : bct.modularity_und(self.mat, 10)[1],
        }
        return data_dict, data_dict.keys()

    def vector_properties(self):
        data_dict = {
            "degrees" : bct.degrees_und(self.mat),
            "clustering_coef" : bct.clustering_coef_wu(self.mat), ##better help documentation on the sign version
            "local_assortativity" : bct.local_assortativity_wu_sign(self.mat)[0], #becayse [1] is NaN becuase all values are positive 
            "betweenness_centrality" : bct.betweenness_wei(self.mat), 
            "eigenvector_centrality" : bct.eigenvector_centrality_und(self.mat),
            "pos_strength" : bct.strengths_und_sign(self.mat)[0],
            "vertex_eccentricity": bct.charpath(self.mat)[2],
            "nodal_btw_vec" : bct.edge_betweenness_wei(self.mat)[1],
            "opt_community_struct_gam0_1" : bct.modularity_und(self.mat, 0.1)[0],
            "opt_community_struct_gam1" : bct.modularity_und(self.mat, 1)[0],
            "opt_community_struct_gam10" : bct.modularity_und(self.mat, 10)[0],
        }
        return data_dict, data_dict.keys()
    
    def matrix_properties(self):
        data_dict = {
            "agreement" : bct.agreement(self.mat), #근데 이거 SC를 input으로 넣는게 맞느지 모르겠따... (일단은 돌려봄)
            "mean_first_passage_time" : bct.mean_first_passage_time(self.mat),
            "dist_mat" : bct.distance_wei(self.mat)[0],
            "num_edge_shortest_path" : bct.distance_wei(self.mat)[1],
            "shortest_path_len" : bct.distance_wei_floyd(self.mat)[0],
            "num_edges_shortest_path":  bct.distance_wei_floyd(self.mat)[1],
            "Pmat":  bct.distance_wei_floyd(self.mat)[2],
            "edge_btw_mat" : bct.edge_betweenness_wei(self.mat)[0],
        }
        return data_dict, data_dict.keys()

    
    
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

#bct.score_wu(sample_sc) #s value needed
#help(bct.score_wu)

#bct.pagerank_centrality(sample_sc) #need dampning factor, d as input
#help(bct.pagerank_centrality)
################################################################################


################the things below need ci(community affiliation vector) as input###################
#bct.module_degree_zscore(sample_sc) 
#bct.participation_coef(sample_sc)
#bct.participation_coef_sign(sample_sc)
#bct.gateway_coef_sign(sample_sc)
#bct.diversity_coef_sign(sample_sc)                     
######################################################################


##############other quirky properties that couldn't be used for whatever reason
#bct.link_communities(sample_sc) ##type clustering = "single" or "complete"  두개 있다! #오래걸림!~
## 일단은 오래 걸려서 commented out

##vector but varying length (depending on the SC property)
#bct.rich_club_wu(sample_sc).shape
#help(bct.rich_club_wu)



