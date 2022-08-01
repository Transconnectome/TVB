# dMRI의 streamline count data를 input으로 받아서 다양한 network measurement를 구하는 code, pipe line(?)
# weighted / undirected version, calculation based on brain connectivity toolbox(bct)
# 구해지는 measurement
# global measure : average clustering coefficient, global efficiency, modularity
# nodal measure : degree, strength, clustering coefficient, nodal efficiency, 
#                 within module degree z-score, participation coefficient, betweenness centrality, k-core index, s-core index
# rich club measure : rich club coefficient

# streamline count를 network measurement로 변환하는 대략적인 과정에 대한 서술 :
# streamline count data read --> matrix form으로 변환 --> thresholding --> fragmented subject 제거 --> connectivity matrix scaling
# --> connection length matrix, distance matrix generation --> network measurement calculation --> file saving

###########################
# import relavent package #
###########################
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/')

import numpy as np
import pandas as pd
import bct
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from BNM_calculation_function import *

import argparse
####`#######################


#########################################
########## calculate BNM ################
#########################################

# theshold setting
parser = argparse.ArgumentParser()

parser.add_argument('--th_scheme', required=True, help="input threshold scheme; choose 'density' or 'weight'.")
parser.add_argument('--th', required=True, help='input threshold value')

args = parser.parse_args()

threshold_scheme = args.th_scheme
if threshold_scheme == 'weight':
    threshold = int(args.th)
elif threshold_scheme == 'density':
    threshold = float(args.th)

# import structural connectivity data (streamline count data) (outlier filtered)
count_data_dir = '/scratch/connectome/seojw/BNM analysis/data/con_aparc_count_outlier_filtered.csv'
count_data = pd.read_csv(count_data_dir).set_index('subjectkey')
n_node = 84

# node name list
temporary_list_for_region_list = count_data.columns.str.split('_')
region_list = []
region_list.append(temporary_list_for_region_list[0][1])
for i in range(n_node - 1):
    region_list.append(temporary_list_for_region_list[i][2])

# connectivity data transform : vector form --> matrix form
count_mat_data = from_vec_form_to_mat_form(count_data, n_node)

# thresholding : weight-based thresholding or density-based thresholding
if threshold_scheme == 'weight':
    thresholded_count_mat_data = weight_based_threshold(count_mat_data, threshold)

    # check n_components & select not fragmented subjects
    n_comp = calcul_n_comp(thresholded_count_mat_data)
    not_fragmented_condition = (n_comp == 1)
    subjectkeys = count_data.index[not_fragmented_condition]
    count_mat_data = thresholded_count_mat_data[not_fragmented_condition]

    
```
밑의 것 `thresholdscheme == density`는 할필요 없다!~! 한방으로 끝냈다 왜 이렇게 하는지 잘 이해가 안됨(여쭤보기?)
```
elif threshold_scheme == 'density':  #outlier 제거하는 과정(?)

    # exclude the subjects having density less than threshold before analysis
    # to make all analyized subjects  have the same density.
    
    #density-based outlier removal
    density = calcul_density(count_mat_data)
    inclusion_criteria = (density >= threshold)
    print(f'the number of excluded subjects is {len(count_data) - sum(inclusion_criteria)}')
    print(f'remaining subjects is {sum(inclusion_criteria)}')
    count_mat_data = count_mat_data[inclusion_criteria]
    subjectkeys = count_data.index[inclusion_criteria]

    thresholded_count_mat_data = density_based_threshold(count_mat_data, threshold)
    
    #fragmentaitno (all connected?)여부로 outlier removal
    # check n_components & select not fragmented subjects
    n_comp = calcul_n_comp(thresholded_count_mat_data) #81 in our case
    not_fragmented_condition = (n_comp == 1)
    subjectkeys = subjectkeys[not_fragmented_condition]
    count_mat_data = thresholded_count_mat_data[not_fragmented_condition]

# connectivity data 0-1 linear scaling with respect to the max streamline count of the whole dataset
count_mat_data = count_mat_data / np.max(count_mat_data)

#density_pd = pd.DataFrame(calcul_density(count_mat_data), index=subjectkey, columns=['density'])
connection_length_mat = calcul_connection_length_mat(count_mat_data)
distance_mat, NOE_in_SP = calcul_distance_mat(connection_length_mat)

# if you want to use the number of edges in shortest path(NOE_in_SP) for your analysis, un-comment and use below block
"""
NOE_in_SP_name_list = []
for i in range(len(temporary_list_for_region_list)):
    NOE_in_SP_name_list.append('NOE_in_SP_' + temporary_list_for_region_list[i][1] + '_' + temporary_list_for_region_list[i][2])

NOE_in_SP_line_form = np.zeros(count_data.loc[subjectkeys].shape)
for i in range(len(count_mat_data)):
    NOE_in_SP_line_form[i] = from_mat_form_to_vec_form(NOE_in_SP[i])
NOE_in_SP_pd = pd.DataFrame(NOE_in_SP_line_form, columns=NOE_in_SP_name_list, index=subjectkeys)
"""

###################################
###### Brain Network Measures #####
###################################
"""
count_mat_data는 그냥 self.mat 이다
"""
# degree
degree = calcul_degree(count_mat_data, n_node)
degree_name_list = make_nodal_column_list(region_list, 'deg')
degree_pd = pd.DataFrame(degree, columns=degree_name_list, index=subjectkeys)

# strength
strength = calcul_strength(count_mat_data, n_node)
strength_name_list = make_nodal_column_list(region_list, 'stren')
strength_pd = pd.DataFrame(strength, columns=strength_name_list, index=subjectkeys)

# global efficiency
efficiency = calcul_global_efficiency(distance_mat)
efficiency_pd = pd.DataFrame(efficiency, columns=['E_glob'], index=subjectkeys)

# nodal clustering coefficient & average clustering coefficient
clustering_coefs = calcul_clust_coef(count_mat_data, n_node)
clustering_coef_list = make_nodal_column_list(region_list, 'clust_coef')
clustering_coef_pd = pd.DataFrame(clustering_coefs, columns=clustering_coef_list, index=subjectkeys)
clustering_coef_pd['avg_clustering_coef'] = clustering_coef_pd.mean(axis=1)



# modular structure & modularity
modular_structures, modularities = calcul_module_and_modularity_Louvain(count_mat_data, n_node)
module_name_list = make_nodal_column_list(region_list, 'module_index_of')
modular_structure_pd = pd.DataFrame(modular_structures, columns=module_name_list, index=subjectkeys)
module_num_pd = pd.DataFrame(modular_structure_pd.max(axis=1), columns=['module_num'], index=subjectkeys)
modularity_pd = pd.DataFrame(modularities, columns=['optimal modularity'], index=subjectkeys)

"""
#=====FROM BELOW========
"""

####### Measures of Centrality #################
# : s-core index, k-core index, Closeness Centrality(Cc)=nodal efficiency, Betweenness Centrality(BC),
# + Within module degree z-score, participation coefficient  #########

# score index
s_core_index, s_core_size = calcul_s_core(count_mat_data, n_node)
score_name_list = make_nodal_column_list(region_list, 'score')
s_core_index_pd = pd.DataFrame(s_core_index, index=subjectkeys, columns=score_name_list)
s_core_size_pd = pd.DataFrame(s_core_size, index=subjectkeys,
                              columns=[f'score_size_s={round(i, 2)}' for i in np.arange(0, 1, 0.01)])

# measure of core structure, core-centrality
# k-coreness (k-core-index, k-core size)
kcore_index = calcul_k_core(count_mat_data, n_node)
kcore_index_name_list = make_nodal_column_list(region_list, 'kcore_index_of')
kcore_index_pd = pd.DataFrame(kcore_index, columns=kcore_index_name_list, index=subjectkeys)

# closeness centrality : Cc = nodal efficiency
# (Ref. Rubinov & Sporns 2010, Complex network measures of brain connectivity:Uses and interpretations)
closeness_centrality = calcul_closeness_centrality(distance_mat, n_node)
closeness_centrality_list = make_nodal_column_list(region_list, 'Cc')
closeness_centrality_pd = pd.DataFrame(closeness_centrality, columns=closeness_centrality_list, index=subjectkeys)

# betweenness centrality : BC
betweenness_centrality = calcul_betweenness_centrality(connection_length_mat, n_node)
betweenness_centrality_list = make_nodal_column_list(region_list, 'BC')
betweenness_centrality_pd = pd.DataFrame(betweenness_centrality, columns=betweenness_centrality_list, index=subjectkeys)

# within module degree z-score
within_module_degree_zscore = calcul_within_module_degree_zscore(count_mat_data, modular_structures, n_node)
within_module_degree_list = make_nodal_column_list(region_list, 'within_module_deg')
within_module_degree_pd = pd.DataFrame(within_module_degree_zscore, columns=within_module_degree_list, index=subjectkeys)

# participation coeffcient
participation_coefficient = calcul_participation_coefficient(count_mat_data, modular_structures, n_node)
participation_coef_list = make_nodal_column_list(region_list, 'participation_coef')
participation_coefficient_pd = pd.DataFrame(participation_coefficient, columns=participation_coef_list, index=subjectkeys)

# rich club coefficient
rich_club_coef, rich_club_coef_name_list = calcul_rich_club_coef(count_mat_data, degree)
rich_club_coef_pd = pd.DataFrame(rich_club_coef, columns=rich_club_coef_name_list, index=subjectkeys).fillna(0)

####################################
####################################


####################################
######### save BNM data ############
####################################

BNM = pd.concat([degree_pd, strength_pd, efficiency_pd, clustering_coef_pd, modularity_pd, s_core_index_pd, kcore_index_pd,
               closeness_centrality_pd, betweenness_centrality_pd, within_module_degree_pd, participation_coefficient_pd,
                rich_club_coef_pd], axis=1)
scaler = StandardScaler()
BNM_zscaled = pd.DataFrame(scaler.fit_transform(BNM), columns=BNM.columns, index=BNM.index)

if threshold_scheme == 'weight':
    file_name_th = 'ab'
elif threshold_scheme == 'density':
    file_name_th = 'den'

BNM.to_csv('/scratch/connectome/seojw/BNM analysis/data/BNM_'+file_name_th+str(threshold)+'.csv')
BNM_zscaled.to_csv('/scratch/connectome/seojw/BNM analysis/data/BNM_'+file_name_th+str(threshold)+'_zscaled.csv')
