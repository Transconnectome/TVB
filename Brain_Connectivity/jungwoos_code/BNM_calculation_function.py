# import relavent package
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/')

import numpy as np
import pandas as pd
import bct
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

########################
### define functions ###
########################

# connectivity vector form data --> matrix form
def from_vec_form_to_mat_form(input_data, n_node):
    # input : connectivity vector form data
    # output : connectivity matrix form data

    init = time.time()

    count_mat_data = np.zeros((len(input_data), n_node, n_node))
    connectivity_matrix = np.zeros((n_node, n_node))
    for sub in range(len(input_data)):
        loc = 0
        for j in range(n_node):
            connectivity_matrix[j, j+1:] = input_data.iloc[sub, loc:(loc + (n_node - 1) - j)]
            loc += (n_node - 1) - j
        count_mat_data[sub] = connectivity_matrix + connectivity_matrix.T

    print(f'vector form --> matrix form conversion finish, time = {time.time() - init}')
    return count_mat_data     # type of this matrix is 3-dim np.array


# matrix form connectivity data --> vector form data
def from_mat_form_to_vec_form(input_data):
    connectivity_line_form_data = np.zeros(int(len(input_data) * (len(input_data)-1) / 2))
    loc = 0
    for j in range(len(input_data)):
        connectivity_line_form_data[loc:(loc+len(input_data) - 1 - j)] = input_data[j, j + 1:]
        loc += len(input_data) - 1 -j
    return connectivity_line_form_data


def weight_based_threshold(input_data, threshold: int):
    # remove connection when streamline count below the threshold
    # math :: if streamline count <= threshold : streamline counts = 0
    
    #threshold_absolute를 써야한다! (정우쌤코드보다는 absolute threshold를 쓰면 된다

    init = time.time()

    thresholded_data = input_data.copy()
    condition = thresholded_data <= threshold
    thresholded_data[condition] = 0

    print(f'weight-based thresholding is finished, time = {time.time() - init}')
    return thresholded_data

def density_based_threshold(input_data, threshold):
    # making the connection matrix to have density = threshold
    # input : connectivity matrix
    # output : thresholded connectivity matrix

    init = time.time()

    thresholded_data = np.zeros(input_data.shape)
    for i in range(len(input_data)):
        thresholded_data[i] = bct.threshold_proportional(input_data[i], threshold)
    print(f'density based thresholding is finished, time = {time.time() - init}')

    return thresholded_data


def calcul_density(input_data):
    # input data : connectivity "matrix"
    # output data : density

    init = time.time()

    density = np.zeros(len(input_data))
    for i in range(len(input_data)):
        density[i], b, c = bct.density_und(input_data[i])
    print(f'density calculation is finished, time = {time.time() - init}')
    return density

def calcul_n_comp(input_data):
    # input : connectivity matrix
    # output : number of components of the network

    init = time.time()

    n_comp = np.zeros(len(input_data))
    for i in range(len(input_data)):
        n_comp[i] = len(set(bct.get_components(input_data[i])[0]))
    print(f'the number of network component calculation is finished, time = {time.time() - init}')
    return n_comp

def calcul_connection_length_mat(input_data):
    # input : connectivity matrix
    # output : connection length matrix
    # math : connection length  = 1 / connection weight

    init = time.time()

    connection_length_mat = np.zeros(input_data.shape)
    for i in range(len(input_data)):
        connection_length_mat[i] = bct.weight_conversion(input_data[i], 'lengths')

    print(f'connection length matrix calculation is finished, time = {time.time() - init}')
    return connection_length_mat


def calcul_distance_mat(connection_length_mat):
    # input : connection length matrix
    # output : distance matrix, number of edges in shortest path (NOE_in_SP)

    init = time.time()

    distance_mat = np.zeros(connection_length_mat.shape)
    NOE_in_SP = np.zeros(connection_length_mat.shape)
    for i in range(len(connection_length_mat)):
        distance_mat[i], NOE_in_SP[i] = bct.distance_wei(connection_length_mat[i])

    print(f'distance matrix calculation is finished, time = {time.time() - init}')

    return distance_mat, NOE_in_SP


def make_nodal_column_list(region_list, variable_name):
    column_name_list = region_list.copy()
    for i in range(len(region_list)):
        column_name_list[i] = variable_name + '_' + region_list[i]
    return column_name_list

def calcul_degree(input_data, n_node):
    # input : connectivity matrix
    # output : degree
    init = time.time()
    degree = np.zeros((len(input_data),n_node))
    for i in range(len(input_data)):
        degree[i] = bct.degrees_und(input_data[i])
    print(f'degree calculation is finished, time = {time.time() - init}')
    return degree

def calcul_strength(input_data, n_node):
    # input : connectivity matrix
    # output : strength
    init = time.time()
    strength = np.zeros((len(input_data),n_node))
    for i in range(len(input_data)):
        strength[i] = bct.strengths_und(input_data[i])
    print(f'strength calculation is finished, time = {time.time() - init}')
    return strength

def calcul_global_efficiency(distance_mat):
    # input : distance matrix
    # output : global efficiency of the network

    init = time.time()
    efficiency = np.zeros(len(distance_mat))
    for i in range(len(distance_mat)):
        a, efficiency[i], c, d, e = bct.charpath(distance_mat[i])
    print(f'global efficiency calculation is finished, time = {time.time() - init}')
    return efficiency

def calcul_clust_coef(input_data, n_node):
    # input : connectivity matrix
    # output : clustering coefficient
    init_1 = time.time()
    clustering_coefs = np.zeros((len(input_data), n_node))
    for i in range(len(input_data)):
        clustering_coefs[i] = bct.clustering_coef_wu(input_data[i])
    print('clustering_coef calculation time = %f' % (time.time() - init_1))
    return clustering_coefs


def calcul_module_and_modularity_Louvain(input_data, n_node):
    init_1 = time.time()

    modular_structures = np.zeros((len(input_data), n_node))
    modularities = np.zeros(len(input_data))
    for i in range(len(input_data)):
        j = 0

        # to constraint the number of modules ~ 5
        while modular_structures[i].max() != 5:
            modular_structures[i], modularities[i] = bct.community_louvain(input_data[i])
            j += 1
            if j > 10000:
                if (modular_structures[i].max() == 6) or (modular_structures[i].max() == 4):
                    break
    print('modular structure calculation time = %f' %(time.time() - init_1))
    return modular_structures, modularities

def calcul_s_core(input_data, n_node):
    # input : connectivity matrix
    # output : s-core index of the nodes (and if you want s-core size also)
    init = time.time()
    s_core_index = np.zeros((len(input_data), n_node))
    s_core_size = np.zeros((len(input_data), 100))
    for sub_num in range(len(input_data)):
        s_core_index_list = np.zeros(n_node)
        score_size_list = np.zeros(100)
        for i in range(100):
            ss, score_size_list[i] = bct.score_wu(input_data[sub_num], 0.01 * i)
            cond = ss.sum(axis=1) != 0
            s_core_index_list[cond] = 0.01 * i

        s_core_index[sub_num] = s_core_index_list
        s_core_size[sub_num] = score_size_list
    print(f's-core index calculation is finished, time = {time.time() - init}')
    return s_core_index, s_core_size

def calcul_k_core(input_data, n_node):
    # input : connectivity matrix
    # output : k-core index
    init = time.time()
    kcore_index = np.zeros((len(input_data), n_node))
    for i in range(len(input_data)):
        kcore_index[i], b = bct.kcoreness_centrality_bu(np.sign(input_data[i]))
    print(f'k-core index calculation is finished, time = {time.time() - init}')
    return kcore_index

def calcul_closeness_centrality(distance_mat, n_node):
    # input : distance_matrix
    # output : closeness centrality = nodal efficiency
    init = time.time()
    closeness_centrality = np.zeros((len(distance_mat), n_node))
    for i in range(len(distance_mat)):
        for j in range(n_node):
            for k in range(n_node):
                if j != k:
                    closeness_centrality[i][j] += 1 / distance_mat[i][j][k]
            closeness_centrality[i][j] /= (n_node - 1)
    print('time consumed in calculating Cc, %f sec' % (time.time() - init))
    return closeness_centrality

def calcul_betweenness_centrality(connection_length_mat, n_node):
    # input : connection length matrix
    # output : betweenness centrality of nodes
    init = time.time()
    betweenness_centrality = np.zeros((len(connection_length_mat), n_node))
    for i in range(len(connection_length_mat)):
        betweenness_centrality[i] = bct.betweenness_wei(connection_length_mat[i])
    print('time consumed in calculating BC, %f sec' % (time.time() - init))
    return betweenness_centrality

def calcul_within_module_degree_zscore(input_data, modular_structures, n_node):
    # input : connectivity matrix, modular structures devided by calcul_module_and_modularity_Louvain
    # output : within module degree z-score of the nodes
    init = time.time()
    within_module_degree_zscore = np.zeros((len(input_data),n_node))
    for i in range(len(input_data)):
        within_module_degree_zscore[i] = bct.module_degree_zscore(input_data[i], modular_structures[i])
    print(f'within module degree z-score calculation is finished, time = {time.time() - init}')
    return within_module_degree_zscore

def calcul_participation_coefficient(input_data, modular_structures, n_node):
    # input : connectivity matrix and modular structures
    # output : participation coefficient
    init = time.time()
    participation_coefficient = np.zeros((len(input_data),n_node))
    for i in range(len(input_data)):
        participation_coefficient[i] = bct.participation_coef(input_data[i], modular_structures[i])
    print('participation coef calcultaion time = %f' % (time.time() - init))
    return participation_coefficient

def calcul_rich_club_coef(input_data, degree):
    # input : connectivity matrix, degree
    # output : rich club coefficient, rich club coefficient name list
    init = time.time()
    rich_club_coef = np.zeros((len(input_data), int(np.max(degree))))
    for i in range(len(input_data)):
        rich_club_coef[i] = bct.rich_club_wu(input_data[i], int(np.max(degree))) #이거는 이미 되어있다
    rich_club_coef_name_list = []
    for i in range(int(np.max(degree))):
        rich_club_coef_name_list.append('rich_club_coef_k=%d'%(i+1))
    print('time consumed in calculating Rich club, %f sec' % (time.time() - init))
    return rich_club_coef, rich_club_coef_name_list
