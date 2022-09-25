import bct
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
import pingouin
import matplotlib.pyplot as plt
import warnings
sys.path.append("/scratch/connectome/TVB/TVB_RESEARCH")
sys.path
import utils #now imports becasue it looks at the last added sys path

import warnings
warnings.filterwarnings('ignore')
import time
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description = "argparse")
parser.add_argument('--threshold', type = float, help = "threshold to use")
args = parser.parse_args()
 
data_dir = '/storage/bigdata/ABCD/TVB/data'

####load dataset and stuff 
dataset = utils.load_data.ABCDDataset(data_dir)  #이제 된다
subjects = dataset.list_subjects()


start = time.time()
warnings.filterwarnings('once') #여기서는 한번만 나오게 하기 (여러번 돌아가면서 계속 뜨는 것은 싫다)

###how much to keep
sample_size = len(subjects) #if you wanna run all
threshold = args.threshold

##scalar meausres initialization
sample_sub = dataset.load_sc(subjects[0])
data, names = utils.conn_measures.compute_bct_UW(sample_sub, threshold = 0.3).scalar_properties()


num_sub = len(subjects) #8238
num_measures = len(names)
scalar_bct_results = np.empty((min(num_sub,sample_size),num_measures))
scalar_bct_results[:] = np.nan


###SC + SCALAR (vector, matrix는 할지말지, 그리고 한다면 언제할지는 모르겠다)
warnings.filterwarnings("ignore")
tqdm_loop = tqdm(subjects[:sample_size]) #tqdm_loop이라는 tqdm object로 만들기 => 나중에 descrioption을 둘거라서
fail_count = 0
for i,sub in enumerate(tqdm_loop):#enumerate(subjects[:sample_size]):#tqdm(subjects[:sample_size])):
    sub_sc = dataset.load_sc(sub)
    #do FC too!!!
        
    #trying to load the thing 
        #utils.conn_measuers 그거 initiailize할때, this has "check usaiblity" built-in (in __init__) 
        #threfore, when the matrix_loaded fails, it means the matrix is not usable
    try :
        matrix_loaded = utils.conn_measures.compute_bct_UW(sub_sc, threshold=threshold)
    except : 
        fail_count += 1
        tqdm_loop.set_postfix_str("sub-{} failed! || {}th fail".format(sub,fail_count))
        continue

    data, names = matrix_loaded.scalar_properties()
    scalar_bct_results[i] = np.array(list(data.values()))

total_scalar_results = pd.DataFrame(scalar_bct_results, index = subjects[:sample_size], columns = names)
end = time.time()
print("run_time(s) :", end-start)


BCT_result_save_dir = "/storage/bigdata/ABCD/TVB/BCT_results/SC_ses_2/"

total_scalar_results.to_csv(BCT_result_save_dir+"scalar_results_threshold_{}.csv".format(threshold))
pd.read_csv(BCT_result_save_dir+"scalar_results_threshold_{}.csv".format(threshold), index_col=0) #use the first column as the rows
os.listdir(BCT_result_save_dir)
