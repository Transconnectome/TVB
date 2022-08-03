import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np


print("IMPORTED")


def grouped_corr(data, y, p = 0.95, filter_bonf = True, n = None ):
    """
    INPUT : data , y , filter하게 된다면 쓸 p value,  filter_bonf 여부(bonferonni p-value filtering), n (number of cases to do bonferonni correction)
        * data : DATAFRAME of the things we want to do correlation study with
            * columns : measures (ex : connection1, connection2, etC)
            * rows : subject ID 
        * y : target dataframe SERIES
            * must be a series (ex : `dataframe['age']` or sth)
                (most probably : age Series)
        * p, filter_bonf, n :
            * p : filter을 하게 된다면 쓸, p-value (엄밀히, bonferonni한 p값을 쓴다
            * filter_bonf : True면 output이 bonfernnoi corrected p value보다 더 significant한것들만 output
            * n : custom # of cases (원래는 자동으로 갯수를 새주나, 직접 넣어줄 수도 있다)
                * 즉, filter_bonf = True면, p_value < p/n 인 아이들만 살림

    OUTPUT : data의 columns와 y 의 correlation 결과물들 (r, p value and so on) (possibly filtered by bonferrnoi corrected p value), SORTED by r value
    """
    corr_result = pd.DataFrame()
    merged_data = pd.concat([data, y], join='inner', axis=1) #merged_data를 만들어서 이것을 쓰는 이유 : subject 겹치지 않는 경우가 있으면 에러떠서, 굳이 이것을 쓰는 것 
    for measure in data.columns:
        corr_result = pd.concat([corr_result, pd.DataFrame(pg.corr(merged_data[measure], y))], axis=0)
    corr_result.index = data.columns
    
    #i.e. if filtering is True,
    if filter_bonf == True :
        #if no custom n value is specified, we infer n from the data column갯수
        if n == None:
            n = len(data.columns)
        cond = corr_result['p-val'] < p/n #p/n : bonferonni corrected p value
        corr_result = corr_result.loc[cond, :] #위에서 찾은 p value만족하는 columns만 살리기
    
    corr_result = corr_result.sort_values(by = 'r') #r 값으로 sorting
    return corr_result



def lin_reg():
    return None
