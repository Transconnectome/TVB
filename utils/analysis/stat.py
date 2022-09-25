import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np

def grouped_corr(data, y, p = 0.05, filter_bonf = True, n = None, covar = None ):
    """description about this method is long, so it is put after the function definition below"""
    corr_result = pd.DataFrame()
    
    if covar is not None:
        merged_data = pd.concat([data, y, covar], join='inner', axis=1) 
    else :
        merged_data = pd.concat([data, y], join='inner', axis=1) 
            
    for measure in data.columns:
        if covar is not None:
            corr_result = pd.concat([corr_result, pd.DataFrame(pg.partial_corr(data = merged_data, x = measure, y = y.name, covar = [var for var in covar.columns]))])#, axis = 0)
        else:
            corr_result = pd.concat([corr_result, pd.DataFrame(pg.corr(merged_data[measure], merged_data[y.name]))], axis=0)
                                     
    corr_result.index = data.columns
    
    #filter significant things only
    if filter_bonf == True :
        #if no custom n value is specified, we infer n from the data column갯수
        if n == None:
            n = len(data.columns)
        cond = corr_result['p-val'] < p/n #p/n : bonferonni corrected p value
        corr_result = corr_result.loc[cond, :] #위에서 찾은 p value만족하는 columns만 살리기
    
    corr_result = corr_result.sort_values(by = 'r') #r 값으로 sorting
    return corr_result

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
        * covar :
            * if covariance dataframe is provided, will use it to run covariances into account

    OUTPUT : data의 columns와 y 의 correlation 결과물들 (r, p value and so on) (possibly filtered by bonferrnoi corrected p value), SORTED by r value
    """
    #merged_data를 만들어서 이것을 쓰는 이유 : subjects가 안겹치면 안되서
        #=> 어 그러면 그냥 겹치는 index 쓰면 안되냐? => 혹시나 나중에, merged 되기전 데이터가 필요할까봐 남겨둠 (코딩하기도 귀찮고.. 메모리 많은데)

        
        
def group_reg():
    return print("NOT IMPLEMENTED!! HAHA")
