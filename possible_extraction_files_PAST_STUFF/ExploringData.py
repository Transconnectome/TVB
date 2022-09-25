import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

a=np.array([10])
data=pd.read_csv("DATA.csv", index_col='subjectkey') #imported the data #set index as subjectkey

'''====step0: divide into features_matrix/target arrays (training등으로 바꾸는건 나중에)'''
output=data['sex']
input=data.drop(['sex'], axis=1) #drop column, so axis=1

'''===========step0-1: see the min/max values of each column to see if they are normalized========'''

aa=input.mean(axis=0) #이렇게 하면 column별로의 average를 볼 수 있는 것 같다

#이제 이 aa를 plot하든 distribution을 보든지 해서, 값이 일정 range안에 모두 존재하는지 보자
#만약 값들이 scale이 다르면 normalization을 거쳐야 할듯

plt.hist(aa.values,bins=10)
plt.show()
#너무나도 큰 값의 차이가 보이기에, normalization을 거처야 할듯하다.

'''====step0-2. Go through normalizatoin(?) to tame it a bit'''
#값들이 0이상이니 그냥 arctanh()로 하자
input=np.tanh(input) #normalize the inputs using tanh

#open this when I us eit
input.to_csv("input.csv")#saves in the current cwd with file name I provide (and also the extension (.csv) I provide)
output.to_csv("output.csv")#saves in the current cwd with file name I provide (and also the extension (.csv) I provide)



