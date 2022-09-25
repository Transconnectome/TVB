import pandas as pd
import numpy as np

'''===step0: import the datas===='''

phenotype=pd.read_csv("demo.total.csv")
con_count=pd.read_csv("con_aparc_count.csv")
con_fa=pd.read_csv("con_aparc_fa.csv")
con_length=pd.read_csv("con_aparc_length.csv")
con_rd=pd.read_csv("con_aparc_rd.csv")
#if I want to use the absolute file, i need to put r in front of the string (see link below)
#used https://stackoverflow.com/questions/37400974/unicode-error-unicodeescape-codec-cant-decode-bytes-in-position-2-3-trunca#

wholedata=[phenotype,con_count,con_fa,con_length,con_rd] #for use in for loops


'''====step1: set the index as the subject key==='''
for i in wholedata:
    i.set_index('subjectkey',inplace=True)#BECAREFUL!!!(inplace is needed!!)
    #this is the same as
    #i=i.set_index('subjectkey')

'''
**below: to see if the subjectkeys have successfully been made into the index
for i in wholedata:
    print(i.head())
'''
#extra step 0: phenotype subject key is weird: so change it
phenotype.index=phenotype.index.str.replace("_","") #changes _ with nothing


data_sex=phenotype['sex'] #only extract the phenotype data


wholedata[0]=data_sex #set data_sex as new
#meta-data of data_sex Series로 "sex"가 있기에 굳이 column name을 지정해줄 필요가 없음
#(원래는 Series에는 column name이 없으니 따로 지정해줘야하는줄 알았다)
'''=====step2: concatenate the data===(along axis=1, with only the intersections'''
total=pd.concat([i for i in wholedata],axis=1,join="inner") #join inner means only taking key values that appear in both cases

#total is the final thing we want!

'''=====step4: check for NaN values and remove samples with that '''
print((np.count_nonzero(total.isnull()))) #we can see that lots of null values exist!!

total.dropna(inplace=True) #inplace=True so that the total itself is changed
#axis=0 by default, how='any' by default,
#for reference, look at :https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

print((np.count_nonzero(total.isnull()))) #prints zero, as expected (no null values)

total.to_csv("DATA.csv")#saves in the current cwd with file name I provide (and also the extension (.csv) I provide)





#todo: read https://intellij-support.jetbrains.com/hc/en-us/community/posts/360004715019-My-python-file-will-execute-code-from-another-python-file-and-when-I-deleted-the-file-the-code-was-excuted-from-other-files-it-gave-me-an-error

#todo: 학교 tech support에 전화해서 microsoft word다시 작동하게 하기(논문작성 마무리해야함)

