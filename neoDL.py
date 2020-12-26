# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 00:29:08 2020

@author: HYF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from keras.models import load_model
import os
import sys,getopt

def main(argv):
    path = ''
    inputfile = ''
    modelp = ''
    try:
        opts, args = getopt.getopt(argv,"ho:i:m:",["path=","ifile","modelp="])
    except getopt.GetoptError:
        print('test.py -o <path> -i <inputfile> -m <model_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -o <path> -i <inputfile> -m <model_path>')
            sys.exit()
        elif opt in ("-o", "--path"):
            path = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-m", "--modelp"):
            modelp = arg
    return [path,inputfile,modelp]

if __name__ == "__main__":
    a=main(sys.argv[1:])
    path=a[0]
    inputfile=a[1]
    modelp=a[2]

kmf = KaplanMeierFitter()

outter_path = path+"/ex_v_results"
surv_pic_path = outter_path+"/surv_pics"
test_result_path = outter_path+"/test_cluster_result"
os.makedirs(surv_pic_path)
os.makedirs(test_result_path)

#########load the best model
model=load_model(modelp)

########get test set from pri cohort, randomly select 60% from each labeled group
def get_test(i):
    df=pd.read_csv(inputfile)
    feat1=df[(df.feat==1)]#extract data in group1
    feat2=df[(df.feat==2)]#extract data in group2
    #group1
    feat1=feat1.sample(frac=1.0)#disorganize the order of the data
    cut_idx=int(round(0.6 * feat1.shape[0]))#set the ratio of training set vs test set
    feat1_test = feat1.iloc[:cut_idx]
    #group2
    feat2=feat2.sample(frac=1.0)
    cut_idx2=int(round(0.6 * feat2.shape[0]))
    feat2_test = feat2.iloc[:cut_idx2]
    df_test=pd.concat([feat1_test,feat2_test])#joint group1 and 2
    df_test=df_test.sample(frac=1.0)#disorganize the order
    df_test.to_csv(test_result_path+'/test%s.csv'%i)#save
    test=df_test.drop(columns=['name','sample','days','vital','Num_mutations','feat'])#delete useless columns
    return test

########survival analysis
def surv(f,j):
    ax=plt.subplot(111)
    f=f[['days','vital','res']]
    #separate the data into 2 groups with deep learning result
    f0=f[(f.res==0)]
    f1=f[(f.res==1)]
    t0=f0['days']
    e0=f0['vital']
    t1=f1['days']
    e1=f1['vital']
    #survival analysis and making plots
    kmf.fit(t0,e0,label='0')
    kmf.plot(ax=ax)
    kmf.fit(t1,e1,label='1')
    kmf.plot(ax=ax)
    plt.ylim(0,1)
    results = logrank_test(t0, t1, e0, e1)
    p=results.p_value
    plt.text(1,1,'P=%s'%str(p))
    plt.savefig(surv_pic_path+'/surv_fig%s.png'%j)
    plt.close()
    return p

#########whole process
ps=[]
n=0
for j in range(300):#iterations of external validation
    #specificate the format for deep learning
    a=get_test(j)
    test=np.array(a.values)
    test=test.reshape(test.shape[0],1,test.shape[1]) 
    pred=model.predict(test)#cluster results
    #save the results
    res=[]
    for i in pred:
        res.append(round(i[0]))#get integers of the results
    #joint the result with the original test set
    res=pd.DataFrame(res)
    patha=test_result_path+'/test%s.csv'%j
    file=pd.read_csv(patha)
    file['res']=res
    file.to_csv(patha)  
    p=surv(file,j)#p value in survival analysis
    #count numbers of p that lower than 0.05
    if p<0.05:
        n+=1
    ps.append(p)
    print('finish round%s'%j)
ps.append(n)
p=pd.DataFrame(ps)
p.to_csv(outter_path+'/pvalues.csv')