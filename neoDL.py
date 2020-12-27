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

outter_path = path+"/neoDL_results"
os.mkdir(outter_path)

#########load the best model
model=load_model(modelp)

########get test set from pri cohort, randomly select 60% from each labeled group
def test_input():
    t1=pd.read_csv(inputfile)
    t1.to_csv(outter_path+'/testnresult.csv')
    test_final=t1.drop(columns=['name','sampleID','days','vital'])
    return test_final

########survival analysis
def surv(f):
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
    plt.savefig(outter_path+'/surv_fig.png')
    plt.close()
    return p

#########whole process
#specificate the format for deep learning
a=test_input()
test=np.array(a.values)
test=test.reshape(test.shape[0],1,test.shape[1]) 
pred=model.predict(test)#cluster results
#save the results
res=[]
for i in pred:
    res.append(round(i[0]))#get integers of the results
#joint the result with the original test set
res=pd.DataFrame(res)
patha=outter_path+'/testnresult.csv'
file=pd.read_csv(patha)
file['res']=res
file.to_csv(patha)  
p=surv(file)#p value in survival analysis
#output p
print('finish predicting')
print("P-value from survival analysis is:%s"%p)
