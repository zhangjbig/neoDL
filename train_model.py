# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 00:23:49 2020

@author: HYF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
import sys,getopt
kmf = KaplanMeierFitter()

def main(argv):
    path = ''
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"ho:i:",["ifile"])
    except getopt.GetoptError:
        print('test.py -o <path> -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -o <path> -i <inputfile>')
            sys.exit()
        elif opt in ("-o", "--path"):
            path = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    return [path,inputfile]

if __name__ == "__main__":
    a=main(sys.argv[1:])
    path=a[0]
    inputfile=a[1]

outter_path = path+"/in_v_results"
surv_pic_path = outter_path+"/surv_pics"
train_path = outter_path+"/train_sets"
test_path = outter_path+"/test_sets"
model_path = outter_path+"/models"
os.makedirs(surv_pic_path)
os.makedirs(train_path)
os.makedirs(test_path)
os.makedirs(model_path)

############separate data and label
def sep_dl(lofl):
    labels=[]
    for l in lofl:
        labels.append([l[-1]])
        del l[-1]
    return([lofl,labels]) #[data,label]


############separate data into training set and test set for internal validation
def s_internal_frac(i,f):
    df=pd.read_csv(inputfile)#load data
    feat1=df[(df.feat==1)]#extract data in group1
    feat2=df[(df.feat==2)]#extract data in group2
    #group1
    feat1=feat1.sample(frac=1.0)#disorganize the order of the data
    cut_idx=int(round(f * feat1.shape[0]))#set the ratio of training set vs test set
    feat1_test, feat1_train = feat1.iloc[:cut_idx], feat1.iloc[cut_idx:]#separate traning set and test set
    #group2
    feat2=feat2.sample(frac=1.0)
    cut_idx2=int(round(f * feat2.shape[0]))
    feat2_test, feat2_train = feat2.iloc[:cut_idx2], feat2.iloc[cut_idx2:]
    #joint group1 and 2
    #joint test set
    df_test=pd.concat([feat1_test,feat2_test])
    df_test=df_test.sample(frac=1.0)
    #joint training set
    df_train=pd.concat([feat1_train,feat2_train])
    df_train=df_train.sample(frac=1.0)
    #save data
    df_test.to_csv(test_path+'/test%s.csv'%i)
    df_train.to_csv(train_path+'/train%s.csv'%i)
    #delete useless columns
    test=df_test.drop(columns=['name','sample','days','vital','Num_mutations'])
    train=df_train.drop(columns=['name','sample','days','vital','Num_mutations'])
    te=test.values.tolist()#csv转list
    tr=train.values.tolist()
    for i in te:
        i[-1]=float(i[-1])-1#change group features into 0/1
    for i in tr:
        i[-1]=float(i[-1])-1
    train_final=sep_dl(tr)#separate data and label
    test_final=sep_dl(te)
    return [train_final,test_final]
      
###########lstm deep learning model 
def lstm_model(trdata,trlabels,tsdata,i):
    trdata=np.array(trdata)
    trlabels=np.array(trlabels)
    tsdata=np.array(tsdata)
    trdata = trdata.reshape((trdata.shape[0], 1, trdata.shape[1]))
    tsdata = tsdata.reshape((tsdata.shape[0], 1, tsdata.shape[1]))
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(output_dim=128, input_shape=(trdata.shape[1], trdata.shape[2]), return_sequences=True))
    model.add(LSTM(output_dim=32,return_sequences=True))
    model.add(LSTM(output_dim=8,return_sequences=False))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    model.fit(trdata, trlabels, epochs=1000)
    model.summary()
    model.save(model_path+'/LSTM_model%s.h5'%i)
       
    pred=model.predict(tsdata)
    res=[]
    for i in pred:
        res.append(round(i[0]))
    return res

##############survival analysis
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
    plt.savefig(surv_pic_path+'/surv_fig%s'%j)
    plt.close()
    results = logrank_test(t0, t1, e0, e1)
    p=results.p_value
    print(p)
    return p

############make histogram plots
def plot_hist(ys):
    pmax=max(ys)
    pmin=min(ys)
    bins=np.linspace(pmin,pmax,num=50,endpoint=True)
    plt.hist(ys,bins=bins,histtype='bar')
    plt.savefig(outter_path+'/p_value_fig.png')
 
##########whole process 
#internal validation
def all_internal(iteration):
    ys=[]
    skip=0
    for j in range(iteration):
        a=s_internal_frac(j,0.4)
        test_patha=test_path+'/test%s.csv'%j
        trdata=a[0][0]#training set data
        trlabels=a[0][1]#training set label
        tsdata=a[1][0]#test set data
        pred=lstm_model(trdata,trlabels,tsdata,j)#cluster results
        res=pd.DataFrame(pred)#joint result and original table
        f=pd.read_csv(test_patha)
        f['res']=res
        f.to_csv(test_patha)
        try: 
            p=surv(f,j)#get p value in survival analysis
            ys.append(p)
        except:
            skip+=1#if the data cannot be analysed,skip+1
    plot_hist(ys)#plot histogram
    #count fraction which p value lower than 0.05  
    n=0    
    for i in ys:
        if i<=0.05:
            n+=1  
    ys.append(n)    
    ys.append(skip)
    ys=pd.DataFrame(ys)
    ys.to_csv(outter_path+'/p_values.csv')

#########running the whole process 
iteration=300 #iterations for internal validation
all_internal(iteration)
