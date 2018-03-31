#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import time

font = {'family':'TakaoGothic' }
plt.rc('font', **font)
#5*5の2値画像データ
train1=np.array([-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1])
train2=np.array([-1,1,1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1])
train3=np.array([-1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1])
train4=np.array([1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1])
train5=np.array([-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1])
train6=np.array([1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1])

train=[train1,train2,train3,train4]
#全試行回数counts
counts=10000
#グラフ
tmp=0
x= np.arange(0.05,0.51,0.01)#グラフx軸
y=np.zeros(x.size)#グラフy軸
y2=np.zeros(x.size)
for q in x:
    similalities=0
    accuracy=0
    for count in range(counts):
        #重み
        W=np.zeros((25,25))
        #閾値
        bias=0
        #データを記憶
        for i in range(len(train)):
            W+=np.dot(train[i][:,np.newaxis],train[i][np.newaxis,:])
        W/len(train)
        for i in range(25):#自己結合なし
            W[i,i]=0
            
        #ノイズを加えたデータ（更新される）
        train_noise=[]
        for i in range(len(train)):
            train_noise.append(np.copy(train[i]))
        chosen_num=np.random.randint(0,len(train))#ノイズを加えるデータを選ぶimp
        ##q=0.1###ノイズ確率
        for j in range(25):
            if 0==np.random.choice(2,1,p=[q,1-q])[0]: 
                train_noise[chosen_num][j]=-train_noise[chosen_num][j]
        #ノイズを加えたデータ(更新されない）
        train_noise_before=np.copy(train_noise[chosen_num])

        #更新
        loop=1000#更新を行う回数
        for i in range(25):
            for j in range(loop):
                if train_noise[chosen_num][i]==np.sign(np.dot(W[i,:],train_noise[chosen_num][:,np.newaxis])-bias):
                    break
                else:
                    train_noise[chosen_num][i]=np.sign(np.dot(W[i,:],train_noise[chosen_num][:,np.newaxis])-bias)
            
        #類似度
        similality=np.dot(train_noise[chosen_num],train[chosen_num])/25.0
        similalities+=similality
        if similality==1.0:
            accuracy+=1.0
    y[tmp]=float(similalities/counts)
    y2[tmp]=float(accuracy/counts)
    tmp+=1


print ("similalities={0}".format(similalities))
print ("accuracy={0}".format(accuracy))
plt.plot(x, y,label= "similality")
plt.plot(x,y2,label=u"accuracy")
plt.xlabel("noise")
plt.ylabel(u"accuracy・similality")
plt.title(u"4 patterns")
plt.legend(loc='lower left')
plt.show()
