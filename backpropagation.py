#!/usr/bin/env python
# -*- coding: utf-8 -*-

###訓練データをimport
##訓練データの画素値(0~255)を0~1に変換
import numpy as np
import matplotlib.pyplot as plt
"""
train_data_set=np.loadtxt('train-images.txt')/255.0
train_label_set=np.loadtxt('train-labels.txt').astype(np.int32)
test_data_set=np.loadtxt('t10k-images.txt')/255.0
test_label_set=np.loadtxt('t10k-labels.txt').astype(np.int32)
np.save("train_data_set.npy",train_data_set)
np.save("train_label_set.npy",train_label_set)
np.save("test_data_set.npy",test_data_set)
np.save("test_label_set.npy",test_label_set)
"""
train_data_set=np.load("train_data_set.npy")
train_label_set=np.load("train_label_set.npy")
test_data_set=np.load("test_data_set.npy")
test_label_set=np.load("test_label_set.npy")
####################準備##################
#シグモイド関数
def sigmoid(x):
    return 1./(1+np.exp(-x))

#ソフトマックス関数
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

T=np.zeros((10,10))#教師信号(mnistを用いた出力層のユニット数は10)
for i in range(0,10):
    T[i][i]=1.0
unit_num=100#隠れ層ユニット数
H=np.zeros(unit_num)#隠れ層の値
test_H=np.zeros(unit_num)#テストデータでの隠れ層の値
O=np.zeros(10)#出力層の値
test_O=np.zeros(10)#テストデータでの出力層の値
dk=np.zeros(10)#出力層誤差
dj=np.zeros(unit_num)#隠れ層誤差
W=np.random.rand(784,unit_num)#入力層→隠れ層重み(初期値）
W =W*0.16-0.08#初期値を-0.08~0.08に
V=np.random.rand(unit_num,10)#隠れ層→出力層重み（初期値）
V =V*0.16-0.08#初期値を-0.08~0.08に
h=0.05#学習係数 η=0.01~0.5


#W = np.load("W.npy")
#V = np.load("V.npy")

acc = 0#訓練データの識別率
acc_Test=0#テストデータの識別率
keisan_kaisu=100000
y=np.zeros(keisan_kaisu)#グラフy軸
y_Test=np.zeros(keisan_kaisu)
x= np.arange(keisan_kaisu)#グラフx軸

#####収束するまで以下の計算を繰り返す（今回は計算回数を多くとっただけで収束条件は使っていない）##########
for times in range(0,keisan_kaisu):
    ####確率的にデータを選ぶ
    test_num=np.random.randint(10000)
    data_num=np.random.randint(60000)

    test_label=test_label_set[test_num]
    test_data=test_data_set[test_num]

    train_label=train_label_set[data_num]
    train_data=train_data_set[data_num]

    ##noise
    p=0####ノイズ確率
    change=np.random.randint(0,784.0,int(784.0*p))
    train_data[change]=np.random.rand(int(784.0*p))
    #print train_data

    #隠れ層の値を計算
    ##for j in range(0,unit_num):
    ##    H[j]=0
    ##    test_H[j]=0
    ##    u_j=0
    ##    output_j=0
    ##    for i in range(0,784):
    ##        u_j+=train_data[i]*W[i][j]
    ##        output_j+=test_data[i]*W[i][j]
    ##    H[j]=sigmoid(u_j)
    ##    test_H[j]=sigmoid(output_j)
    ##for文を内積計算に直して記述
    ##u_j=np.dot(train_data,W)
    H=sigmoid(np.dot(train_data,W))
    output_j=np.dot(test_data,W)
    test_H=sigmoid(output_j)
        
    #出力層の値を計算
    #u_k=np.zeros(10)
    #output_k=np.zeros(10)
    #for k in range(0,10):
    #    O[k]=0
    #    u_k[k]=0
    #    for j in range(0,unit_num):
    #        u_k[k]+=H[j]*V[j][k]
    #        output_k[k]+=test_H[j]*V[j][k]
    ##for文を内積計算になおして記述
    #u_k=np.dot(H,V)
    O=softmax(np.dot(H,V))
    output_k=np.dot(test_H,V)
    test_O=softmax(output_k)
    

    #誤差逆伝播法(出力層誤差を計算）
    #for k in range(0,10):
    #    O[k]=softmax(u_k)[k]
    #    test_O[k]=softmax(output_k)[k]
    #    dk[k]=O[k]-T[train_label][k]
    ##for文を内積計算になおして記述
    dk=O-T[train_label]


    #誤差逆伝播法（隠れ層の誤差）
    #for j in range(0,unit_num):
    #    temp=0#誤差＊重みの和
    #    for k in range(0,10):
    #        temp+= dk[k]*V[j][k]
    #    dj[j]=temp*H[j]*(1-H[j])
    ##for文を内積計算になおして記述
    dj=np.dot(dk,np.transpose(V))*H*(1.0-H)
    
    #重み更新
    ##for j in range(0,unit_num):
    ##    for k in range(0,10):
    ##        V[j][k]-=h*dk[k]*H[j]
    V=V-h*np.dot(H[:, np.newaxis],dk[np.newaxis,:])
    ##for i in range(0,784):
    ##    for j in range(0,unit_num):
    ##        W[i][j]-=h*dj[j]*train_data[i]
    W=W-h*np.dot(train_data[:, np.newaxis],dj[np.newaxis,:])
    ##識別率計算
    acc += (train_label==np.argmax(O))
    acc_Test+=(test_label==np.argmax(test_O))
    y[times]=float(acc)/float(times+1)
    y_Test[times]=float(acc_Test)/float(times+1)
    ##print times, float(acc)/float(times+1)
    ##, train_label==np.argmax(O)

    if times%10000 == 0:
        print times
print times
print float(acc)/float(times+1)
print  float(acc_Test)/float(times+1)
np.save('W.npy',W)
np.save('V.npy',V)
plt.plot(x, y,label="train")
plt.plot(x,y_Test,label="test")
plt.xlabel("data_num")
plt.ylabel("accuracy")
plt.title("accuracy: unit_num= $100$")
plt.legend(loc='lower right')
plt.show()

