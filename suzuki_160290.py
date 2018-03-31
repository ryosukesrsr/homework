#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import time

###訓練データの準備###
N=200#データ数
#二次元正規分布に従う乱数の生成#
#平均
mean1 = [-1, 3]
mean2 = [1, -1]
#共分散
cov = [[0.8,0.5], [0.5, 0.8]]
#訓練データ
class1=np.random.multivariate_normal(mean1, cov, N/2)
class2=np.random.multivariate_normal(mean2, cov, N/2)
X=np.r_[class1,class2]
#テストデータ
test1=np.random.multivariate_normal(mean1, cov, N/2)
test2=np.random.multivariate_normal(mean2, cov, N/2)
Y=np.r_[test1,test2]

#ラベル付け#
T=np.ones(N)
for i in range(N):
    if i>=N/2:
        T[i]=-1

####学習###
#ラグランジュ乗数
a = np.zeros(N)
#切片
b = 1.0
#学習係数
eta_a = 0.000015#変更したパラメータ
eta_b = 0.1 
#更新回数
count = 5000#変更したパラメータ
#最急降下法
start=time.time()
for k in range(count):
    for i in range(N):
        delta = 1 - (T[i] * X[i]).dot(a * T * X.T).sum() - b * T[i] * a.dot(T)
        a[i] += eta_a * delta
    for i in range(N):
        b += eta_b * a.dot(T) ** 2 / 2
elapsed_time = time.time() - start
print ("学習時間:{0}".format(elapsed_time) + "[秒]")
#サポートベクターを見つける際のしきい値
threshold=0.1
#サポートベクターの要素
index = a > threshold
index_count=1*index
print np.count_nonzero(index_count)#サポートベクター数
w = (a * T).T.dot(X)
#条件を満たすすべてのサポートベクターを平均
b = (T[index] - X[index].dot(w)).mean()


#識別
start = time.time()
Z = np.sign(np.dot(Y,w) + b)
elapsed_time = time.time() - start
print ("識別時間:{0}".format(elapsed_time) + "[秒]")

TP = np.sum(T[(T == 1) & (Z == 1)])
FP = np.sum(Z[(T == -1) & (Z == 1)])
FN = np.sum(T[(T == 1) & (Z == -1)])
precision = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
F_measure = 2*precision*recall/(precision+recall)
print("適合率 : {:.2%}".format(precision))
print("再現率 : {:.2%}".format(recall))
print("F値 : {:.2%}".format(F_measure))

###グラフ描画###
print a
print index
seq = np.arange(-6, 6, 0.02)
plt.figure(figsize = (6, 6))
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.plot(seq, -(w[0] * seq + b) / w[1], 'k-')
plt.plot(X[T ==  1,0], X[T ==  1,1], 'ro')
plt.plot(X[T == -1,0], X[T == -1,1], 'bo')
"""plt.savefig('graph2.png')
plt.show()
plt.figure(figsize = (6, 6))
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.plot(seq, -(w[0] * seq + b) / w[1], 'k-')"""
plt.plot(Y[T ==  1,0], Y[T ==  1,1], 'mx')
plt.plot(Y[T == -1,0], Y[T == -1,1], 'gx')
plt.savefig('testgraph.png')
plt.show()
