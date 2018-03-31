#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import time

#5*5の2値画像データ
train1=np.array([-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1])
#重み
W=np.zeros((25,25))
#閾値
bias=0
#データを記憶
W=np.dot(train1[:,np.newaxis],train1[np.newaxis,:])
for i in range(25):#自己結合なし
    W[i,i]=0
#ノイズを加えたデータ
train1_noise=np.copy(train1)

q=0.2####ノイズ確率
for i in range(25):
    if 0==np.random.choice(2,1,p=[q,1-q])[0]: 
         train1_noise[i]=-train1_noise[i]

train1_noiseBefore=np.copy(train1_noise)


print train1_noiseBefore
#更新
loop=20#更新を行う回数
for i in range(25):
    for j in range(loop):
        print train1_noiseBefore
        train1_noise[i]=np.sign(np.dot(W[i,:],train1_noise[:,np.newaxis])-bias)

print train1_noiseBefore
#画像の表示

train1_img=255*(train1==1).reshape((5,5))
plt.imshow(train1_img, cmap = 'gray', interpolation = 'none')
plt.title("記憶データ")
plt.show()
train1_noiseBefore_img=255*(train1_noiseBefore==1).reshape((5,5))
plt.imshow(train1_noiseBefore_img, cmap = 'gray',interpolation = 'none')
plt.show()
train1_noise_img=255*(train1_noise==1).reshape((5,5))
plt.imshow(train1_noise_img, cmap = 'gray', interpolation = 'none')
plt.show()
