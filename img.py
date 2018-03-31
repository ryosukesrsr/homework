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
train4=np.array([1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1])
train5=np.array([1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1])
train6=np.array([-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1])


train=[train1,train2,train3,train4,train5,train6]
X=np.zeros((6,6))
for i in range(6):
	for j in range(6):	
		X[i][j]=np.dot(train[i],train[j])
print X/25.0
"""#画像表示
train1_img=255*(train1==1).reshape((5,5))
plt.subplot(161)
plt.imshow(train1_img, cmap = 'gray', interpolation = 'none')
plt.title(u"記憶データ1")

train2_img=255*(train2==1).reshape((5,5))
plt.subplot(162)
plt.imshow(train2_img, cmap = 'gray',interpolation = 'none')
plt.title(u"記憶データ2")

train3_img=255*(train3==1).reshape((5,5))
plt.subplot(163)
plt.imshow(train3_img, cmap = 'gray', interpolation = 'none')
plt.title(u"記憶データ3")

train4_img=255*(train4==1).reshape((5,5))
plt.subplot(164)
plt.imshow(train4_img, cmap = 'gray', interpolation = 'none')
plt.title(u"記憶データ4")

train5_img=255*(train5==1).reshape((5,5))
plt.subplot(165)
plt.imshow(train5_img, cmap = 'gray',interpolation = 'none')
plt.title(u"記憶データ5")

train6_img=255*(train6==1).reshape((5,5))
plt.subplot(166)
plt.imshow(train6_img, cmap = 'gray', interpolation = 'none')
plt.title(u"記憶データ6")
plt.show()"""


