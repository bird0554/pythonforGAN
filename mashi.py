#coding=utf-8
from __future__ import division
import sys
import numpy as np
reload(sys)
# sys.setdefaultencoding('utf-8')

def mashi_distance(x,y):
  print x
  print y
  #马氏距离要求样本数要大于维数，否则无法求协方差矩阵
  #此处进行转置，表示10个样本，每个样本2维
  X=np.vstack([x,y])
  print X
  XT=X.T
  print XT
  #方法一：根据公式求解
  S=np.cov(X)  #两个维度之间协方差矩阵
  SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
  #马氏距离计算两个样本之间的距离，此处共有4个样本，两两组合，共有6个距离。
  n=XT.shape[0]
  d1=[]
  for i in range(0,n):
    for j in range(i+1,n):
      delta=XT[i]-XT[j]
      d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))
      print d
      d1.append(d)
if __name__ == '__main__':
  # 第一列
  x = [3, 5, 2, 8]
  # 第二列
  y = [4, 6, 2, 4]
  mashi_distance(x,y)