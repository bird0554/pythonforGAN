# coding=utf-8
# from macpath import split
import sys
from coverage import data
from cv2 import compare
from numpy.lib.shape_base import split
from numpy.ma.extras import corrcoef
from scipy.stats.stats import zscore
from sklearn.svm.libsvm import fit
import numpy as np

li = [[1, 1], [1, 3], [2, 2], [4, 3], [2, 3]]
li1 = np.array(li)
li2 = li1[:, [1, 0]]
matrix = np.mat(li)
matrix2 = np.mat(li2)
print('test!')
print('test2')
print('matrix:\n', matrix)
print(matrix2)
# 求均值
mean_matrix = np.mean(matrix, axis=0)
# print(mean_matrix.shape)
# 减去平均值
Dataadjust = matrix - mean_matrix
# print(Dataadjust.shape)
# 计算特征值和特征向量
covMatrix = np.cov(Dataadjust, rowvar=0)
covMatrix1 = np.cov(matrix, rowvar=0)
covMatrix2 = np.cov(matrix2, rowvar=0)
eigValues, eigVectors = np.linalg.eig(covMatrix)
eigValues1, eigVectors1 = np.linalg.eig(covMatrix1)
eigValues2, eigVectors2 = np.linalg.eig(covMatrix2)
print(eigValues, eigValues1, eigValues2)
# print eigValues1
# print eigValues2
# sys.exit()

testx = [2, 2.2, 2.4, 1.9]
meanx = np.mean(testx)
print(np.cov(testx))
var_exp = [((i - meanx) ** 2) for i in testx]

print(sum(var_exp) / (len(testx) - 1))
# sys.exit()
x = [80, 175, 250, 410]
y = [1.7, 9.9, 2.1, 13.1]
a = []
a.append(y[0] / ((x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3])))
a.append(y[1] / ((x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3])))
a.append(y[2] / ((x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3])))
a.append(y[3] / ((x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2])))
dec = []
dec.append(a[0] + a[1] + a[2] + a[3])
print(dec)
student = ['a', 'b', 'c', 'd']
print(student, '\n', student[1:2], '\n', student[0], '\n', student[-2:-1])

# sys.exit()
X = np.array([[1, 2, 3, 4, 5],
              [5, 6, 7, 8, 6],
              [9, 10, 11, 12, 7],
              [13, 14, 15, 16, 9]], dtype=float)
Y = np.array([[1],
              [2],
              [3],
              [4]])
for x in X:
    print(x[0])
print(X[:, 0])
print(X[:, 4])
print(np.average(X[:, 0]))
print(len(X[0]))
# sys.exit()
y1 = zscore(Y, ddof=1)
yavg = np.average(Y)
print(yavg)
print('***********')
# y2=zscore(Y,1,1)
x1 = zscore(X, ddof=1)
print(y1)
print(x1)
ysum = 0
for i in Y:
    ysum += i
print(ysum)
avgy = float(ysum) / len(Y)
print(avgy)
print('***********')
ysumq = 0.0
for i in Y:
    ysumq += (i - avgy) ** 2
print('ysumq1:' + str(ysumq))
ysumq = ysumq / (len(Y) - 1)
print(ysumq)
ysumq = ysumq ** 0.5
print('ysumq2:' + str(ysumq))
for i in Y:
    i1 = float((i - avgy)) / ysumq
    print(i1)
# z=corrcoef(X,Y)
# print z
# print y2
# data=np.array[X,Y]
# train, test = np.split(data,2)
# model = fit(train.X, train.y)
# predictions = model.predict(test.X)
# skill = compare(test.y, predictions)
