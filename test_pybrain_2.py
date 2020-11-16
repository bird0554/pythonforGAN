#!usr/bin/env python
# _*_coding:utf-8_*_
'''
Created on 2017年4月14日
Topic：Building a Dataset
@author: Stuart斯图尔特
'''
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import UnsupervisedDataSet

ds = SupervisedDataSet(3, 1)

ds.addSample((0, 0, 1.2), (0,))
ds.addSample((0, 1, 1.3), (1,))
ds.addSample((1, 0, 2.1), (1,))
ds.addSample((1, 1, 0.9), (0,))
print '检查数据集的长度'
print len(ds)
print'用for循环迭代的方式访问数据集'
for inpt, target in ds:
    print inpt, target
for i,j in ds:
    print i,j
print '直接访问输入字段和标记字段的数组'
print ds['input']
print ds['target']
print '清除数据集'
# ds.clear()
# print ds['input']
# print ds['target']
# coding=utf-8
