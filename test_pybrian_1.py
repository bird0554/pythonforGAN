#!usr/bin/env python
# _*_coding:utf-8_*_
'''
Created on 2017年4月13日
Topic：Building a Network
@author: Stuart斯图尔特
'''
import sys
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer
import math
from pybrain.datasets import SupervisedDataSet

print math.e
print  math.exp(math.e)
print  math.exp(0)
print  math.exp(1)
# sys.exit(0)
# 构建一个神经网络，简单的进行激活并打印各层的名称信息
net = buildNetwork(2, 4, 5, 1)
print net.activate([2, 1])
print "net['in'] = ", net['in']
print "net['hidden0'] = ", net['hidden0']
print "net['out'] = ", net['out']

# 自定义复杂网络，把神经网络的默认隐含层参数设置为Tanh函数而不是Sigmoid函数
# from pybrain.structure import TanhLayer
net = buildNetwork(2, 4, 6, 1, hiddenclass=TanhLayer)
print "net['hidden0_1'] = ", net['hidden0']
print "net['hidden1'] = ", net['hidden1']
print net.activate([1.1, 1 / 2])
# 自定义复杂网络，修改输出层的类型
# from pybrain.structure import  SoftmaxLayer
net = buildNetwork(2, 3, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
print net.activate((2, 3))

net = buildNetwork(2, 3, 1, bias=True)
print net['bias']
