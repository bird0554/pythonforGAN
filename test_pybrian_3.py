#!usr/bin/env python
# _*_coding:utf-8_*_
'''
Created on 2017年4月14日
Topic:Training your Network on your Dataset
@author: Stuart斯图尔特
'''
# 引入建立神经网络所需的相关模块
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer

# 引入建立数据集所需的相关模块
from pybrain.datasets import SupervisedDataSet

# 引入BackpropTrainer反向训练器
from pybrain.supervised.trainers import BackpropTrainer

# 引入之前建好的建立数据集中的module
import test_pybrain_2  # my_pybrain是包名，test_pybrain_2是module名

# import test_pybrain_2

ds = SupervisedDataSet(2,1)
ds.addSample((0,0), (0,))
ds.addSample((0,1), (1,))
ds.addSample((1,0), (1,))
ds.addSample((1,1), (0,))

net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
# trainer = BackpropTrainer(net, test_pybrain_2.ds)  # 应用之前建立的ds数据集
# 通过调用train()方法来对网络进行训练
print trainer.train()
# 通过调用trainUntilConvertgence()方法对网络训练直到收敛
print trainer.trainUntilConvergence()
