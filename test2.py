# coding=utf-8
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

ds = SupervisedDataSet(6, 1)

tf = open('weather.csv', 'r')

for line in tf.readlines():
    try:
        data = [float(x) for x in line.strip().split(',') if x != '']
        indata = tuple(data[:6])
        outdata = tuple(data[6:])
        ds.addSample(indata, outdata)
    except ValueError as e:
        print("error", e, "on line")

n = buildNetwork(ds.indim, 8, 8, ds.outdim, recurrent=True)
t = BackpropTrainer(n, learningrate=0.001, momentum=0.05, verbose=True)
t.trainOnDataset(ds, 10000)
t.testOnData(verbose=True)
data2014 = n.activate([0, 1, 0, 1, 0, 1])
print('data2014', data2014)
