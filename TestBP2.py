# coding: utf-8

from collections import Counter

import sys
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import (LinearLayer,
                               TanhLayer,
                               FullConnection,
                               )
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


def generateDS(input_, out, num, xtrain, ytrain):
    alldata = ClassificationDataSet(input_, out, nb_classes=num)

    for x, y in zip(xtrain, ytrain):
        alldata.addSample(x, y)
    tstdata_temp, trndata_temp = alldata.splitWithProportion(.3)

    tstdata = ClassificationDataSet(input_, out, nb_classes=num)
    for n in xrange(tstdata_temp.getLength()):
        tstdata.addSample(*[tstdata_temp.getSample(n)[i] for i in range(2)])

    trndata = ClassificationDataSet(input_, out, nb_classes=num)
    for n in xrange(trndata_temp.getLength()):
        trndata.addSample(*[trndata_temp.getSample(n)[i] for i in range(2)])

    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    return trndata, tstdata


def buildBP(input_, hidden, output, trndata):
    fnn = FeedForwardNetwork()

    inLayer = LinearLayer(input_, 'inLayer')
    hidden0 = TanhLayer(hidden, 'hiddenLayer')
    outLayer = LinearLayer(output, 'outLayer')

    fnn.addInputModule(inLayer)
    fnn.addModule(hidden0)
    fnn.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hidden0)
    hidden_to_out = FullConnection(hidden0, outLayer)

    fnn.addConnection(in_to_hidden)
    fnn.addConnection(hidden_to_out)
    fnn.sortModules()

    trainer = BackpropTrainer(fnn, trndata, verbose=True, learningrate=.001)
    trainer.trainUntilConvergence(maxEpochs=3000)

    return fnn


def test(_p1, _p2="this"):
    print _p1
    print _p1 + _p2
    return (_p1 + _p2)


def result(data, fnn, char='test'):
    value = 0
    length = data.getLength()
    for i in xrange(length):
        predict = fnn.activate(data.getSample(i)[0]).argmax()
        real = data.getSample(i)[1].argmax()
        # print predict, real

        if predict == real:
            value += 1
    print ('%s' % char).center(60, '*')
    print 'accuracy:  %s' % (value / float(length))


if __name__ == '__main__':
    # print test("hahah","kdd")+"%%%%%"
    # sys.exit(0)
    iris = load_iris()
    x, y = iris.data, iris.target
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x)
    # print x_train[:10]

    input_ = x_train.shape[1]
    out = 3
    num = len(Counter(y).keys())
    trndata, tstdata = generateDS(input_, 1, num, x_train, y)
    fnn = buildBP(input_, 30, out, trndata)
    result(tstdata, fnn)
    result(trndata, fnn, 'train')
