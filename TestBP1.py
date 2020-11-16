#coding:utf-8

'''
env:
    python 2.7
    pybrain 0.3.3
    sklearn 0.18.1
'''
import sys
import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

#从sklearn数据集中读取用来模拟的数据
boston = load_boston()
x = boston.data
y = boston.target.reshape(-1,1)
# for i in range(0,x.shape[0]):
#     for j in range(0,x.shape[1]):
#         print (x[i][j])
# print x.shape
# sys.exit();

# for x in x:
#     print x
# print x
# print y
# sys.exit(0)
#直接采用不打乱的方式进行7:3分离训练集和测试集
per = int(len(x) * 0.7)
#对数据进行归一化处理（一般来说使用Sigmoid时一定要归一化）
sx = MinMaxScaler()
sy = MinMaxScaler()
xTrain = x[:per]
# print xTrain[0][0]
xTrain = sx.fit_transform(xTrain)
yTrain = y[:per]
yTrain = sy.fit_transform(yTrain)

xTest = x[per:]
xTest = sx.transform(xTest)
yTest = y[per:]
yTest = sy.transform(yTest)
# print xTest.shape
# for x in xTest:
#     print x
# sys.exit()

#初始化前馈神经网络
fnn = FeedForwardNetwork()

#构建输入层，隐藏层和输出层，一般隐藏层为3-5层，不宜过多
inLayer = LinearLayer(x.shape[1], 'inLayer')

# hiddenLayer = TanhLayer(3, 'hiddenLayer')
hiddenLayer = TanhLayer(12, 'hiddenLayer')
outLayer = LinearLayer(1, 'outLayer')
# hiddenLayer1 = TanhLayer(5, 'hiddenLayer1')
# outLayer = LinearLayer(1, 'outLayer')

#将构建的输出层、隐藏层、输出层加入到fnn中
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer)
# fnn.addModule(hiddenLayer1)
fnn.addOutputModule(outLayer)

#对各层之间建立完全连接
in_to_hidden = FullConnection(inLayer,hiddenLayer )
hidden_to_out = FullConnection(hiddenLayer, outLayer)
# hidden_to_hidden = FullConnection(hiddenLayer,hiddenLayer1 )
# hidden_to_out = FullConnection(hiddenLayer1, outLayer)

#与fnn建立连接
fnn.addConnection(in_to_hidden)
# fnn.addConnection(hidden_to_hidden)
fnn.addConnection(hidden_to_out)
fnn.sortModules()

#初始化监督数据集
DS = SupervisedDataSet(x.shape[1],1)

#将训练的数据及标签加入到DS中
for i in range(len(xTrain)):
    DS.addSample(xTrain[i],yTrain[i])

#采用BP进行训练，训练至收敛，最大训练次数为1000
trainer = BackpropTrainer(fnn, DS, learningrate=0.01, verbose=True)
trainer.trainUntilConvergence(maxEpochs=1000)

# print "1"
# print fnn.activate(x)
# print sy.inverse_transform(fnn.activate(x))
# print sy.inverse_transform(fnn.activate(x))[0]
#在测试集上对其效果做验证
values = []
# sy.inverse_transform()
# for x in xTest:
#     values.append(sy.inverse_transform(fnn.activate(x))[0])
for x in xTest:
    x1=fnn.activate(x)
    x2=sy.inverse_transform(x1.reshape(-1,1))
    values.append(x2[0])
print("2")
#计算RMSE (Root Mean Squared Error)均方差
totalsum=sum(map(lambda x: x ** 0.5,map(lambda x,y: pow(x-y,2), boston.target[per:], values))) / float(len(xTest))
print(totalsum)
print("3")

#将训练数据进行保存
NetworkWriter.writeToFile(fnn, 'pathName.xml')
joblib.dump(sx, 'sx.pkl', compress=3)
joblib.dump(sy, 'sy.pkl', compress=3)

#将保存的数据读取
fnn = NetworkReader.readFrom('pathName.xml')
sx = joblib.load('sx.pkl')
sy = joblib.load('sy.pkl')