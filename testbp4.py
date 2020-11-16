# coding=utf-8
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from pybrain.structure import *
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError


def generate_data():
    """generate original data of u and y"""
    u = np.random.uniform(-1, 1, 200)
    y = []
    former_y_value = 0
    for i in np.arange(0, 200):
        y.append(former_y_value)
        next_y_value = (29.0 / 40.0) * np.sin((16.0 * u[i] + 8.0 * former_y_value) / (3.0 + 4.0 * (u[i] ** 2.0) \
                                                                                      + 4.0 * (former_y_value ** 2.0))) \
                       + (2.0 / 10.0) * u[i] + (2.0 / 10.0) * former_y_value
        former_y_value = next_y_value
    return u, y


# obtain the original data
u, y = generate_data()
# percentError()

# createa neural network
fnn = FeedForwardNetwork()

# create three layers, input layer:2 input unit; hidden layer: 10 units; output layer: 1 output
inLayer = LinearLayer(1, name='inLayer')
hiddenLayer0 = SigmoidLayer(300, name='hiddenLayer0')
outLayer = LinearLayer(1, name='outLayer')

# add three layers to the neural network
fnn.addInputModule(inLayer)
fnn.addModule(hiddenLayer0)
fnn.addOutputModule(outLayer)

# link three layers
in_to_hidden0 = FullConnection(inLayer, hiddenLayer0)
hidden0_to_out = FullConnection(hiddenLayer0, outLayer)

# add the links to neural network
fnn.addConnection(in_to_hidden0)
fnn.addConnection(hidden0_to_out)

# make neural network come into effect
fnn.sortModules()

# definite the dataset as two input , one output
DS = SupervisedDataSet(1, 1)

# add data element to the dataset
for i in np.arange(199):
    DS.addSample(u[i], y[i])

# you can get your input/output this way
X = DS['input']
Y = DS['target']

# split the dataset into train dataset and test dataset
dataTrain, dataTest = DS.splitWithProportion(0.8)
xTrain, yTrain = dataTrain['input'], dataTrain['target']
xTest, yTest = dataTest['input'], dataTest['target']

# train the NN
# we use BP Algorithm
# verbose = True means print th total error
trainer = BackpropTrainer(fnn, dataTrain, verbose=True, learningrate=0.0001)
# set the epoch times to make the NN  fit
trainer.trainUntilConvergence(maxEpochs=10000)

# prediction = fnn.activate(xTest[1])
# print("the prediction number is :",prediction," the real number is:  ",yTest[1])
predict_resutl = []
for i in np.arange(len(xTest)):
    predict_resutl.append(fnn.activate(xTest[i])[0])
print 'predict:' + str(predict_resutl)
print 'true:' + str(yTest)
for mod in fnn.modules:
    print ("Module:", mod.name)
    if mod.paramdim > 0:
        print ("--parameters:", mod.params)
    for conn in fnn.connections[mod]:
        print ("-connection to", conn.outmod.name)
        if conn.paramdim > 0:
            print ("- parameters", conn.params)
    if hasattr(fnn, "recurrentConns"):
        print ("Recurrent connections")
        for conn in fnn.recurrentConns:
            print ("-", conn.inmod.name, " to", conn.outmod.name)
            if conn.paramdim > 0:
                print ("- parameters", conn.params)
# font = {'family': 'MicroSoft Yahei',
#        'weight': 'bold',
#        'size': 12}
# matplotlib.rc("font", **font)

plt.figure()
plt.plot(np.arange(0, len(xTest)), predict_resutl, 'ro--', label='predict')
plt.plot(np.arange(0, len(xTest)), yTest, 'ko-', label='true')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

plt.show()
