# coding:utf-8

'''
env:
    python 2.7
    pybrain 0.3.3
    sklearn 0.18.1
'''
import sys
from numpy import random

import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, TanhLayer, FullConnection, SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import CrossValidator, ModuleValidator
from pybrain.tools.customxml.networkwriter import NetworkWriter
# from pybrain.tools.xml.networkreader import NetworkReader
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import joblib

import xlrd
import xlwt


class BMTrainer:
    # 隐藏层神经元节点数：
    # hiddendim = 3
    # # 读取训练数据源文件：
    # srcname = 'trainer.xlsx'
    # # 存储训练数据文件：
    # destname = 'buildBMTrainer.xml'
    # 源文件中结果列为几列（输出层节点数）
    # rescol = 1
    # 是否显示计算中间迭代过程
    # verbose = True
    # # 总体容差
    # finalerror = 0
    # # restest = []
    # __fnn = None
    # __sy = None

    def __init__(self, _hiddendim=3, _srcnmae='trainer.xlsx', _destname='buildBMTrainer.xml'):
        self.hiddendim = _hiddendim
        self.srcname = _srcnmae
        self.destname = _destname
        self.restest = []
        self.rescol = 1
        self.verbose = True
        # 总体容差
        self.finalerror = 0.0
        self.validationerror = 0.0
        # restest = []
        self.__fnn = None
        self.__sy = None

    def readexcel(self):
        workbook = xlrd.open_workbook(self.srcname)
        sheet1 = workbook.sheet_by_index(0)
        if (self.verbose):
            print ('训练集共：' + str(sheet1.nrows) + '行，' + str(sheet1.ncols) + '列；其中结果为：' + str(self.rescol) + '列')

        if (sheet1.nrows > 1 and sheet1.ncols > self.rescol):
            x = np.zeros((sheet1.nrows - 1, sheet1.ncols - self.rescol), dtype=np.float)
            y = np.zeros((sheet1.nrows - 1, self.rescol), dtype=np.float)
            for i in range(sheet1.nrows - 1):
                for j in range(sheet1.ncols):
                    if (j < sheet1.ncols - self.rescol):
                        # print sheet1.cell(i + 1, j).value
                        x[i][j] = sheet1.cell(i + 1, j).value
                    else:
                        y[i][j - sheet1.ncols + self.rescol] = sheet1.cell(i + 1, j).value
        return x, y

    def buildBMTrainer(self):

        x, y = self.readexcel()

        per = int(len(x))
        # 对数据进行归一化处理（一般来说使用Sigmoid时一定要归一化）
        sx = MinMaxScaler()
        sy = MinMaxScaler()
        xTrain = x[:per]
        xTrain = sx.fit_transform(xTrain)
        yTrain = y[:per]
        yTrain = sy.fit_transform(yTrain)

        # 初始化前馈神经网络
        self.__fnn = FeedForwardNetwork()

        # 构建输入层，隐藏层和输出层，一般隐藏层为3-5层，不宜过多
        inLayer = LinearLayer(x.shape[1], 'inLayer')
        hiddenLayer0 = SigmoidLayer(int(self.hiddendim / 3), 'hiddenLayer0')
        hiddenLayer1 = TanhLayer(self.hiddendim, 'hiddenLayer1')
        hiddenLayer2 = SigmoidLayer(int(self.hiddendim / 3), 'hiddenLayer2')
        outLayer = LinearLayer(self.rescol, 'outLayer')

        # 将构建的输出层、隐藏层、输出层加入到fnn中
        self.__fnn.addInputModule(inLayer)
        self.__fnn.addModule(hiddenLayer0)
        self.__fnn.addModule(hiddenLayer1)
        self.__fnn.addModule(hiddenLayer2)
        self.__fnn.addOutputModule(outLayer)

        # 对各层之间建立完全连接
        in_to_hidden = FullConnection(inLayer, hiddenLayer0)
        hidden_to_hidden0 = FullConnection(hiddenLayer0, hiddenLayer1)
        hidden_to_hidden1 = FullConnection(hiddenLayer1, hiddenLayer2)
        hidden_to_out = FullConnection(hiddenLayer2, outLayer)

        # 与fnn建立连接
        self.__fnn.addConnection(in_to_hidden)
        self.__fnn.addConnection(hidden_to_hidden0)
        self.__fnn.addConnection(hidden_to_hidden1)
        self.__fnn.addConnection(hidden_to_out)
        self.__fnn.sortModules()
        # 初始化监督数据集
        DS = SupervisedDataSet(x.shape[1], self.rescol)

        # 将训练的数据及标签加入到DS中
        # for i in range(len(xTrain)):
        #     DS.addSample(xTrain[i], yTrain[i])
        for i in range(len(xTrain)):
            DS.addSample(xTrain[i], yTrain[i])

        # 采用BP进行训练，训练至收敛，最大训练次数为1000
        trainer = BackpropTrainer(self.__fnn, DS, learningrate=0.01, verbose=self.verbose)
        trainingErrors, validationerrors = trainer.trainUntilConvergence(maxEpochs=1000, continueEpochs=100,
                                                                         validationProportion=0.1)
        # CV = CrossValidator(trainer, DS, n_folds=4, valfunc=ModuleValidator.MSE)
        # CV.validate()
        # CrossValidator
        # trainingErrors = trainer.trainUntilConvergence(maxEpochs=10000,continueEpochs=2000, validationProportion=0.5)
        self.finalerror = trainingErrors[-2]
        self.validationerror = validationerrors[-1]
        # self.validationerror=validationerrros[0][-1]
        # if (self.verbose):
        #     print '最后总体容差:', self.finalError
        self.__sy = sy
        for i in range(len(xTrain)):
            self.restest.append(sy.inverse_transform(self.__fnn.activate(xTrain[i]).reshape(-1, 1)))
            # print sy.inverse_transform(self.__fnn.activate(xTrain[i]).reshape(-1, 1))
            # sys.exit()
            # print sy.inverse_transform(fnn.activate(x))[0]
            # 在测试集上对其效果做验证
            # values = []
            # sy.inverse_transform()
            # for x in xTest:
            #     values.append(sy.inverse_transform(fnn.activate(x))[0])
            # for x in xTest:
            #     x1 = fnn.activate(x)
            #     x2 = sy.inverse_transform(x1.reshape(-1, 1))
            #     values.append(x2[0])
            # print "2"
            # 计算RMSE (Root Mean Squared Error)均方差
            # totalsum = sum(map(lambda x: x ** 0.5, map(lambda x, y: pow(x - y, 2), boston.target[per:], values))) / float(len(xTest))
            # print totalsum
            # print "3"

            # 将训练数据进行保存

    def saveresult(self):
        NetworkWriter.writeToFile(self.__fnn, self.destname)
        joblib.dump(self.__sy, 'sy.pkl', compress=3)
        # joblib.dump(sx, 'sx.pkl', compress=3)
        # joblib.dump(sy, 'sy.pkl', compress=3)

        # 将保存的数据读取
        # fnn = NetworkReader.readFrom('BM.xml')
        # sx = joblib.load('sx.pkl')
        # sy = joblib.load('sy.pkl')

        # for mod in fnn.modules:
        #     print ("Module:", mod.name)
        #     if mod.paramdim > 0:
        #         print ("--parameters:", mod.params)
        #     for conn in fnn.connections[mod]:
        #         print ("-connection to", conn.outmod.name)
        #         if conn.paramdim > 0:
        #             print ("- parameters", conn.params)
        #
        #     if hasattr(fnn, "recurrentConns"):
        #         print ("Recurrent connections")
        #         for conn in fnn.recurrentConns:
        #             print ("-", conn.inmod.name, " to", conn.outmod.name)
        #             if conn.paramdim > 0:
        #                 print ("- parameters", conn.params)


if __name__ == '__main__':
    terror = 5.0
    for i in range(10):
        bmt = BMTrainer()
        bmt.hiddendim = 30
        bmt.rescol = 1
        bmt.verbose = False
        bmt.buildBMTrainer()
        print(str(i) , '最终容差：' , str(bmt.finalerror))
        print(str(i) , '最终验证：' , str(bmt.validationerror))
        #     if (bmt.finalError < terror):
        #         bmt.saveresult()
        #         terror = bmt.finalError
        #         rtest = bmt.restest
        # print '最终容差：' + str(terror)
        # print '最终验证结果：' + str(rtest)
