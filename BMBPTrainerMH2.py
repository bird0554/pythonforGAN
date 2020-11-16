# coding:utf-8

'''
env:
    python 3.6
    pybrain 0.3.3
    sklearn 0.18.1
'''
# import sys
# from math import sqrt
# import xlutils.copy as xlucopy
# import xlutils
# from matplotlib.testing import compare
from numpy import random, fabs
# import random
import os
import numpy as np

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.validation import CrossValidator, ModuleValidator
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
import joblib
import xlrd
import xlwt
from xlutils.copy import copy as xlucopy
import matplotlib.pyplot as plt

from BMBackpropTrainer import BMBackpropTrainer
from Contribution import contrib


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

    def __init__(self, _hiddendim=3, _srcnmae='trainer.xlsx', _destxls='trainerdest.xls',
                 _destname='buildBMTrainer'):
        self.hiddendim = _hiddendim
        self.srcname = _srcnmae
        self.destxls = _destxls
        self.destname = _destname
        self.restest = []
        self.rescol = 1
        self.verbose = True
        # 总体容差
        self.finalerror = 0
        # restest = []
        self.__fnn = None
        self.__sy = None
        self.__sx = None
        self.realy = None
        self.weights = []
        self.srcx = []
        self.srcy = []
        self.destx = []
        self.desty = []
        self.sx = None
        self.sy = None
        self.myalg = True
        self.npin = 0

    # 按条件读取excel
    def readexcel(self):
        workbook = xlrd.open_workbook(self.srcname)
        sheet1 = workbook.sheet_by_index(0)
        if (self.verbose):
            print('训练集共：' + str(sheet1.nrows) + '行，' + str(sheet1.ncols) + '列；其中结果为：' + str(self.rescol) + '列')
        self.srcx = []
        self.srcy = []
        if (sheet1.nrows > 1 and sheet1.ncols > self.rescol):
            self.srcx = np.zeros((sheet1.nrows - 1, sheet1.ncols - self.rescol), dtype=np.float)
            self.srcy = np.zeros((sheet1.nrows - 1, self.rescol), dtype=np.float)
            for i in range(sheet1.nrows - 1):
                for j in range(sheet1.ncols):
                    if (j < sheet1.ncols - self.rescol):
                        self.srcx[i][j] = sheet1.cell(i + 1, j).value
                    else:
                        self.srcy[i][j - sheet1.ncols + self.rescol] = sheet1.cell(i + 1, j).value
        return self.srcx.copy(), self.srcy.copy()

    def writeexcel(self, x=None, size=0, savexls=''):
        if x == None:
            x = np.array(self.srcx).copy()
        if savexls == '':
            savexls = self.destxls
        if size > 0:
            workbook = xlwt.Workbook()
            worksheet = workbook.add_sheet('dest')
            self.destx = np.zeros((size, len(x[0])), dtype=np.float)
            # 模拟数据行数：
            for i in range(size):
                for j in range(len(x[0])):
                    cellval = round(random.uniform(min(x[:, j]), max(x[:, j])), 3)
                    self.destx[i][j] = cellval
                    worksheet.write(i, j, cellval)
            workbook.save(savexls)

    def testdest(self):
        # 获取测试数据：
        workbook = xlrd.open_workbook(self.destxls)
        sheet1 = workbook.sheet_by_index(0)
        workbookw1 = xlucopy(workbook)
        sheetw1 = workbookw1.get_sheet(0)
        self.destx = np.zeros((sheet1.nrows, sheet1.ncols), dtype=np.float)
        for i in range(sheet1.nrows):
            for j in range(sheet1.ncols):
                self.destx[i][j] = sheet1.cell(i, j).value
        destx1 = self.sx.transform(self.destx)
        for i in range(sheet1.nrows):
            # for j in range(sheet1.ncols):
            testy = self.sy.inverse_transform(self.__fnn.activate(destx1[i]).reshape(-1, 1))
            self.desty.append(testy)
            sheetw1.write(i, sheet1.ncols, testy[0][0])
        workbookw1.save(self.destxls)
        maxy = max(self.srcy)
        miny = min(self.srcy)

        pmax = []
        pmin = []
        for i in range(sheet1.nrows):
            pmax.append(maxy)
            pmin.append(miny)
        plt.figure()
        plt.subplot(121)
        plt.plot(np.arange(0, sheet1.nrows), pmax, label='max', color='r', linestyle='--')
        plt.plot(np.arange(0, sheet1.nrows), np.array(self.desty).reshape(-1, 1), label='test', color='b',
                 linestyle=':', marker='|')
        plt.plot(np.arange(0, sheet1.nrows), pmin, label='min', color='k', linestyle='--')
        plt.legend()
        plt.xlabel("PointCount")
        plt.ylabel("Rate")
        print('###################################')
        # for i in self.desty:q
        #     if i<pmin[0]:
        # print self.desty
        # print pmax[0]
        # print pmin[0]
        # print 'max:' + str(np.maximum(self.desty, pmax[0]))
        npmax = [i for i in self.desty if i > pmax[0]]
        # print npmax
        # print len(npmax)


        npin = [i for i in self.desty if (i < pmax[0] and i > pmin[0])]
        # print npin
        # print len(npin)

        npmin = [i for i in self.desty if i < pmin[0]]
        # print npmin
        # print len(npmin)
        print(str(float(len(npmin)) / len(self.desty) * 100), '% 小于' + str(pmin[0]))
        self.npin = float(len(npin)) / len(self.desty) * 100
        print(str(float(len(npin)) / len(self.desty) * 100) + '% 在所在区间[' + str(pmin[0]) + ',' + str(pmax[0]) + ']中')
        print(str(float(len(npmax)) / len(self.desty) * 100) + '%  大于' + str(pmax[0]))

        # print 'min:' + str(np.minimum(self.desty, pmin[0]))
        print('###################################')
        # plt.show()

    def buildBMTrainer(self):
        x, y = self.readexcel()
        # 模拟size条数据：
        # self.writeexcel(size=100)
        # resx=contrib(x,0.9)
        # print '**********************'
        # print resx
        # x1=x[:,[3,4,5,6,7,8,9,10,11,0,1,2]]
        # resx1=contrib(x1)
        # print '**********************'
        # print resx1

        self.realy = y
        per = int(len(x))
        # 对数据进行归一化处理（一般来说使用Sigmoid时一定要归一化）
        self.sx = MinMaxScaler()
        self.sy = MinMaxScaler()

        xTrain = x[:per]
        xTrain = self.sx.fit_transform(xTrain)
        yTrain = y[:per]
        yTrain = self.sy.fit_transform(yTrain)

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
        trainer = BMBackpropTrainer(self.__fnn, DS, learningrate=0.0001, verbose=self.verbose)
        if self.myalg:
            trainingErrors = trainer.bmtrain(maxEpochs=10000, verbose=True, continueEpochs=3000, totalError=0.0001)
        else:
            trainingErrors = trainer.trainUntilConvergence(maxEpochs=10000, continueEpochs=3000,
                                                           validationProportion=0.1)
        # CV = CrossValidator(trainer, DS, n_folds=4, valfunc=ModuleValidator.MSE)
        # CV.validate()
        # CrossValidator
        # trainingErrors = trainer.trainUntilConvergence(maxEpochs=10000,continueEpochs=5000, validationProportion=0.1)
        # self.finalError = trainingErrors[0][-2]
        # self.finalerror=trainingErrors[0][-2]
        # if (self.verbose):
        #     print '最后总体容差:', self.finalError
        self.__sy = self.sy
        self.__sx = self.sx
        for i in range(len(xTrain)):
            a = self.sy.inverse_transform(self.__fnn.activate(xTrain[i]).reshape(-1, 1))
            self.restest.append(self.sy.inverse_transform(self.__fnn.activate(xTrain[i]).reshape(-1, 1))[0][0])
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

    def saveresult(self, destname=None):
        if destname == None:
            destname = self.destname
        NetworkWriter.writeToFile(self.__fnn, destname + '.xml')
        joblib.dump(self.__sy, destname + '_sy.pkl', compress=3)
        joblib.dump(self.__sx, destname + '_sx.pkl', compress=3)
        # joblib.dump(sx, 'sx.pkl', compress=3)
        # joblib.dump(sy, 'sy.pkl', compress=3)

        # 将保存的数据读取
        # fnn = NetworkReader.readFrom('BM.xml')
        # sx = joblib.load('sx.pkl')
        # sy = joblib.load('sy.pkl')

    def printresult(self):
        for mod in self.__fnn.modules:
            print("Module:", mod.name)
            if mod.paramdim > 0:
                print("--parameters:", mod.params)
            for conn in self.__fnn.connections[mod]:
                print("-connection to", conn.outmod.name)
                # conn.whichBuffers
                if conn.paramdim > 0:
                    print("- parameters", conn.params)

            if hasattr(self.__fnn, "recurrentConns"):
                print("Recurrent connections")
                for conn in self.__fnn.recurrentConns:
                    print("-", conn.inmod.name, " to", conn.outmod.name)
                    if conn.paramdim > 0:
                        print("- parameters", conn.params)

    def getweight(self):
        self.weights = []
        for mod in self.__fnn.modules:
            for conn in self.__fnn.connections[mod]:
                print("-connection to", conn.outmod.name)
                if (conn.paramdim > 0) and (conn.inmod.name == 'inLayer'):
                    weights1 = conn.params.reshape(conn.indim, conn.outdim)
                    for pw in weights1:
                        dw = 0.0
                        for pw1 in pw:
                            dw += fabs(pw1)
                        self.weights.append(dw)
                    print('weights:', str(self.weights))
                    print("- parameters", conn.params)
        sw = MinMaxScaler()
        sw = sw.fit_transform(np.asarray(self.weights, dtype=float).reshape(-1, 1))
        print('sw:', str(sw))

    def printpilt(self, y, realy, savepng='', show=True):
        # plt.figure()
        plt.subplot(122)
        plt.plot(np.arange(0, len(y)), y, 'ro--', label='predict number')
        plt.plot(np.arange(0, len(y)), realy, 'ko-', label='true number')
        plt.legend()
        plt.xlabel("PointCount")
        plt.ylabel("Rate")
        if savepng != '':
            plt.savefig(savepng + '.png')
        # plt.get_current_fig_manager().frame.Maximize(True)
        # plt.get_current_fig_manager().full_screen_toggle()
        # plt.get_current_fig_manager().window.state('zoomed')
        if show:
            plt.show()
            # plt.close('all')


if __name__ == '__main__':
    # BMTrainer().writeexcel(size=1)
    for k in range(20):
        bmt = BMTrainer()
        # 中间隐含层神经元数目：
        bmt.hiddendim = 24
        bmt.rescol = 1
        bmt.verbose = True
        bmt.myalg = False
        # bmt.srcname = "all.xlsx"
        bmt.buildBMTrainer()
        if bmt.myalg:
            myalg = 'my'
        else:
            myalg = 'nomy'
        # 模拟数据点数：
        size = 1000
        bmt.writeexcel(size=size)
        # 模拟数据测试：
        bmt.testdest()
        # sys.exit()
        # terror = 5.0
        # for i in range(1):
        # bmt = BMTrainer()
        # bmt.hiddendim = 30
        # bmt.rescol = 1
        # bmt.verbose = True
        # bmt.buildBMTrainer()
        # if (bmt.finalError < terror):
        #     bmt.saveresult()
        #     terror = bmt.finalError
        rtest = bmt.restest
        # print '最终容差：' + str(terror)

        print('真实值：', str(bmt.realy.reshape(1, -1)))
        print('最终验证结果：', str(rtest))
        # bmt.getweight()
        # bmt.printresult()
        savefilename = str(bmt.npin) + '_' + myalg + '_' + str(bmt.hiddendim) + '_' + str(size)
        if os.path.exists(savefilename + '.xls'):
            os.remove(savefilename + '.xls')
        os.rename(bmt.destxls, savefilename + '.xls')
        bmt.saveresult(savefilename)
        bmt.printpilt(rtest, bmt.realy, savepng=savefilename, show=False)
        # compare
