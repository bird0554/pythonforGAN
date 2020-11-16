# coding:utf-8

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
import xlrd
import xlwt
class BMTrainer:
    #隐藏层神经元节点数：
    hiddendim=3
    #读取训练数据源文件：
    srcname='trainer.xlsx'
    #存储训练数据文件：
    destname='buildBMTrainer.xml'
    #源文件中结果列为几列（输出层节点数）
    rescol=1
    #是否显示计算中间迭代过程
    verbose=True
    #总体容差
    finalerror=0
    __fnn=None
    __sy=None
    def __init__(self, _hiddendim=3,_srcnmae='trainer.xlsx',_destname='buildBMTrainer.xml'):
        self.hiddendim=_hiddendim
        self.srcname=_srcnmae
        self.destname=_destname
    def readexcel(self):
        workbook = xlrd.open_workbook(self.srcname)
        sheet1 = workbook.sheet_by_index(0)
        if (self.verbose):
            print('训练集共：'+str(sheet1.nrows) + '行，' + str(sheet1.ncols) + '列；其中结果为：'+str(self.rescol)+'列')
        # data = np.empty()
        # target = np.empty()
        # for i, d in enumerate(sheet1):
        #     data[i] = np.asarray(d[:-1], dtype=np.float64)i
        #     target[i] = np.asarray(d[-1], dtype=np.float64)
        # test = [[0 for i in range(sheet1.nrows)] for j in range(sheet1.ncols)]
        if (sheet1.nrows > 1 and sheet1.ncols > self.rescol):
            x = np.zeros((sheet1.nrows - 1, sheet1.ncols -  self.rescol), dtype=np.float)
            y = np.zeros((sheet1.nrows - 1,  self.rescol), dtype=np.float)
            for i in range(sheet1.nrows - 1):
                for j in range(sheet1.ncols):
                    if (j < sheet1.ncols -  self.rescol):
                        # print sheet1.cell(i + 1, j).value
                        x[i][j] = sheet1.cell(i + 1, j).value
                    else:
                        y[i][j - sheet1.ncols +  self.rescol] = sheet1.cell(i + 1, j).value
        return x, y
    def buildBMTrainer(self):
        # print np.random.rand(1)
        # print np.random.rand(1)
        x, y = self.readexcel()
        # 从sklearn数据集中读取用来模拟的数据
        # boston = load_boston()
        # x = boston.data
        # y = boston.target.reshape(-1, 1)
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
        # 直接采用不打乱的方式进行7:3分离训练集和测试集
        # per = int(len(x) * 0.7)
        per=int(len(x))
        # 对数据进行归一化处理（一般来说使用Sigmoid时一定要归一化）
        sx = MinMaxScaler()
        sy = MinMaxScaler()
        xTrain = x[:per]
        xTrain = sx.fit_transform(xTrain)
        yTrain = y[:per]
        # print yTrain
        yTrain = sy.fit_transform(yTrain)
        # print yTrain
        # print sy.inverse_transform(yTrain)
        # sys.exit()
        # xTest = x[per:]
        # xTest = sx.transform(xTest)
        # yTest = y[per:]
        # yTest = sy.transform(yTest)
        # print xTest.shape
        # for x in xTest:
        #     print x
        # sys.exit()

        # 初始化前馈神经网络
        self.__fnn = FeedForwardNetwork()

        # 构建输入层，隐藏层和输出层，一般隐藏层为3-5层，不宜过多
        inLayer = LinearLayer(x.shape[1], 'inLayer')

        # hiddenLayer = TanhLayer(3, 'hiddenLayer')
        hiddenLayer = TanhLayer(self.hiddendim, 'hiddenLayer')
        outLayer = LinearLayer(self.rescol, 'outLayer')
        # hiddenLayer1 = TanhLayer(5, 'hiddenLayer1')
        # outLayer = LinearLayer(1, 'outLayer')

        # 将构建的输出层、隐藏层、输出层加入到fnn中
        self.__fnn.addInputModule(inLayer)
        self.__fnn.addModule(hiddenLayer)
        # fnn.addModule(hiddenLayer1)
        self.__fnn.addOutputModule(outLayer)

        # 对各层之间建立完全连接
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        # in_to_hidden.setName('in_to_hidden')
        # in_to_hidden._setParameters([0 for i in range(30)])
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        # hidden_to_out.setName('hidden_to_out')
        # hidden_to_out._setParameters([1 for i in range(3)])
        # hidden_to_hidden = FullConnection(hiddenLayer,hiddenLayer1 )
        # hidden_to_out = FullConnection(hiddenLayer1, outLayer)

        # 与fnn建立连接
        self.__fnn.addConnection(in_to_hidden)
        # fnn.addConnection(hidden_to_hidden)
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
        trainer = BackpropTrainer(self.__fnn, DS, learningrate=0.001, verbose=self.verbose)
        trainingErrors= trainer.trainUntilConvergence(maxEpochs=10000)
        self.finalError=trainingErrors[0][-2]
        if (self.verbose):
            print('最后总体容差:',self.finalError)
        self.__sy=sy
        # print "1"
        # print fnn.activate(x)
        for i in range(len(xTrain)):
            print(sy.inverse_transform(self.__fnn.activate(xTrain[i]).reshape(-1,1)))
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
    terror=5.0
    for i in range(1):
        bmt = BMTrainer()
        bmt.hiddendim =20
        bmt.rescol = 1
        bmt.verbose=True
        bmt.buildBMTrainer()
        if (bmt.finalError<terror):
            bmt.saveresult()
            terror=bmt.finalError
    print('最终容差：'+str(terror))

    # buildBMTrainer(10)
