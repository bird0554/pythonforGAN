# coding=utf-8
import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import xlrd


class BMBPTester:
    __fnn = None
    fnnname = 'buildBMTrainer.xml'
    srctestname = 'tester.xlsx'

    def __init__(self, psrctestname='tester.xlsx', pfnnname='buildBMTrainer.xml'):
        self.fnnname = pfnnname
        self.srctestname = psrctestname
        self.__fnn = FeedForwardNetwork()

    def test(self):
        self.__fnn = NetworkReader.readFrom(self.fnnname)
        workbook = xlrd.open_workbook(self.srctestname)
        sheet1 = workbook.sheet_by_index(0)
        x = np.zeros((sheet1.nrows, sheet1.ncols), dtype=np.float)
        for i in range(sheet1.nrows):
            for j in range(sheet1.ncols):
                x[i][j] = sheet1.cell(i, j).value
        stestx = MinMaxScaler()
        xtest = stestx.fit_transform(x)
        sy = joblib.load('sy.pkl')
        print(sy)
        values = []
        for x1 in xtest:
            values.append(sy.inverse_transform(self.__fnn.activate(x1).reshape(-1, 1)))
            print(self.__fnn.activate(x1))
        print(values)
        # for mod in self.__fnn.modules:
        #     print ("Module:", mod.name)
        #     if mod.paramdim > 0:
        #         print ("--parameters:", mod.params)
        #     for conn in self.__fnn.connections[mod]:
        #         print ("-connection to", conn.outmod.name)
        #         if conn.paramdim > 0:
        #             print ("- parameters", conn.params)
        #
        #     if hasattr(self.__fnn, "recurrentConns"):
        #         print ("Recurrent connections")
        #         for conn in self.__fnn.recurrentConns:
        #             print ("-", conn.inmod.name, " to", conn.outmod.name)
        #             if conn.paramdim > 0:
        #                 print ("- parameters", conn.params)
        # print values


if __name__ == '__main__':
    bmtest = BMBPTester()
    bmtest.test()
