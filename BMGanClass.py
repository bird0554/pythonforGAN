import sys
from numpy import random

import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system

import xlrd
import xlwt


class BMGan:
    def __init__(self):

        tf.set_random_seed(1)
        np.random.seed(1)
        # 设置超参数
        self.batch_size = 64  # 批量大小（数据行数）
        self.lr_g = 0.00001  # 生成器的学习率
        self.lr_d = 0.00001  # 判别器的学习率
        self.n_ideas = 7  # 生成器输入层
        self.art_components = 1  # 生成器输出层
        self.steps = 5000  # 最大迭代次数
        self.srcname = 'trainer.xlsx'  # 设置数据源：
        self.destxls = 'trainerdest.xls'  # 设置存储文件：
        self.rescol = 1  # 数据目的行数
        self.verbose = True  # 是否显示中间过程
        self.paint = []  # 实际点
        self.sess = None
        self.destx = []
        self.desty = []
        self.size = 1000  # 生成目的数据行数
        self.destdata = []

        # 列表解析式代替了for循环，paint_points.shape=(64,15),
        # np.vstack()默认逐行叠加（axis=0）
        self.paint_points = np.vstack([np.linspace(-1, 1, self.art_components) for _ in range(self.batch_size)])

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

    # 保存目的数据：
    def writeexcel(self, data=[], savexls=''):
        if savexls == '':
            if data == []:
                data = self.destdata
            savexls = self.destxls
            workbook = xlwt.Workbook()
            worksheet = workbook.add_sheet('dest')
            # 模拟数据行数：
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    worksheet.write(i, j, data[i][j])
            workbook.save(savexls)

    # 判别器实际数据：
    def artist_works(self, paint=[]):
        if paint == []:
            # a为64个1到2均匀分布抽取的值，shape=(64,1)
            a = np.random.uniform(1, 2, size=self.batch_size)[:, np.newaxis]
            # a = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
            # a = a[:, np.newaxis]
            paintings = a * np.power(self.paint_points, 2) + (a - 1)
        else:
            paintings = paint
        return paintings

    # 模拟数据：
    def simdata(self, srcx=[], size=100):
        if (srcx != [] and size > 0):
            data = np.zeros((size, len(srcx[0])))
            for i in range(size):
                for j in range(len(srcx[0])):
                    data[i, j] = round(random.uniform(min(srcx[:, j]), max(srcx[:, j])), 3)
        return data

    # 训练数据：
    def train(self):

        with tf.variable_scope('Generator'):
            G_in = tf.placeholder(tf.float32, [None, self.n_ideas])  # 随机的ideals（来源于正态分布）
            G_l1 = tf.layers.dense(G_in, 256, tf.nn.relu)
            G_l2 = tf.layers.dense(G_l1, 128, tf.nn.relu)
            G_out = tf.layers.dense(G_l2, self.art_components)  # 得到1个点
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

        with tf.variable_scope('Discriminator'):
            real_art = tf.placeholder(tf.float32, [None, self.art_components], name='real_in')  # 接受实际点
            D_l0 = tf.layers.dense(real_art, 256, tf.nn.relu, name='l')
            D_l1 = tf.layers.dense(D_l0, 128, tf.nn.relu, name='l1')
            prob_artist0 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out')  # 代入实际点，判别器判断实际点的概率

            # 再次利用生成器生成的15个点
            D_l2 = tf.layers.dense(G_out, 256, tf.nn.relu, name='l', reuse=True)
            D_l3 = tf.layers.dense(D_l2, 128, tf.nn.relu, name='l1', reuse=True)  # 接受生成点
            prob_artist1 = tf.layers.dense(D_l3, 1, tf.nn.sigmoid, name='out', reuse=True)  # 代入生成的点，判别器判断概率

        G_loss = tf.reduce_mean(tf.log(1 - prob_artist1))

        D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))  #

        train_G = tf.train.AdamOptimizer(self.lr_g).minimize(
            G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

        train_D = tf.train.AdamOptimizer(self.lr_d).minimize(
            D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # plt.ion()  # 连续画图
        # plt.rcParams['font.family'] = 'STSong'  # 中文宋体
        # plt.rcParams['font.size'] = 15

        for step in range(self.steps):
            artist_paintings = self.artist_works(self.paint)  # 是实际点
            # G_ideas = np.random.randn(self.batch_size, self.n_ideas)  # 生成器点
            G_ideas = self.simdata(self.destx, self.batch_size)
            G_paintings, pa0, pa1, Dl = self.sess.run(
                fetches=[G_out, prob_artist0, prob_artist1, D_loss, train_D, train_G],
                feed_dict={G_in: G_ideas, real_art: artist_paintings})[:4]  # 训练和获取结果
            if (round(pa0.mean(), 3) == 0.5 and round(pa1.mean(), 3) == 0.5):
                break
            if self.verbose:
                print('pa0:', pa0.mean(), '\npa1:', pa1.mean(), '\ndl:', Dl, '\nGP', G_paintings)
        # 记录结果数据：
        resdata = self.simdata(self.destx, self.size)
        self.desty,_,_ = self.sess.run(fetches=[G_out,D_vars,G_vars],
                                   feed_dict={G_in: resdata})[:3]
        # 合并模拟数据和结果数据：
        self.destdata = np.hstack((resdata, self.desty))
        if self.verbose:
            print('D_vars:',dvars,'\nG_vars:'+gvars)
        # print('pa0:', pa0)
        # print('pa1:', pa1)
        # print('Dl:', Dl)
        # print(sess.run([-tf.reduce_mean(tf.log(pa0) + tf.log(1 - pa1))]))

        # if step % 50 == 0:  # 每50步训练画一次图
        #     plt.cla()
        #     plt.plot(self.paint_points[0], G_paintings[0], c='#4AD631', lw=3, label='预测值', )
        #     plt.plot(self.paint_points[0], artist_paintings[0], c='#FAD600', lw=3, label='真实值', )
        #     plt.plot(self.paint_points[0], 2 * np.power(self.paint_points[0], 2) + 1, c='#74BCFF', lw=3, label='上限')
        #     plt.plot(self.paint_points[0], 1 * np.power(self.paint_points[0], 2) + 0, c='#FF9359', lw=3, label='下限')
        #
        #     # print('dsfdsfsdfsdfdsfds\n', pa0.mean())
        #     plt.text(-.5, 2.3, '判别器精度=%.2f (0.5 真实值判别概率)' % pa0.mean(), fontdict={'size': 15})
        #     plt.text(-.5, 2, '判别器精度=%.2f (0.5 生成器模拟值判别概率)' % pa1.mean(), fontdict={'size': 15})
        #     # plt.text(-.5, 1.7, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        #     plt.ylim((0, 3))
        #     # plt.xlim((-2, 2))
        #     plt.legend(loc='upper right', fontsize=12)
        #     plt.draw()
        #     plt.pause(0.01)

        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
    bmg = BMGan()
    x, y = bmg.readexcel()
    xy = np.hstack((x, y))
    # print(np.hstack(x,y))
    # sys.exit()
    xrowcnt = x.shape[0]
    xcolcnt = x.shape[1]
    yrowcnt = y.shape[0]
    ycolcnt = y.shape[1]
    bmg.batch_size = xrowcnt
    bmg.n_ideas = xcolcnt
    bmg.art_components = ycolcnt
    # bmg.paint=xy
    bmg.paint = y
    bmg.lr_d = 0.00001
    bmg.lr_g = 0.00001
    bmg.steps = 6000
    bmg.destx = x
    bmg.size = 1000
    # bmg.simdata(x)
    # bmg.art_components=ycolcnt
    # bmg.artist_works(xy)
    # print('x:', x, ' y:', y)
    #
    # print()
    # sys.exit()
    bmg.train()
    bmg.writeexcel()
