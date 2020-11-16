# coding=utf-8
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

import sys

import numpy as np


class GAN():
    def __init__(self):
        self.img_rows = 3
        self.img_cols = 60
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # 构建和编译判别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 构建生成器
        self.generator = self.build_generator()

        # 生成器输入噪音，生成假的图片
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 为了组合模型，只训练生成器
        self.discriminator.trainable = False

        # 判别器将生成的图像作为输入并确定有效性
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # 训练生成器骗过判别器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # np.prod(self.img_shape)=3x60x1
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        # 输入噪音，输出图片
        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        ############################################################
        # 自己数据集此部分需要更改
        # 加载数据集
        data = np.load('data/相对大小分叉.npy')
        data = data[:, :, 0:60]
        # 归一化到-1到1
        data = data * 2 - 1
        data = np.expand_dims(data, axis=3)
        ############################################################

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  训练判别器
            # ---------------------

            # data.shape[0]为数据集的数量，随机生成batch_size个数量的随机数，作为数据的索引
            idx = np.random.randint(0, data.shape[0], batch_size)

            # 从数据集随机挑选batch_size个数据，作为一个批次训练
            imgs = data[idx]

            # 噪音维度(batch_size,100)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 由生成器根据噪音生成假的图片
            gen_imgs = self.generator.predict(noise)

            # 训练判别器，判别器希望真实图片，打上标签1，假的图片打上标签0
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  训练生成器
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印loss值
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # 没sample_interval个epoch保存一次生成图片
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                if not os.path.exists("keras_model"):
                    os.makedirs("keras_model")
                self.generator.save_weights("keras_model/G_model%d.hdf5" % epoch, True)
                self.discriminator.save_weights("keras_model/D_model%d.hdf5" % epoch, True)

    def sample_images(self, epoch):
        r, c = 10, 10
        # 重新生成一批噪音，维度为(100,100)
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # 将生成的图片重新归整到0-1之间
        gen = 0.5 * gen_imgs + 0.5
        gen = gen.reshape(-1, 3, 60)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                xy = gen[cnt]
                for k in range(len(xy)):
                    x = xy[k][0:30]
                    y = xy[k][30:60]
                    if k == 0:
                        axs[i, j].plot(x, y, color='blue')
                    if k == 1:
                        axs[i, j].plot(x, y, color='red')
                    if k == 2:
                        axs[i, j].plot(x, y, color='green')
                        plt.xlim(0., 1.)
                        plt.ylim(0., 1.)
                        plt.xticks(np.arange(0, 1, 0.1))
                        plt.xticks(np.arange(0, 1, 0.1))
                        axs[i, j].axis('off')
                cnt += 1
        if not os.path.exists("keras_imgs"):
            os.makedirs("keras_imgs")
        fig.savefig("keras_imgs/%d.png" % epoch)
        plt.close()

    def test(self, gen_nums=100, save=False):
        self.generator.load_weights("keras_model/G_model4000.hdf5", by_name=True)
        self.discriminator.load_weights("keras_model/D_model4000.hdf5", by_name=True)
        noise = np.random.normal(0, 1, (gen_nums, self.latent_dim))
        gen = self.generator.predict(noise)
        gen = 0.5 * gen + 0.5
        gen = gen.reshape(-1, 3, 60)
        print(gen.shape)
        ###############################################################
        # 直接可视化生成图片
        if save:
            for i in range(0, len(gen)):
                plt.figure(figsize=(128, 128), dpi=1)
                plt.plot(gen[i][0][0:30], gen[i][0][30:60], color='blue', linewidth=300)
                plt.plot(gen[i][1][0:30], gen[i][1][30:60], color='red', linewidth=300)
                plt.plot(gen[i][2][0:30], gen[i][2][30:60], color='green', linewidth=300)
                plt.axis('off')
                plt.xlim(0., 1.)
                plt.ylim(0., 1.)
                plt.xticks(np.arange(0, 1, 0.1))
                plt.yticks(np.arange(0, 1, 0.1))
                if not os.path.exists("keras_gen"):
                    os.makedirs("keras_gen")
                plt.savefig("keras_gen" + os.sep + str(i) + '.jpg', dpi=1)
                plt.close()
        ##################################################################
        # 重整图片到0-1
        else:
            for i in range(len(gen)):
                plt.plot(gen[i][0][0:30], gen[i][0][30:60], color='blue')
                plt.plot(gen[i][1][0:30], gen[i][1][30:60], color='red')
                plt.plot(gen[i][2][0:30], gen[i][2][30:60], color='green')
                plt.xlim(0., 1.)
                plt.ylim(0., 1.)
                plt.xticks(np.arange(0, 1, 0.1))
                plt.xticks(np.arange(0, 1, 0.1))
                plt.show()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=300000, batch_size=32, sample_interval=2000)
# gan.test(save=True)
