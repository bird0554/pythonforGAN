# coding=utf-8
from sklearn.utils import resample
import numpy as np


def scalegirl(samples):
    count = 0.0
    total = samples.size
    for sex in samples:
        if (sex == 0):
            count += 1.0
    print(count)
    return count / (total - count)


boy = (np.ones(1000))
girl = (np.zeros(800))

# girl/boy=0.8

print(girl.shape)
all = np.hstack((boy, girl))
scale = 0.0
iter = 10000
for i in range(iter):
    bootstrapSamples = resample(all, n_samples=100, replace=1)
    print(bootstrapSamples)
    tempscale = scalegirl(bootstrapSamples)
    print(tempscale)
    scale += tempscale
print(scale / iter)
print(all)
