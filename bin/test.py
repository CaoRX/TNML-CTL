import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(os.path.join(parentdir, 'src'))

import tnml.mnist.mnist as mnist
import tnml.model.model as model
import CTL
import numpy as np
import tnml.funcs.imgfuncs as imgfuncs
import time
import matplotlib.pyplot as plt

zzFlag = True
if __name__ == '__main__':
    CTL.setXP(np)

    # timeBeforeLoad = time.time()
    dataset = mnist.load(compressed = True, px = 2, py = 2, zzFlag = zzFlag)
    # print(dict(dataset))
    print(dataset)

    # model = model.MPSModel(h = 14, w = 14)
    mpsOptions = model.MPSModelOptions(n = 14 * 14, m = 2, classes = 10)
    mpsModel = model.MPSModel(mpsOptions)

    evalBeginTime = time.time()
    evalN = 500
    mpsModel.eval(dataset['testX'][:evalN], dataset['testY'][:evalN])
    evalEndTime = time.time()
    print('eval time of {} data is {} seconds'.format(evalN, evalEndTime - evalBeginTime))

    # compressedImg = imgfuncs.compress(pooledImg, px = 2, py = 2)
    if not zzFlag:
        pooledImg = dataset['trainX'][0]
        plt.imshow(pooledImg)
        plt.show()
    else:
        print('input shape = {}'.format(dataset['trainX'][0].shape))
    # print(imgfuncs.compress(pooledImg, px = 2, py = 2))
    # timeAfterLoad = time.time()

    # np.savez('data/mnist.npz', **dataset)

    # timeBeforeLoadNpz = time.time()
    # ds = np.load('data/mnist.npz')
    # timeAfterLoadNpz = time.time()
    # for key in ds:
    #     print(key, ds[key])

    # # print('raw bytes loading time = {} seconds.'.format(timeAfterLoad - timeBeforeLoad))
    # print('npz loading time = {} seconds'.format(timeAfterLoadNpz - timeBeforeLoadNpz))