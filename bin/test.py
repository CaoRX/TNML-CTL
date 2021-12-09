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

    h = 28
    w = 28
    px = 4
    py = 4

    n = (h // px) * (w // py)

    # timeBeforeLoad = time.time()
    dataset = mnist.load(compressed = True, px = px, py = py, zzFlag = zzFlag)
    # print(dict(dataset))
    # print(dataset)

    # model = model.MPSModel(h = 14, w = 14)
    mpsOptions = model.MPSModelOptions(n = n, m = 20, classes = 10, eta0 = 0.01, eta = 0.01)
    mpsModel = model.MPSModel(mpsOptions)

    trainN = 300
    validN = 100
    trainX, trainY = dataset['trainX'][:trainN], dataset['trainY'][:trainN]
    validX, validY = dataset['testX'][:validN], dataset['testY'][:validN]
    mpsModel.train(trainX, trainY, validX, validY, epochs = 10)

    evalBeginTime = time.time()
    evalN = 5000
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
        plt.plot(mpsModel.trainingLoss, 'o', label = 'training')
        plt.plot(mpsModel.validationLoss, 'o', label = 'validation')
        plt.legend()
        plt.show()
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