# provide functions to access the mnist data

from tnml.funcs.binarydatabuffer import BinaryDataBuffer
import tnml.funcs.funcs as funcs
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tnml.funcs.imgfuncs as imgfuncs

npzFile = 'data/mnist'

maxGray = 255.0

def load(compressed = False, px = 2, py = None, zzFlag = True):

    compressedSuffix = ''
    if py is None:
        py = px
    if compressed:
        compressedSuffix = '-compressed{}-{}'.format(px, py)
    zzSuffix = ''
    if zzFlag:
        zzSuffix += '-zz'
    npzFilename = npzFile + compressedSuffix + zzSuffix + '.npz'
    if os.path.exists(npzFilename):
        print('loading from npz file: {}'.format(npzFilename))
        try:
            timeBeforeLoading = time.time()
            ds = dict(np.load(npzFilename))
            timeAfterLoading = time.time()
            print('loading time: {} seconds.'.format(timeAfterLoading - timeBeforeLoading))
            return ds
        except:
            print('Error in loading npz file. Return to raw bytes file.')
    
    timeBeforeLoading = time.time()
    filenames = {
        'testX': 't10k-images-idx3-ubyte',
        'testY': 't10k-labels-idx1-ubyte',
        'trainX': 'train-images-idx3-ubyte',
        'trainY': 'train-labels-idx1-ubyte',
    }

    res = dict()
    for dataName in filenames:
        filename = filenames.get(dataName)
        
        file = open(os.path.join('data', filename), mode = 'rb')
        buffer = BinaryDataBuffer(file.read())

        if dataName.endswith('X'):
            magic, n, row, col = buffer.getDataElements('!i', 4, 4)
            print('magic = {}, n = {}, row = {}, col = {}'.format(magic, n, row, col))

            images = np.array([np.array(buffer.getDataElements('B', 1, row * col)).reshape((row, col)) for _ in range(n)]) / maxGray

            h = row // px
            w = col // py 

            if (zzFlag):
                zigzag = funcs.zigzagOrder(h, w)
            if compressed:
                res[dataName] = np.array([imgfuncs.compress(x, px = px, py = py) for x in images])
            else:
                res[dataName] = images
            if zzFlag:
                res[dataName] = np.array([np.ravel(x)[zigzag] for x in res[dataName]])

            # firstImage = np.array(buffer.getDataElements('B', 1, row * col)).reshape((row, col))
            # # print('firstImage = {}'.format(firstImage))
            # plt.imshow(firstImage)
            # plt.show()
        
        if dataName.endswith('Y'):
            magic, n = buffer.getDataElements('!i', 4, 2)
            print('magic = {}, n = {}'.format(magic, n))

            # firstLabels = buffer.getDataElements('B', 1, 10)
            # print('first 10 labels = {}'.format(firstLabels))

            labels = np.array(buffer.getDataElements('B', 1, n), dtype = np.ubyte)
            res[dataName] = labels

    print('finish loading, saving to npz file...')
    np.savez(npzFilename, **res)
    print('finish saving to npz file {}'.format(npzFilename))
    timeAfterLoading = time.time()
    print('loading time: {} seconds.'.format(timeAfterLoading - timeBeforeLoading))

    return res