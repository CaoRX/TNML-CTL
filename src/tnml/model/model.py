import numpy as np
import tnml.funcs.funcs as funcs
from CTL.tensor.tensor import Tensor
from CTL.tensor.contract.contract import contractTwoTensors
import time
import functools

from CTL.examples.Schimdt import SchimdtDecomposition, matrixSchimdtDecomposition
from CTL.tensor.contract.link import makeLink
import CTL.examples.MPS as mps

class MPSModelOptions:

    def __init__(self, n, m, classes = 10, eta = 0.01, eta0 = 100):
        # self.h = h
        # self.w = w
        self.n = n
        self.m = m
        self.classes = classes
        self.eta = eta
        self.eta0 = eta0

class NumpyMPS:
    def __init__(self, n, virtualDim, physicalDim, chi, predictIndex):
        self.n = n
        self.chi = chi
        self.mps = mps.makeRandomMPS(self.n, virtualDim = virtualDim, physicalDim = physicalDim, chi = chi)
        self.tensors = [self.mps.getTensor(0).toTensor(labels = ['r', 'o'])] + \
            [self.mps.getTensor(i).toTensor(labels = ['l', 'r', 'o']) for i in range(1, self.n - 1)] + \
            [self.mps.getTensor(self.n - 1).toTensor(labels = ['l', 'o'])]
        # del self.mps
        del self.mps._tensors
        del self.mps
        self.predictIndex = predictIndex
        self.activeIndex = None
        self.canonicalize(normalizeFlag = True, index = self.predictIndex)
        # self.eta = eta

    def normalize(self, index):
        self.tensors[index] /= np.linalg.norm(self.tensors[index])

    def canonicalize(self, index, normalizeFlag = False):
        if (self.activeIndex == index):
            return 
        # canonicalize from 
        for i in range(index):
            # self.tensors[i], self.tensors[i + 1] = 
            self.moveActive(i, i + 1)
            if (normalizeFlag):
                self.normalize(i + 1)
        for i in range(self.n - 1, index, -1):
            self.moveActive(i, i - 1)
            if (normalizeFlag):
                self.normalize(i - 1)

        self.activeIndex = index

    def checkCanonicalize(self):
        if self.activeIndex is None:
            print("Error: no active index provided, cannot check canonicalize")
            return 

        error = 0
        
        leftRes = None
        for i in range(self.activeIndex):
            if (leftRes is None):
                leftRes = self.tensors[i] @ self.tensors[i].T # ['r', 'o']
            else:
                leftRes = np.einsum('ij,ipk,jqk->pq', leftRes, self.tensors[i], self.tensors[i])
                # ['l', 'r', 'o']
            # print('identity error = {}'.format(funcs.identityError(leftRes)))
            errorLeft = funcs.identityError(leftRes)
            error = max(error, errorLeft)
            # print('identity error at index {}(left) = {}'.format(i, errorLeft))
        
        rightRes = None
        for i in range(self.n - 1, self.activeIndex, -1):
            if (rightRes is None):
                rightRes = self.tensors[i] @ self.tensors[i].T # ['l', 'o']
            else:
                rightRes = np.einsum('ij,pik,qjk->pq', rightRes, self.tensors[i], self.tensors[i])
            errorRight = funcs.identityError(rightRes)
            error = max(error, errorRight)
            # print('identity error at index {}(right) = {}'.format(i, errorRight))
        return error

    def moveActive(self, i, j):
        assert (j == i + 1) or (j == i - 1), "Error: NumpyMPS must have j to be either (i - 1) or (i + 1), obtain i = {}, j = {}".format(i, j)

        if (j == i - 1):
            hasLeftFlag = (j > 0)
            hasRightFlag = (i < self.n - 1)

            if (not hasLeftFlag):
                leftLabels = ['r', 'o']
            else:
                leftLabels = ['l', 'r', 'o']
            left = Tensor(data = self.tensors[j], labels = leftLabels)

            if (not hasRightFlag):
                rightLabels = ['l', 'o']
            else:
                rightLabels = ['l', 'r', 'o']
            right = Tensor(data = self.tensors[i], labels = rightLabels)

            makeLink('r', 'l', left, right)

            u, s, v = SchimdtDecomposition(right, left, self.chi)
            # print('s = {}'.format(s))
            sv = contractTwoTensors(s, v)
            self.tensors[i] = u.toTensor(labels = rightLabels)
            self.tensors[j] = sv.toTensor(labels = leftLabels)

        else:
            hasLeftFlag = (i > 0)
            hasRightFlag = (j < self.n - 1)

            if (not hasLeftFlag):
                leftLabels = ['r', 'o']
            else:
                leftLabels = ['l', 'r', 'o']
            left = Tensor(data = self.tensors[i], labels = leftLabels)

            if (not hasRightFlag):
                rightLabels = ['l', 'o']
            else:
                rightLabels = ['l', 'r', 'o']
            right = Tensor(data = self.tensors[j], labels = rightLabels)

            makeLink('r', 'l', left, right)
            u, s, v = SchimdtDecomposition(left, right, self.chi)
            # print('s = {}'.format(s))
            sv = contractTwoTensors(s, v)
            self.tensors[i] = u.toTensor(labels = leftLabels)
            self.tensors[j] = sv.toTensor(labels = rightLabels)

        if self.activeIndex == i:
            self.activeIndex = j

    def leftReduce(self, x, index = None):
        # reduce all tensors up to index)
        # output: vector

        if index is None:
            index = self.predictIndex

        assert (index <= self.predictIndex), 'cannot reduce up to index {}: containing predict index {}.'.format(index, self.predictIndex)

        leftSumOutArrays = [tensor @ vec for tensor, vec in zip(self.tensors[:index], x[:index])]
        return functools.reduce(lambda a, b: a @ b, leftSumOutArrays)

    def rightReduce(self, x, index = None):
        # reduce all tensors down to (index
        if index is None:
            index = self.predictIndex

        assert (index >= self.predictIndex), 'cannot reduce down to index {}: containing predict index {}.'.format(index, self.predictIndex)
        rightSumOutArrays = [tensor @ vec for tensor, vec in zip(self.tensors[(index + 1):], x[index:])]
        rightSumOutArrays = [funcs.transpose(x) for x in reversed(rightSumOutArrays)]
        return functools.reduce(lambda a, b: a @ b, rightSumOutArrays)

    def getPredictTensor(self):
        return self.tensors[self.predictIndex]

    def swapTensors(self, i):
        # swap i and i + 1
        assert (i < self.n - 1) and (i >= 0), "Tensor {} and {} cannot be swapped: index invalid.".format(i, i + 1)

        hasLeftFlag = (i > 0)
        hasRightFlag = (i + 1 < self.n - 1)
        
        # print('before: left shape = {}, right shape = {}'.format(self.tensors[i].shape, self.tensors[i + 1].shape))

        leftOSize = self.tensors[i].shape[-1]
        if (hasLeftFlag):
            leftLSize = self.tensors[i].shape[0]
        
        rightOSize = self.tensors[i + 1].shape[-1]
        if (hasRightFlag):
            rightRSize = self.tensors[i + 1].shape[1]

        if (not hasLeftFlag):
            leftLabels = ['r', 'o']
        else:
            leftLabels = ['l', 'r', 'o']
        left = Tensor(data = self.tensors[i], labels = leftLabels)

        if (not hasRightFlag):
            rightLabels = ['l', 'o']
        else:
            rightLabels = ['l', 'r', 'o']
        right = Tensor(data = self.tensors[i + 1], labels = rightLabels)

        makeLink('r', 'l', left, right)
        left, _, right = SchimdtDecomposition(left, right, self.chi, squareRootSeparation = True, swapLabels = (['o'], ['o']))

        # if (not hasLeft):
        #     newLeftShape = (left.shape[-1], leftOSize)
        # else:
        #     newLeftShape = 

        if self.predictIndex == i:
            self.predictIndex = i + 1
        elif self.predictIndex == i + 1:
            self.predictIndex = i

        left = left.toTensor(labels = leftLabels)
        right = right.toTensor(labels = rightLabels)
        # print('after: left shape = {}, right shape = {}'.format(left.shape, right.shape))

        self.tensors[i] = left
        self.tensors[i + 1] = right

        # leftEndFlag = (i == 0)
        # rightEndFlag = (i == self.n - 1)

    def optimize(self, trainX, trainY, eta, debugFlag = False):
        # x: mapped to vectors
        # y: mapped to onehot
        # self.canonicalize(self.predictIndex)
        if (self.predictIndex <= 1) or (self.predictIndex >= self.n - 2):
            return # on the boundary

        print('updating on index {}'.format(self.predictIndex))
        
        ii = self.predictIndex
        # if debugFlag:
            # print(self.tensors[ii - 1], self.tensors[ii], self.tensors[ii + 1])
        D = np.einsum('ajb,jkc,ked->abcde', self.tensors[ii - 1], self.tensors[ii], self.tensors[ii + 1])
        # { a: left, b: input left, c: output, d: input right, e: right}

        n = len(trainX)

        deltaD = None
        
        for i in range(n):
            # print(trainX[i].shape)
            left = self.leftReduce(trainX[i], ii - 1)
            right = self.rightReduce(trainX[i], ii + 1)
            iLeft = trainX[i][ii - 1]
            iRight = trainX[i][ii]
            if debugFlag:
                print('left norm = {}'.format(np.linalg.norm(left)))
                print('right norm = {}'.format(np.linalg.norm(right)))
                print('iLeft norm = {}'.format(np.linalg.norm(iLeft)))
                print('iRight norm = {}'.format(np.linalg.norm(iRight)))

            # print(left.shape, iLeft.shape, iRight.shape, right.shape, D.shape)

            output = np.einsum('i,j,k,l,ijokl->o', left, iLeft, iRight, right, D)
            delta = trainY[i] - output 

            if deltaD is None:
                deltaD = np.einsum('i,j,k,l,o->ijokl', left, iLeft, iRight, right, delta - output)
            else:
                deltaD += np.einsum('i,j,k,l,o->ijokl', left, iLeft, iRight, right, delta - output)
        
        # print('D = {}, deltaD = {}'.format(D, deltaD))
        if (debugFlag):
            print('norm(D) = {}, norm(deltaD) = {}'.format(np.linalg.norm(D), np.linalg.norm(deltaD)))
        D += eta * deltaD

        D = np.reshape(D, (len(left), -1))
        leftTensor, rightPart = matrixSchimdtDecomposition(D, len(iLeft), chi = self.chi)
        rightPart = rightPart.reshape(leftTensor.shape[-1] * len(output), -1).T
        rightPart = rightPart.reshape(len(iRight), -1)
        rightTensor, predictTensor = matrixSchimdtDecomposition(rightPart, len(right), chi = self.chi)

        predictTensor = predictTensor.reshape(rightTensor.shape[-1], leftTensor.shape[-1], len(output))

        # print('left tensor shape = {}'.format(leftTensor.shape))
        # print('right tensor shape = {}'.format(rightTensor.shape))
        # print('predict tensor shape = {}'.format(predictTensor.shape))

        # leftTensor: (left, iLeft, m)
        # rightTensor: (iRight, right, m)
        # predictTensor: (right, left, output)
        self.tensors[ii] = np.moveaxis(predictTensor, [0, 1], [1, 0])
        self.tensors[ii - 1] = np.moveaxis(leftTensor, [1, 2], [2, 1])
        self.tensors[ii + 1] = np.moveaxis(rightTensor, [0, 1, 2], [2, 1, 0])
        # print(self.tensors[ii - 1].shape, self.tensors[ii].shape, self.tensors[ii + 1].shape)

    def update(self, trainX, trainY, eta, debugFlag = False):
        # print('shape before canonicalize: {}, {}'.format(self.tensors[0].shape, self.tensors[1].shape))
        self.canonicalize(self.predictIndex)
        # print('shape begin: {}, {}'.format(self.tensors[0].shape, self.tensors[1].shape))
        while (self.predictIndex > 0):
            self.swapTensors(self.predictIndex - 1)
            self.moveActive(self.predictIndex + 1, self.predictIndex)
        for i in range(1, self.n - 1):
            # update tensors[i], tensors[i + 1]
            # print(self.tensors[0].shape, self.tensors[1].shape)
            self.swapTensors(i - 1)
            # print(self.tensors[0].shape, self.tensors[1].shape)
            self.moveActive(i - 1, i)
            # error = self.checkCanonicalize()
            # print('canonical error = {}'.format(error))
            # if (error > 1e-5):
            #     return
            self.optimize(trainX, trainY, eta, debugFlag = debugFlag)
        
        # now predictIndex = self.n - 2
        for i in range(self.n - 2, 1, -1):
            self.swapTensors(i - 1)
            self.moveActive(i, i - 1)
            # error = self.checkCanonicalize()
            # print('canonical error = {}'.format(error))
            # if (error > 1e-5):
            #     return
            self.optimize(trainX, trainY, eta, debugFlag = debugFlag)
        
        self.swapTensors(0)
        

class MPSModel:

    # contents of model: an MPS of (n + 1) tensors
    # n physical legs for input(dimension-2), 1 for output

    def mpsInitialize(self):
        # left = self.n // 2
        # right = self.n - left 

        # physicalDims = [2] * left + [self.options.classes] + [2] * right
        physicalDims = [self.options.classes] + [2] * self.n
        virtualDims = self.options.m

        # self.mpsCTL = mps.makeRandomMPS(self.n + 1, virtualDim = virtualDims, physicalDim = physicalDims, chi = self.options.m)
        # mpsCTLTensors = [self.mpsCTL.getTensor(0).toTensor(labels = ['r', 'o'])] + \
        #     [self.mpsCTL.getTensor(i).toTensor(labels = ['l', 'r', 'o']) for i in range(1, self.n)] + \
        #     [self.mpsCTL.getTensor(self.n).toTensor(labels = ['l', 'o'])]
        # self.mps = [self.mps.getTensor(0).toTensor(labels = ['r', 'o'])] + \
        #     [self.mps.getTensor(i).toTensor(labels = ['l', 'r', 'o']) for i in range(1, self.n)] + \
        #     [self.mps.getTensor(self.n).toTensor(labels = ['l', 'o'])]
        self.mps = NumpyMPS(n = self.n + 1, virtualDim = virtualDims, physicalDim = physicalDims, chi = self.options.m, predictIndex = 0)
        # self.mps.tensors = mpsCTLTensors
        
        # self.predictIndex = left

        self.validationLoss = []
        self.trainingLoss = []
    def __init__(self, options : MPSModelOptions):
        # self.mps = mps.makeRandomMPS()
        self.options = options
        # self.zigzag = funcs.zigzagOrder(self.options.h, self.options.w)

        # self.n = self.options.h * self.options.w
        self.n = self.options.n

        # self.predictIndex = -1
        self.mpsInitialize()

    @property
    def predictIndex(self):
        return self.mps.predictIndex

    def predictLabel(self, x) -> int:
        # print(x)
        return self.predict(x).argmax()

    def predict(self, x):
        # TODO : predict vector with mps
        # return np.random.rand(self.options.classes)
        # tensors = [self.mps.getTensor(i).copy() for i in range(self.n + 1)]
        # predictMPS = mps.FreeBoundaryMPS(tensors, inplace = True)


        # copyTimeStart = time.time()
        # tensors = self.mpsCTL.copyOfTensors()
        # copyTimeEnd = time.time()
        # print('copy time = {} seconds.'.format(copyTimeEnd - copyTimeStart))

        # sumOutTimeStart = time.time()

        # currIndex = 0
        # for i in range(self.n + 1):
        #     if (i != self.predictIndex):
        #         tensors[i].sumOutLegByLabel('o', weights = funcs.singleFeatureMapToVector(rawX[currIndex]))
        #         currIndex += 1

        # sumOutTimeEnd = time.time()
        # print('sum out time = {} seconds.'.format(sumOutTimeEnd - sumOutTimeStart))


        # productTimeStart = time.time()
        
        # res = tensors[self.predictIndex]
        # if (self.predictIndex > 0):
        #     leftRes = tensors[0]
        #     for i in range(1, self.predictIndex):
        #         leftRes = contractTwoTensors(leftRes, tensors[i])
        #     res = contractTwoTensors(leftRes, res)
        
        # if (self.predictIndex < self.n):
        #     rightRes = tensors[self.n]
        #     for i in range(self.n - 1, self.predictIndex, -1):
        #         rightRes = contractTwoTensors(rightRes, tensors[i])
        #     res = contractTwoTensors(res, rightRes)
        # # print(res.a)

        # productTimeEnd = time.time()
        # print('product time = {} seconds.'.format(productTimeEnd - productTimeStart))

        # directTimeStart = time.time()
        # arrays = [self.mps.getTensor(0).toTensor(labels = ['r', 'o'])] + \
            # [self.mps.getTensor(i).toTensor(labels = ['l', 'r', 'o']) for i in range(1, self.n)] + \
            # [self.mps.getTensor(self.n).toTensor(labels = ['l', 'o'])]

        # directTimeStart = time.time()
        # arrays = self.mps.tensors

        # leftArrays = arrays[:self.predictIndex]
        # rightArrays = arrays[(self.predictIndex + 1):]
        # # print([x.shape for x in rightArrays])
        # # print(len(rightArrays))

        # leftX = [funcs.singleFeatureMapToVector(xx) for xx in x[:self.predictIndex]]
        # rightX = [funcs.singleFeatureMapToVector(xx) for xx in x[self.predictIndex:]]
        # # print(len(rightX))

        # # print([(x.shape, y.shape) for x, y in zip(leftArrays, leftX)])
        
        # leftSumOutArrays = [tensor @ vec for tensor, vec in zip(leftArrays, leftX)]
        # rightSumOutArrays = [tensor @ vec for tensor, vec in zip(rightArrays, rightX)]
        # rightSumOutArrays.reverse()

        # rightSumOutArrays = [funcs.transpose(x) for x in rightSumOutArrays]

        # print([x.shape for x in leftSumOutArrays])
        # print([y.shape for y in rightSumOutArrays])

        # directA = arrays[self.predictIndex]
        directA = self.mps.getPredictTensor()
        # x = [funcs.singleFeatureMapToVector(xx) for xx in rawX]
        leftFlag = self.predictIndex > 0
        rightFlag = self.predictIndex < self.n
        if leftFlag:
            # leftA = functools.reduce(lambda a, b: a @ b, leftSumOutArrays)
            leftA = self.mps.leftReduce(x)
            if rightFlag:
                directA = np.einsum('i,ijk->jk', leftA, directA)
            else:
                directA = np.einsum('i,ij->j', leftA, directA)
        if rightFlag:
            # rightA = functools.reduce(lambda a, b: a @ b, rightSumOutArrays)
            rightA = self.mps.rightReduce(x)
            directA = np.einsum('i,ij->j', rightA, directA)


        # directTimeEnd = time.time()
        # print('direct calculation time = {} seconds.'.format(directTimeEnd - directTimeStart))

        # print(directA)
        # print(res.a)

        # return res.a
        return directA

    def lossFunc(self, predictions, labels):
        # print(predictions, labels)
        return 0.5 * np.sum((np.array(predictions) - funcs.getOneHot(labels, self.options.classes)) ** 2) / len(labels)

    def sweep(self, trainX, trainY, epoch = None):
        print('begin epoch {}'.format(epoch))
        if (epoch == 0):
            self.mps.update(trainX, funcs.getOneHot(trainY, self.options.classes), self.options.eta0)
        else:
            self.mps.update(trainX, funcs.getOneHot(trainY, self.options.classes), self.options.eta, debugFlag = False)
        print('end epoch {}'.format(epoch))

    def loss(self, dataX, dataY):
        predictions = [self.predict(x) for x in dataX]
        return self.lossFunc(predictions, labels = dataY)

    def train(self, trainX, trainY, validX = None, validY = None, epochs = 5):

        print('Begin training process:')
        trainX = np.array([[funcs.singleFeatureMapToVector(x) for x in data] for data in trainX])
        if validX is not None:
            validX = np.array([[funcs.singleFeatureMapToVector(x) for x in data] for data in validX])

        trainingLoss = self.loss(trainX, trainY)
        print('training loss at beginning = {}'.format(trainingLoss))
        self.trainingLoss.append(trainingLoss)
        if (validX is not None) and (validY is not None):
            validLoss = self.loss(validX, validY)
            print('validation loss at beginning = {}'.format(validLoss))
            self.validationLoss.append(validLoss)
        for epoch in range(epochs):
            self.sweep(trainX, trainY, epoch = epoch)
            trainingLoss = self.loss(trainX, trainY)
            print('training loss in epoch {} = {}'.format(epoch, trainingLoss))
            self.trainingLoss.append(trainingLoss)
            if (validX is not None) and (validY is not None):
                validLoss = self.loss(validX, validY)
                print('validation loss in epoch {} = {}'.format(epoch, validLoss))
                self.validationLoss.append(validLoss)

    def eval(self, testX, testY):
        testX = np.array([[funcs.singleFeatureMapToVector(x) for x in data] for data in testX])
        predictY = np.array([self.predictLabel(x) for x in testX])
        accuracy = np.count_nonzero(predictY == testY) / len(testY)
        print('accuracy = {}'.format(accuracy))
        print('predict y = {}'.format(predictY))
        print('real y = {}'.format(testY))
        return accuracy