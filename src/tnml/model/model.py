import CTL.examples.MPS as mps
import numpy as np
import tnml.funcs.funcs as funcs
from CTL.tensor.contract.contract import contractTwoTensors

class MPSModelOptions:

    def __init__(self, n, m, classes = 10):
        # self.h = h
        # self.w = w
        self.n = n
        self.m = m
        self.classes = classes

class MPSModel:

    # contents of model: an MPS of (n + 1) tensors
    # n physical legs for input(dimension-2), 1 for output

    def mpsInitialize(self):
        left = self.n // 2
        right = self.n - left 

        physicalDims = [2] * left + [self.options.classes] + [2] * right
        virtualDims = self.options.m

        self.mps = mps.makeRandomMPS(self.n + 1, virtualDim = virtualDims, physicalDim = physicalDims, chi = self.options.m)
        self.predictIndex = left

        self.validationLoss = []
        self.trainingLoss = []
    def __init__(self, options : MPSModelOptions):
        # self.mps = mps.makeRandomMPS()
        self.options = options
        # self.zigzag = funcs.zigzagOrder(self.options.h, self.options.w)

        # self.n = self.options.h * self.options.w
        self.n = self.options.n

        self.predictIndex = -1
        self.mpsInitialize()

    def predictLabel(self, x) -> int:
        return self.predict(x).argmax()

    def predict(self, x):
        # TODO : predict vector with mps
        # return np.random.rand(self.options.classes)
        # tensors = [self.mps.getTensor(i).copy() for i in range(self.n + 1)]
        # predictMPS = mps.FreeBoundaryMPS(tensors, inplace = True)
        tensors = self.mps.copyOfTensors()
        currIndex = 0
        for i in range(self.n + 1):
            if (i != self.predictIndex):
                tensors[i].sumOutLegByLabel('o', weights = funcs.singleFeatureMapToVector(x[currIndex]))
                currIndex += 1
        
        res = tensors[self.predictIndex]
        if (self.predictIndex > 0):
            leftRes = tensors[0]
            for i in range(1, self.predictIndex):
                leftRes = contractTwoTensors(leftRes, tensors[i])
            res = contractTwoTensors(leftRes, res)
        
        if (self.predictIndex < self.n):
            rightRes = tensors[self.n]
            for i in range(self.n - 1, self.predictIndex, -1):
                rightRes = contractTwoTensors(rightRes, tensors[i])
            res = contractTwoTensors(res, rightRes)
        # print(res.a)
        return res.a

    def lossFunc(self, predictions, labels):
        return 0.5 * np.sum((np.array(predictions) - funcs.getOneHot(labels, self.options.classes)) ** 2) / len(labels)

    def sweep(self, trainX, trainY, epoch = None):
        pass

    def loss(self, dataX, dataY):
        predictions =[self.predict(x) for x in dataX]
        return self.lossFunc(predictions, labels = dataY)

    def train(self, trainX, trainY, validX = None, validY = None, epochs = 5):

        print('Begin training process:')
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
        predictY = np.array([self.predictLabel(x) for x in testX])
        accuracy = np.count_nonzero(predictY == testY) / len(testY)
        print('accuracy = {}'.format(accuracy))
        print('predict y = {}'.format(predictY))
        return accuracy