import numpy as np 
import time
import struct
import tnml.funcs.funcs as funcs
from collections import deque

class BinaryDataBuffer:

    def __init__(self, data):
        # print(type(data))
        self.data = deque(data)
        self.sizeOfSizeType = 8
    def getBytes(self, byteN):
        # res = self.data[:byteN]
        res = bytes([self.data.popleft() for _ in range(byteN)])
        # self.data = self.data[byteN:]
        return res
    def getByte(self):
        res = self.data.popleft()
        # self.data = self.data[1:]
        return res
    def getDataElement(self, dataType, dataSize):
        # dataType should in 'i', 'q', 'd'
        return struct.unpack(dataType, self.getBytes(dataSize))[0]
    def getDataElements(self, dataType, dataSize, dataN):
        return [self.getDataElement(dataType, dataSize) for _ in range(dataN)]

    def getSizeType(self):
        size = struct.unpack('q', self.getBytes(self.sizeOfSizeType))[0]
        return size
    def empty(self):
        # return (len(self.data) <= 0)
        return bool(self.data)

    # def getStr(self):
    #     length = self.getSizeType()
    #     strValue = self.getBytes(length)
    #     return length, strValue.decode('ascii')

    # def getData(self, dataSize):
    #     # here we only have int and double types
    #     # so dataSize == 4 means int, while dataSize == 8 means double
    #     if (dataSize == 4):
    #         dataType = 'i'
    #     else:
    #         dataType = 'd'
    #     return self.getDataElement(dataType, dataSize)