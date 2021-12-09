import numpy as np 
import string 
import random
import matplotlib.transforms as mplTransforms
import math

def npzFileExist(seed, port):
    return False

def identityError(a):
    """
    Calculate the distance of matrix a to a 2D eye matrix

    Parameters
    ----------
    a : 2-D ndarray.
    
    Returns
    -------
    float
        The norm of matrix (a - I).
    """
    return np.linalg.norm(a - np.eye(a.shape[0], a.shape[1]))

def sizeOfShape(shape):
    res = 1
    for x in shape:
        res *= x 
    return res

def loadNpzFile(fileName):
    # print('loading {}'.format(fileName))
    data = np.load(fileName, allow_pickle = True)
    # print(fileName)
    # data = np.load(fileName)
    # print('npz keys = {}'.format(list(data.keys())))
    res = dict([])
    # print(data)
    # for key in data:
    #     # print(key, data[key])
    #     print(key)
    for key in data:
        # if (key == 'FP'):
        #     continue

        # print(key)
        res[key] = data[key]
    # print(res)
    return res

def binderCumulant(orderValue):
    return 1.0 - (orderValue['4'] / (3.0 * (orderValue['2'] ** 2)))
def binderSeries(x):
    return np.average(x ** 4) / (np.average(x ** 2) ** 2)
def binderError(orderValue, orderError):
    # print('value = {}, error = {}'.format(orderValue, orderError))
    error2 = orderError['2'] / orderValue['2']
    error4 = orderError['4'] / orderValue['4']
    
    divValue = (orderValue['4'] / (3.0 * (orderValue['2'] ** 2)))
    errorDiv = (error2 * 2 + error4) * divValue
    return errorDiv

def binderErrorByOP(op2, op4):
    error2 = op2['error'] / op2['value']
    error4 = op4['error'] / op4['value']
    
    divValue = (op4['value'] / (3.0 * (op2['value'] ** 2)))
    errorDiv = (error2 * 2 + error4) * divValue
    return errorDiv

def getRawDataFolder(seed):
    return '../../build/data/{}/'.format(seed)

def randomString(n = 10):
	return ''.join(random.choice(string.ascii_letters) for i in range(n))

def getNewString(knownSet):
	res = randomString()
	while (res in knownSet):
		res = randomString()
	return res

def rangeFilter(key, low, high):
    return lambda x, y: (key in x) and ((x[key] > low) and (x[key] < high))
def setFilter(key, valueSet):
    # print(valueSet)
    valueSet = set(valueSet)
    return lambda x, y: (key in x) and (x[key] in valueSet)
def tupleFilter(key, valueSet):
    def inSet(x):
        x = tuple(x)
        print('wn = {}'.format(x))
        # print(tuple(x), valueSet)
        for value in valueSet:
            if (value == x):
                return True
        return False 
    return lambda x, y: (key in x) and (inSet(x[key]))

def inclusiveFilter(key, filter):
    return lambda x, y: (not (key in x)) or filter(x, y)
def doubleSetFilter(key, valueSet, eps = 1e-5):
    def inSet(x):
        for value in valueSet:
            if (np.abs(x - value) < eps):
                return True 
        return False
    return lambda x, y: ((key in x) and (inSet(x[key]))) or ((not (key in x)) and (inSet(0.0)))
def notContainOrZeroFilter(key, eps = 1e-5):
    return lambda x, y: (key not in x) or (abs(x[key]) < eps)
def inverseFilter(filter):
    return lambda x, y: not filter(x, y)
def flagFilter(key, val):
    if (val == 0):
        return notContainOrZeroFilter(key)
    else:
        return inverseFilter(notContainOrZeroFilter(key))

def addrFilter(seed = None, port = None):
    if (seed is not None):
        seed = str(seed)
    if (port is not None):
        port = str(port)
    print('seed = {}, port = {}'.format(seed, port))
    def filter(data, addr):
        return ((seed is None) or (addr['seed'] == seed)) and ((port is None) or (addr['port'] == port))
    return filter

def getTimeStamp(seed):
    loc = seed.find('-')
    if (loc == -1):
        return int(seed)
    else:
        return int(seed[:loc])

def timeStampFilter(seed, maxSeed = None):
    print('applying time stamp filter after time {}'.format(seed))
    def filter(data, addr):
        # print('data = {}, addr = {}'.format(data, addr))
        if (maxSeed is None):
            if (seed is None):
                return True
            else:
                return (getTimeStamp(addr['seed']) > seed)
        elif (seed is None):
            return getTimeStamp(addr['seed'] < maxSeed)
        else:
            return (getTimeStamp(addr['seed']) > seed) and (getTimeStamp(addr['seed']) < maxSeed)
    return filter

def floatEqual(a, b, eps = 1e-7):
    return (np.abs(a - b) < eps)
def floatValueFilter(key, value):
    return lambda x, y: (key in x) and (floatEqual(x[key], value))

def describerEqual(desc1, desc2):
    # describer has N, L, V, T four arguments
    # N should not be considered
    # intPartEqual = (desc1['L'] == desc2['L'])
    # floatPartEqual = (floatEqual(desc1['T'], desc2['T']) and floatEqual(desc1['V'], desc2['V']))
    # m1 = 0.0
    # m2 = 0.0
    # if ('m' in desc1):
    #     m1 = desc1['m']
    # if ('m' in desc2):
    #     m2 = desc2['m']
    
    # # mEqual = (('m' not in desc1) and ('m' not in desc2)) or (floatEqual(desc1['m'], desc2['m']))
    # mEqual = floatEqual(m1, m2)
    # return (intPartEqual and floatPartEqual and mEqual)

    intParaSet = ['L', 'wf']
    floatParaSet = ['T', 'V', 'm', 'b', 'mu']
    intEqualFlag = True 
    for intPara in intParaSet:
        val1 = 0
        val2 = 0
        if (intPara in desc1):
            val1 = desc1[intPara]
        if (intPara in desc2):
            val2 = desc2[intPara]
        intEqualFlag = intEqualFlag and (val1 == val2)
        if (not intEqualFlag):
            return False
    
    floatEqualFlag = True 
    for floatPara in floatParaSet:
        val1 = 0.0 
        val2 = 0.0
        if (floatPara in desc1):
            val1 = desc1[floatPara]
        if (floatPara in desc2):
            val2 = desc2[floatPara]
        floatEqualFlag = floatEqualFlag and floatEqual(val1, val2)
        if (not floatEqualFlag):
            return False 
    
    return True

def describerEqualKeys(desc1, desc2, keys):
    res = True
    if ('T' in keys):
        res = (res and (floatEqual(desc1['T'], desc2['T'])))
    if ('V' in keys):
        res = (res and (floatEqual(desc1['V'], desc2['V'])))
    if ('L' in keys):
        res = (res and (desc1['L'] == desc2['L']))
    return res

def describerToStr(describer, legendShown = None):
    # print('describer = {}'.format(describer))
    # print(legendShown)
    floatKeySet = ['T', 'V', 'm']
    keySet = list(describer.keys())
    if (legendShown is not None):
        keySet = legendShown

    res = ""
    for key in keySet:
        if (key in floatKeySet):
            if (key == 'm') and (np.abs(describer[key]) < 1e-5):
                continue
            # print('key = {}'.format(key))
            keyPart = '{} = '.format(key)
            valuePart = '{:.2f}, '.format(describer[key]).rstrip('0')
            res += keyPart 
            res += valuePart
        else:
            res += '{} = {}, '.format(key, describer[key])
    # print('res = {}'.format(res))
    return res[:-2] # remove the last comma

def polishDescriber(describer):
    floatKeySet = ['T', 'V']

    res = dict([])
    for key in describer:
        if (key in floatKeySet):
            res[key] = '{:.3f}'.format(describer[key]).rstrip('0')
        else:
            res[key] = describer[key]
    return res

def sortListBy(keyList, dataList):
    resList = [(key, data) for key, data in zip(keyList, dataList)]
    resList = sorted(resList, key = lambda x: x[0])
    
    resKeyList = [x[0] for x in resList]
    resDataList = [x[1] for x in resList]
    return resKeyList, resDataList

def decodeLogLine(log):
    log = log.strip()
    
    sharpIdx = log.find('#')
    if (sharpIdx == 0):
        return False, '', ''
    
    equalIdx = log.find('=')
    if (equalIdx == -1):
        return False, '', ''
    left = log[:equalIdx].strip()
    right = log[(equalIdx + 1):].strip()
    return True, left, right

def parseInt(s):
    s = s.strip()
    intS = ''
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in range(len(s)):
        if (s[i] in digits):
            intS += s[i]
        else:
            break
    return int(intS)

def makeNormalFunc(average, error):
    
    def normal():
        return average + error * np.random.randn()
    return normal

def identityFunc(x):
    return x

def decodeFolder(setSeed):
    pLoc = setSeed.find('-')
    if (pLoc == -1):
        setNo = -1
        seed = int(setSeed)
    else:
        setNo = int(setSeed[:pLoc])
        seed = int(setSeed[(pLoc + 1):])
    return setNo, seed
def encodeFolder(setNo, seedNo):
    if (setNo == -1):
        return '{}'.format(seedNo)
    else:
        return '{}-{}'.format(setNo, seedNo)

def getIntHistogram(data):
    print(data)
    data = [np.int(d) for d in data]
    maxV = np.max(data)
    res = np.zeros(maxV + 1)
    for d in data:
        res[d] += 1
    return res

def histogram2D(x):
    d0 = np.array([np.int(a) for a in x[:, 0]])
    d1 = np.array([np.int(a) for a in x[:, 1]])
    
    max0 = np.max(d0)
    max1 = np.max(d1)
    sets = []
    for i in range(max0 + 1):
        sets.append([])
        for _ in range(max1 + 1):
            sets[i].append(0)
    for i in range(len(x)):
        sets[d0[i]][d1[i]] += 1
    return np.array(sets)

def discreteHistogram(x):
    # input: x[:, 0], x[:, 1]
    d0 = np.array([np.int(a) for a in x[:, 0]])
    d1 = np.array([np.int(a) for a in x[:, 1]])
    max0 = np.max(d0)
    max1 = np.max(d1)
    sets = []
    for i in range(max0 + 1):
        sets.append([])
        for _ in range(max1 + 1):
            sets[i].append(0)
    for i in range(len(x)):
        sets[d0[i]][d1[i]] += 1
    
    xAxis = []
    yAxis = []
    for i in range(max0 + 1):
        for j in range(max1 + 1):
            if (sets[i][j] == 0):
                continue
            distance = np.sqrt(i * i + j * j)
            flag = False
            for k in range(len(xAxis)):
                if (floatEqual(distance, xAxis[k])):
                    yAxis[k] += sets[i][j]
                    flag = True 
                    break
            if (not flag):
                xAxis.append(distance)
                yAxis.append(sets[i][j])

    xAxis, yAxis = sortListBy(xAxis, yAxis)
    return np.array(xAxis), np.array(yAxis)

def combineHistogram(h1, h2):
    x1 = h1[0]
    y1 = h1[1]

    x2 = h2[0]
    y2 = h2[1]

    resX = list(np.copy(x1))
    resY = list(np.copy(y1))
    # print('resX = {}, resY = {}'.format(resX, resY))
    # print('x2 = {}, y2 = {}'.format(x2, y2))
    
    for i in range(len(x2)):
        flag = False
        for j in range(len(resX)):
            if (floatEqual(x2[i], resX[j])):
                resY[j] += y2[i]
                flag = True
        if (not flag):
            resX.append(x2[i])
            resY.append(y2[i])
    
    resX, resY = sortListBy(resX, resY)
    return np.array([resX, resY])

def combine2DHistogram(a1, a2):

    if (a1.shape == a2.shape):
        return a1 + a2
    
    h1, w1 = a1.shape 
    h2, w2 = a2.shape 
    
    maxH = np.max([h1, h2])
    maxW = np.max([w1, w2])

    res = np.zeros((maxH, maxW))
    res[:h1, :w1] += a1 
    res[:h2, :w2] += a2 
    return res

def log2DHistogram(a, lims = None):
    if (lims is None):
        aa = np.zeros(a.shape)
    else:
        aShape = a.shape 
        limX = max([lims, aShape[0]])
        limY = max([lims, aShape[1]])
        aa = np.zeros((limX, limY))

    for i in range(len(a)):
        for j in range(len(a[0])):
            if (a[i][j] > 0):
                aa[i][j] = np.log(a[i][j])
    return aa

def combine2DHistogramList(hList):
    # for h in hList:
    #     print(h.shape)
    res = hList[0]
    for i in range(1, len(hList)):
        res = combine2DHistogram(res, hList[i])
    return res
    
def combineHistogramList(hList):
    res = hList[0]
    for i in range(1, len(hList)):
        res = combineHistogram(res, hList[i])
    
    histList = []
    for i in range(len(res[0])):
        histList.append([])
    for hL in hList:
        for j in range(len(res[0])):
            flag = False
            for i in range(len(hL[0])):
                if (floatEqual(hL[0][i], res[0][j])):
                    histList[j].append(hL[1][i])
                    flag = True
            if (not flag):
                histList[j].append(0)

    for i in range(len(histList)):
        histList[i] = np.array(histList[i])
    # print(histList)
    # print(hList)
    return res, histList

def setErrorBar(x):
    return np.std(x) / np.sqrt(len(x) - 1)

def getkbr(x, y):
    axx = np.average(x ** 2)
    ayy = np.average(y ** 2)
    axy = np.average(x * y)
    ax = np.average(x)
    ay = np.average(y)

    k = (axy - ax * ay) / (axx - ax ** 2)
    b = ay - ax * k
    r = (axy - ax * ay) / np.sqrt((axx - ax ** 2) * (ayy - ay * ay))
    return k, b, r

def analyzeHistogram(x, y, label):
    n = len(x) // 2
    xFrag = np.array(x[n:])
    yFrag = np.log(np.array(y[n:]))

    # xFrag = np.log(np.array(x[1:n]))
    # # yFrag = np.log(np.array(y[1:n]))
    # yFrag = np.array(y[1:n])

    k, b, r = getkbr(xFrag, yFrag)
    print('{}: k = {}, b = {}, r = {}'.format(label, k, b, r))
    return k

def maximumGroup(groups, excep = None):
    maxLength = 0
    res = []
    maxKey = None
    for key in groups:
        if (key == excep):
            continue
        if (len(groups[key]) > maxLength):
            maxLength = len(groups[key])
            res = groups[key]
            maxKey = key

    return maxKey, res

def dealCheckerBoard(x):
    resX = []
    resY = []
    for i in range(len(x)):
        if (i % 2 == 1):
            resX.append(i)
            resY.append(x[i])
    
    return {'x': resX, 'y': resY}

def getFitting(data):
    x = np.array(data['x'])
    y = np.array(data['y'])

    if (len(x) < 8):
        print('data not enough for fitting, stop.')
        return None
    print('data length = {}'.format(len(x)))
    # startIdx = int(len(x) * 0.25)
    # endIdx = int(len(x) * 0.75)
    startIdx = 2
    endIdx = 8

    xFit = np.log(x[startIdx : endIdx])
    yFit = y[startIdx : endIdx]

    k, b, r = getkbr(xFit, yFit)
    return {'k': k, 'b': b, 'r': r}
    # return
            
def isTimeObsName(dataName):
    return (dataName.find('time') != -1)

def getList(x):
    if (isinstance(x, int) or isinstance(x, float)):
        return [x]
    else:
        return x

def getSingle(x):
    if (isinstance(x, int) or isinstance(x, float)):
        return x
    else:
        return x[0]

def makeFitting(x, y):
    # print('x = {}, y = {}'.format(x, y))
    x = np.array(x) 
    y = np.array(y)

    axx = np.average(x * x) 
    axy = np.average(x * y)
    ayy = np.average(y * y)

    ax = np.average(x)
    ay = np.average(y)

    k = (axy - ax * ay) / (axx - ax * ax)
    b = ay - k * ax 
    r = (axy - ax * ay) / np.sqrt((axx - ax * ax) * (ayy - ay * ay))
    return k, b, r

def keyCombine(key1, val1, key2, val2, funcs):
    key1, val1 = sortListBy(key1, val1)
    key2, val2 = sortListBy(key2, val2)

    cur1 = 0
    cur2 = 0

    resKey = []
    resVal = []
    while (cur1 < len(key1)) and (cur2 < len(key2)):
        if (floatEqual(key1[cur1], key2[cur2])):
            resKey.append(key1[cur1])
            resVal.append(funcs(val1[cur1], val2[cur2]))
            cur1 += 1
            cur2 += 1
            continue 

        if (key1[cur1] < key2[cur2]):
            cur1 += 1
        else:
            cur2 += 1
    
    return resKey, resVal

def getTotalSize(sizeTuple):
    res = 1
    for x in sizeTuple:
        res *= x 
    
    return res

def decodeSnapshot(snapshot, totalBondN, snapshotBit):
    res = []
    bondCount = 0
    while (bondCount < totalBondN):
        for x in snapshot:
            for bit in range(snapshotBit): 
                resBit = (x >> bit) & 1
                if (resBit == 1):
                    res.append(True)
                else:
                    res.append(False)
                bondCount += 1
                if (bondCount >= totalBondN):
                    return res
    
    return res

def decode2DDimer(idx, h, w):
    direct = idx // (h * w)
    x = (idx - direct * h * w) // w
    y = idx % w 
    return (direct, x, y)

def decode2DCorr(x, L):
    # xx = (x - (-0.0625)) / 0.25
    xx = x
    return np.reshape(xx[:(L * L)], (L, L)), np.reshape(xx[(L * L):], (L, L))

def zipLists(*args):
    assert (len(args) > 0), "Error: length of args must be positive for zipLists(*args)."
    if (len(args) == 1):
        return [(x, ) for x in args[0]]
    else:
        partRes = zipLists(*args[1:])
        res = []
        for x in args[0]:
            for y in partRes:
                res.append((x, *y))

    return res

def generateList(x, type):
    if (isinstance(x, type)):
        return [x]
    else:
        return x

def adjustAxixRange(ax, xRange = 1.0, yRange = 1.0):
    pos = ax.get_position()

    x0, x1, y0, y1 = pos.x0, pos.x1, pos.y0, pos.y1 
    newX1 = x0 + (x1 - x0) * xRange 
    newY1 = y0 + (y1 - y0) * yRange 

    ax.set_position(mplTransforms.Bbox([[x0, y0], [newX1, newY1]]))

def setNewAxisAtRight(fig, ax, xRange = 0.90, yScale = 0.5):
    pos = ax.get_position()

    x0, x1, y0, y1 = pos.x0, pos.x1, pos.y0, pos.y1 
    # print(x0, x1, y0, y1)

    newX1 = x0 + (x1 - x0) * xRange 
    # return fig.add_axes([newX1, y0, x1 - newX1, y1 - y0])
    # return fig.add_axes([newX1, x1 - newX1, y0, y1 - y0])
    newAx = fig.add_axes([newX1, y0, x1 - newX1, y1 - y0])
    # newAx.set_position(mplTransforms.Bbox([[x0, y0], [newX1, y1]]))
    return newAx

def addColorBar(ax, fig, im, adjustRange = 0.85, colorBarRange = 0.90):
    cax = setNewAxisAtRight(fig, ax, xRange = colorBarRange)
    adjustAxixRange(ax, xRange = adjustRange)
    fig.colorbar(im, cax = cax)

def normalizeArray(x, errorBar = None):
    xArray = np.array(x) 
    lowV = np.min(xArray)
    highV = np.max(xArray)
    resX = ((xArray - lowV) / (highV - lowV)) * 2.0 - 1.0
    if (errorBar is not None):
        errorBar = np.array(errorBar) * 2.0 / (highV - lowV)
        return resX, errorBar
    else:
        return resX

def floorInt(x, eps = 1e-8):
    return math.floor(x + eps)

def flipAppend(x):
    return np.array(list(x) + list(-x))

def resortMVariables(x):
    return np.array([x[1], -x[0], x[3], -x[2]])

def getM0011(x):
    return x[0] - x[1] - x[2] + x[3], x[0] + x[1] - x[2] - x[3]

def makeHist(lim, bins, x):

    data, _, _ = np.histogram2d(x = x[:, 0], y = x[:, 1], bins = bins, range = lim)
    return {'lim0': np.array([lim[0][0], lim[1][0]]), 'lim1': np.array([lim[0][1], lim[1][1]]), 'steps': np.array(bins), 'data': data.flatten()}
    # return {'lim0': self.data[dataName + ' bins begin'], 'lim1': self.data[dataName + ' bins end'], 'steps': self.data[dataName + ' bins'], 'data': self.data[dataName + ' bins data']}

def weightedBinder(x, weights, dim = 2):
    xArray = np.array(x) 
    weightArray = np.array(weights)
    x2 = np.sum((xArray ** 2) * weightArray)
    x4 = np.sum((xArray ** 4) * weightArray)
    if (dim == 0):
        return x4 / (x2 ** 2)
    else:
        return 1.0 - (x4 / (x2 * x2 * dim))

def binderPreparation(psiX, psiY):
    # first consider theta series
    # theta = np.arctan2(psiX, psiY)
    # cos2Theta = np.cos(2 * theta)

    # cos(2theta) = cos^2(theta) - sin^2(theta) = (psiX^2 - psiY^2) / |psi|^2
    cos2Theta = ((psiX ** 2) - (psiY ** 2)) / (psiX ** 2 + psiY ** 2)
    atanhCos2Theta = np.arctanh(cos2Theta)
    atanAtanhCos2Theta = (2.0 / np.pi) * np.arctan((2.0 / np.pi) * atanhCos2Theta)
    # print(psiX, psiY, atanhCos2Theta, atanAtanhCos2Theta)
    return atanhCos2Theta, atanAtanhCos2Theta

def floatAllTheSame(l):
    if len(l) == 0:
        return True
    v = l[0]
    for vv in l:
        if not floatEqual(v, vv):
            return False
    return True

def printBits(x, bitN = 32):
    res = []
    for i in range(bitN):
        res.append(str(x % 2))
        x = x // 2
    print(''.join(res))

def zigzagOrder(h, w):
    res = []
    for i in range(h):
        if (i % 2 == 0):
            res += list(range(i * w, (i + 1) * w))
        else:
            res += list(range((i + 1) * w - 1, i * w - 1, -1))
    return res

def getOneHot(targets, classes):
    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    res = np.eye(classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[classes])

def singleFeatureMapToVector(x):
    return np.array([np.cos(x * 0.5 * np.pi), np.sin(x * 0.5 * np.pi)])

def transpose(x):
    if len(x.shape) == 1:
        return x
    else:
        return x.T
