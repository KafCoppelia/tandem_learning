from collections import OrderedDict
import json
from copy import deepcopy
from enum import Enum
import numpy as np
import math
from .hwConfig import Hardware
from .transformer import S2LocalPlace, S2LogicCore
from .transformer import genLocalPlace, genSoftwareNetwork
from .utils import multiCast

def getCore(direction, baseCoreId, levelId, hardwareCoreYBit, isOffline):
    NoCY = Hardware.getAttr("NoCLevelsY", isOffline)[levelId]
    directionX = direction // NoCY
    directionY = direction % NoCY
    coreXId = directionX << (hardwareCoreYBit + levelId)
    coreYId = directionY << levelId
    coreId = baseCoreId | coreXId | coreYId
    return coreId

class BasicOpInfo:
    def __init__(self):
        self.bitWidth = 4
        self.mode = "ann" #or "snn"
        self.kind = "None"
        self.name = "None"
        self.threshold = 1
        self.memPotential = 0
        self.scale = 1
        self.pool = False
        self.resetMode = "None"
        self.inputSize = None
        self.outputSize = None
        self.weightBase = 1
        self.bitTrunc = 0

        self.isOffline = True

        return
    
    def load(self, info):
        self.bitWidth = info['bitWidth']
        self.mode = info['mode']
        self.kind = info['kind']
        self.name = info['name']
        self.threshold = info['threshold']
        self.memPotential = info['memPotential']
        self.scale = info['scale']
        self.resetMode = info['resetMode']
        # self.inputNames = info['inputNames']
        # self.outputNames = info['outputNames']
        self.inputSize = info['inputSize']
        self.outputSize= info['outputSize']
        self.weightBase = info['weightBase']
        return
    
    def store(self):
        info = self.__dict__
        return info
    
    def inputLen(self):
        return 0
    
    def size(self):
        return np.prod(self.outputSize)
    
    def minLCN(self):
        return 1
    
    def transform(
        self, computeGroup, softwareNetwork, softNeuronId, 
        fakeSoftIdBounds, bitWidth, copyNum, groupId, LCN
    ):
        return groupId

# these operators can be set as 'ann', 'snn', 'online'
class Conv2dInfo(BasicOpInfo):
    
    def __init__(self):
        super().__init__()
        self.kind = "conv2d"
        self.kernelSize = [1,1,1,1]
        self.inputSize =  [1,1,1]
        self.outputSize = [1,1,1]
        self.padding = [0,0]
        self.stride = [1,1]
        self.dilation = [1,1]
        self.groups = 1
        self.bias = None
        self.weight = None

        return
    
    def load(self, info):
        super().load(info)
        self.kernelSize = info['kernelSize']
        self.padding = info['padding']
        self.stride = info['stride']
        self.dilation = info['dilation']
        self.groups = info['groups']
    
    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__
    
    def inputLen(self):
        inputLen = \
            (np.prod(self.kernelSize[1:]) / self.groups)
        return inputLen
    
    def selectCoreLevel(self, bitWidth, maxCopyNum, LCN, hardwareType):
        inputSize  = self.inputSize[0]
        outputSize = self.outputSize
        
        # ic: input channel
        # oc: output channel
        icStride =  inputSize[0] // self.groups
        ocStride =  outputSize[0] // self.groups
        hStride = self.stride[0]
        wStride = self.stride[1]
    
        # bow: block output width
        # boh: block output height
        resBow = 0
        resBoh = 0
        resBoc = 0
        resCopy = 1
        minCores = 1<<20

        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        # assert self.isOffline
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", self.isOffline)
        maxNeuronSize = hardwareNeuronNum * inputWidth
        
        if hardwareType == 'v2':
            if inputWidth == 8:
                inputCombination = LCN * bitWidth
                if inputCombination == 1:
                    maxNeuronSize = 1888
                elif inputCombination == 2:
                    maxNeuronSize = 2728
                elif inputCombination == 4:
                    maxNeuronSize = 3504
                else:
                    maxNeuronSize = 4096
        
        maxAxonSize = hardwareAxonNum // inputWidth

        # while LCN <= Hardware.MAXLCN:
        realNeuronNum = maxNeuronSize // (LCN * bitWidth)
        realAxonNum = int(maxAxonSize * LCN)
        
        for boc in range(1, outputSize[0] + 1, 1):
            if boc % ocStride != 0 or ocStride % boc != 0:
                continue
            for boh in range(1, outputSize[1] + 1):
                for bow in range(1, outputSize[2] + 1):
                    for copy in range(1, maxCopyNum + 1):
                        if copy * boc * boh * bow > realNeuronNum:
                            break
                        c = math.ceil(boc / ocStride) * icStride
                        h = (boh - 1) * hStride + 1 + (self.kernelSize[2] - 1) * self.dilation[0]
                        w = (bow - 1) * wStride + 1 + (self.kernelSize[3] - 1) * self.dilation[1]
                        if c * h * w > realAxonNum:
                            break
                        blkNum = math.ceil(outputSize[1] / boh) * \
                                    math.ceil(outputSize[2] / bow) * \
                                    math.ceil(outputSize[0] / boc) * \
                                    math.ceil(maxCopyNum / copy)
                        if minCores > blkNum:
                            minCores = blkNum
                            resBoc = boc
                            resBoh = boh
                            resBow = bow
                            resCopy = copy
                            # resLCN = LCN
            # LCN *= 2

        print(f"MIN_CORE: select block size = {(resBoc,resBoh,resBow)}, block_num = {minCores}\n")
        return  (resBoc, resBoh, resBow, resCopy)
    
    def selectCommLevel(
        self, outBaseSize, loops, commLoops
    ):
        selections = [
                (1,1,1,4), (1,1,2,2), (1,1,4,1), (1,2,1,2),(1,2,2,1), 
                (1,4,1,1), (2,1,1,2), (2,1,2,1),(2,2,1,1), (4,1,1,1)
            ]
        if loops[0] == 1 and loops[1] == 1 and loops[2] == 1 and loops[3] == 1:
            return
        
        levelId = len(commLoops)
        maxLevelId = Hardware.getAttr("NOCLEVEL", self.isOffline)
        if levelId >= maxLevelId:
            commLoops.append([loops[0], loops[1], loops[2], loops[3]])
            return
        NoCX = Hardware.getAttr("NoCLevelsX", self.isOffline)[levelId]
        NoCY = Hardware.getAttr("NoCLevelsY", self.isOffline)[levelId]
        NoCNum = NoCX * NoCY
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        ocStride = outputSize[0] // self.groups
        icStride = inputSize[0] // self.groups
        inBaseC = math.ceil(outBaseSize[0] / ocStride) * icStride
        inBaseH = 1 + (outBaseSize[1] - 1) * self.stride[0] + (self.kernelSize[2] - 1) * self.dilation[0]
        inBaseW = 1 + (outBaseSize[2] - 1) * self.stride[1] + (self.kernelSize[3] - 1) * self.dilation[1]
        
        overlapH = self.kernelSize[2] * self.dilation[0] - self.stride[0]
        overlapW = self.kernelSize[3] * self.dilation[1] - self.stride[1]
        if outBaseSize[0] % ocStride == 0:
            overlap1 = 0
        else:
            overlap1 = icStride * inBaseH * inBaseW 
        overlap2 = inBaseC * overlapH * inBaseW
        overlap3 = inBaseC * overlapW * inBaseH
        overlap4 = inBaseC * inBaseH * inBaseW 
        # center = inBaseC * overlapH * overlapW

        commLoop = [1,1,1,1]
        maxOutNeuron = 0
        overlap = 0
        for selection in selections:
            tmpOverlap = 0
            if loops[0] == 1 and selection[0]>1:
                continue
            if loops[1] == 1 and selection[1]>1:
                continue
            if loops[2] == 1 and selection[2]>1:
                continue
            if loops[3] == 1 and selection[3]>1:
                continue

            realC = min(loops[0], selection[0])
            realH = min(loops[1], selection[1])
            realW = min(loops[2], selection[2])
            realB = min(loops[3], selection[3])
            num = realC * realH * realW * realB
            if num > NoCNum:
                continue
            tmpOverlap += overlap2 * (realH - 1) * realW 
            tmpOverlap += overlap3 * (realW - 1) * realH
            # tmpOverlap -= center * (realH - 1) * (realW - 1)
            # tmpOverlap += (overlap1 - tmp_overlap) * (real_s_c - 1)
            tmpOverlap += overlap1 * (realC - 1) * realH * realW
            tmpOverlap += overlap4 * (realB - 1) * realC * realH * realW
            if tmpOverlap > overlap or (tmpOverlap == overlap and maxOutNeuron < num):
                overlap = tmpOverlap
                maxOutNeuron = num
                commLoop = np.array([realC, realH, realW, realB])
        newLoops = [
            math.ceil(loops[0] / commLoop[0]),
            math.ceil(loops[1] / commLoop[1]),
            math.ceil(loops[2] / commLoop[2]),
            math.ceil(loops[3] / commLoop[3])
        ]
        newBaseSize = [
            outBaseSize[0] * commLoop[0],
            outBaseSize[1] * commLoop[1],
            outBaseSize[2] * commLoop[2],
            outBaseSize[3] * commLoop[3],
        ]
        commLoops.append(commLoop)
        self.selectCommLevel(newBaseSize, newLoops, commLoops)
        return 

    def output2input(self, oc, oh, ow, ocStride, icStride):
        inputSize = self.inputSize[0]
        icBeg = (oc // ocStride) * icStride
        icEnd = icBeg + icStride
        
        ihBeg = oh * self.stride[0] - self.padding[0]
        ihEnd = ihBeg + (self.kernelSize[2] - 1)* self.dilation[0] + 1
        ihEnd = min(ihEnd, inputSize[1])
        ihStride = self.dilation[0]

        iwBeg = ow * self.stride[1] - self.padding[1]
        iwEnd = iwBeg + (self.kernelSize[3] - 1) * self.dilation[1] + 1
        iwEnd = min(iwEnd, inputSize[2])
        iwStride = self.dilation[1]

        i = 0 
        j = 0 
        k = 0
        weightDict = dict()
        for ic in range(icBeg, icEnd, 1):
            for ih in  range(ihBeg, ihEnd, ihStride):
                if ih < 0:
                    j += 1
                    continue
                for iw in range(iwBeg, iwEnd, iwStride):
                    if iw < 0:
                        k += 1
                        continue
                    inputPos = ic * (inputSize[1] * inputSize[2]) + ih * inputSize[2] + iw
                    weightDict[inputPos] = self.weight[oc, i, j, k,:]
                    k += 1
                k = 0
                j += 1
            k = 0
            j = 0
            i += 1
        return weightDict

    def buildSLogicCore(self, s2LocalPlace, coreId, baseLocation, outBlockSize):
        outputSize = self.outputSize
        inputSize = self.inputSize[0]
        ocSize = outBlockSize[0]
        ohSize = outBlockSize[1]
        owSize = outBlockSize[2]
        obSize = outBlockSize[3]
        baseOc = baseLocation[0]
        baseOh = baseLocation[1]
        baseOw = baseLocation[2]
        endOc = min(outputSize[0], ocSize + baseOc)
        endOh = min(outputSize[1], ohSize + baseOh)
        endOw = min(outputSize[2], owSize + baseOw)
        ocStride = outputSize[0] // self.groups
        icStride = inputSize[0] // self.groups
        weightDict = dict()
        neurons = list()
        axons = set()
        for oc in range(baseOc, endOc):
            for oh in range(baseOh, endOh):
                for ow in range(baseOw, endOw):
                    pos = oc * (outputSize[1] * outputSize[2]) + oh * outputSize[2] + ow
                    neurons += [pos] * obSize
                    weightDict[pos] = self.output2input(oc,oh,ow,ocStride,icStride)
                    axons |= set(weightDict[pos].keys())
        if len(axons) > 0:
            s2LocalPlace.addCore(coreId, axons, neurons)
        return weightDict
    
    def buildSLogicPlace(
        self, s2LocalPlaces, commLoops, levelSizes, baseLoop, baseCoreId,
        levelId, bitWidth, LCN
    ):
        oc = commLoops[levelId][0]
        oh = commLoops[levelId][1]
        ow = commLoops[levelId][2]
        b  = commLoops[levelId][3]
        direction = 0
        weightDict = dict()
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)
        for i in range(oc):
            for j in range(oh):
                for k in range(ow):
                    for t in range(b):
                        if levelId == hardwareNoCLevel:
                            s2LocalPlaces.append(
                                S2LocalPlace(inputWidth, outputWidth, bitWidth, LCN, self.isOffline)
                            )
                        if levelId < hardwareNoCLevel:
                            # coreXId = (direction >> 1) << (hardwareCoreYBit + levelId)
                            # coreYId = (direction & 1) << levelId
                            # coreId = baseCoreId | coreXId | coreYId
                            coreId = getCore(direction, baseCoreId, levelId, hardwareCoreYBit, self.isOffline)
                        else:
                            coreId = 0
                        
                        baseLocation = levelSizes[levelId] * np.array([i,j,k,t]) + baseLoop
                        if levelId == 0:
                            tmpWeightDict = self.buildSLogicCore(
                                s2LocalPlaces[-1],  coreId, baseLocation, levelSizes[0]
                            )
                            weightDict.update(tmpWeightDict)
                        else:
                            tmpWeightDict = self.buildSLogicPlace(
                                s2LocalPlaces, commLoops, levelSizes, baseLocation, coreId, 
                                levelId - 1, bitWidth, LCN
                            )
                            weightDict.update(tmpWeightDict)
                        direction += 1
        return weightDict

    def minLCN(self):
        inputSize  = self.inputSize[0]
        inputNum = np.prod(self.kernelSize[1:]) // self.groups
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth
        LCN =  math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN =  tmpLCN
        return LCN

    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        
        outBlockSize = self.selectCoreLevel(bitWidth, copyNum, LCN, hardwareType)

        loops = [
            math.ceil(outputSize[0] / outBlockSize[0]), 
            math.ceil(outputSize[1] / outBlockSize[1]), 
            math.ceil(outputSize[2] / outBlockSize[2]),
            math.ceil(copyNum / outBlockSize[3])
        ]
        
        commLoops = list()
        self.selectCommLevel(outBlockSize, loops, commLoops)
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        if len(commLoops) <= hardwareNoCLevel:
            commLoops += [
                np.array([1,1,1,1]) \
                    for i in range(hardwareNoCLevel + 1 - len(commLoops))
            ]
        levelSizes = [np.array(outBlockSize)]
        for i, commLoop in enumerate(commLoops):
            levelSizes.append(levelSizes[-1] * commLoop)
        s2LocalPlaces = list()
        weightDict = self.buildSLogicPlace(
            s2LocalPlaces, commLoops, levelSizes, np.array([0,0,0,0]), 
            0, len(commLoops) - 1, bitWidth, LCN)
        
        for s2LocalPlace in s2LocalPlaces:
            s2LocalPlace.multicast()
        
        inputNum = np.prod(inputSize)
        biasDict = dict()
        outputNum = np.prod(outputSize)
        oneChannel = outputSize[1] * outputSize[2]
        for i in range(outputNum):
            biasDict[i] = self.bias[i // oneChannel]

        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        
        genSoftwareNetwork(   
            softwareNetwork, s2LocalPlaces, self.name, bitWidth, groupId, 
            LCN, inputNum, outputNum, inputWidth, outputWidth, self.isOffline
        )
        inId = groupId

        lowerMem = -(1 << 6) if self.bitWidth == 1 else -(1 << 31)
        groupId, weightMapping = genLocalPlace(
            s2LocalPlaces, weightDict, biasDict, self.resetMode, self.threshold, 
            self.bitTrunc, groupId, computeGroup, self.mode,self.pool, self.isOffline,
            # below are online core parameters
            np.zeros([60]),  # LUT
            0,               # resetMem
            0,               # lateral_inhi_val
            -(1 << 7),       # lowerWeight
            (1 << 7) - 1,    # upperWeight
            0,               # weightDecay
            lowerMem,        # lowerMem
            0                # learnMode
        )

        return groupId, weightMapping

class FcInfo(BasicOpInfo):
    
    def __init__(self):
        super().__init__()
        self.kind = "fc"
        self.inputSize = [1]
        self.outputSize = [1]
        self.bias = None
        self.weight = None

        return
    
    def load(self, info):
        super().load(info)
        self.bias = info['bias']
        self.weight = info['weight']
    
    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__
    
    def inputLen(self):
        return self.kernelSize[1]
    
    def genWeightDict(self):
        inputNum = self.inputSize[0][0]
        outputNum = self.outputSize[0]
        weightDict = dict()
        for i in range(outputNum):
            weightDict[i] = dict()
            for j in range(inputNum):
                weightDict[i][j] = self.weight[i,j,:]
        return weightDict
    
    def buildSLogicCore(self, s2LocalPlace, coreId, baseLocation, outBlockSize, copyNum):
        outputNum = self.outputSize[0]
        inputNum = self.inputSize[0][0]
        outputNumAll = outputNum * copyNum
        if baseLocation >= outputNumAll:
            return 
        axons = set(range(inputNum))
        endNum = min(baseLocation + outBlockSize, outputNumAll)
        neurons = (np.arange(baseLocation, endNum) // copyNum).tolist()
        s2LocalPlace.addCore(coreId, axons, neurons)
    
    def minLCN(self):
        # compute LCN
        inputSize = self.inputSize[0]
        inputNum = inputSize[0]
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth 
        LCN = math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN = tmpLCN
        return LCN

    def buildSLogicPlace(
        self, s2LocalPlaces, commLoops, levelSizes, baseLoop,
        baseCoreId, levelId, bitWidth, LCN, copyNum
    ):
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            # inputWidth = 8
            outputWidth = 1
        commLoop = commLoops[levelId]
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)
        for i in range(commLoop):
            if levelId == hardwareNoCLevel:
                s2LocalPlaces.append(
                    S2LocalPlace(inputWidth, outputWidth, bitWidth, LCN, self.isOffline)
                )
            if levelId < hardwareNoCLevel:
                # coreXId = (i >> 1) << (hardwareCoreYBit + levelId)
                # coreYId = ((i & 1) << (levelId))
                # coreId = baseCoreId | coreXId | coreYId
                coreId = getCore(i, baseCoreId, levelId, hardwareCoreYBit, self.isOffline)
            else:
                coreId = 0
            baseLocation = levelSizes[levelId] * i + baseLoop
            if levelId == 0:
                self.buildSLogicCore(
                    s2LocalPlaces[-1], coreId, baseLocation, levelSizes[0], copyNum
                )
            else:
                self.buildSLogicPlace(
                    s2LocalPlaces, commLoops, levelSizes, baseLocation, coreId,
                    levelId - 1, bitWidth, LCN, copyNum
                )
    
    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            # inputWidth = 8
            outputWidth = 1
        
        #compute outBlockSize
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", self.isOffline)
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        maxNeuronSize = hardwareNeuronNum * inputWidth
        if hardwareType == 'v2':
            if inputWidth == 8:
                inputCombination = LCN * bitWidth
                if inputCombination == 1:
                    maxNeuronSize = 1888
                elif inputCombination == 2:
                    maxNeuronSize = 2728
                elif inputCombination == 4:
                    maxNeuronSize = 3504
                else:
                    maxNeuronSize = 4096
        vldNeuron = math.ceil(maxNeuronSize / (LCN * bitWidth))
        blockNum = math.ceil((outputSize[0] * copyNum) / vldNeuron)
        outBlockSize = math.ceil((outputSize[0] * copyNum) / blockNum)
        
        # loop scheduling
        outputNum = blockNum
        commLoops = list()

        while outputNum > 1:
            if len(commLoops) >= hardwareNoCLevel:
                commLoops.append(outputNum)
                outputNum = 1
                break
            levelId = len(commLoops)
            NocX = Hardware.getAttr("NoCLevelsX", self.isOffline)[levelId]
            NocY = Hardware.getAttr("NoCLevelsY", self.isOffline)[levelId]
            NocNum = NocX * NocY
            if outputNum > NocNum:
                outputNum = math.ceil(outputNum / NocNum)
                commLoops.append(NocNum)
            else:
                commLoops.append(outputNum)
                outputNum = 1
        if len(commLoops) <= hardwareNoCLevel:
            commLoops += [1 for i in range(hardwareNoCLevel + 1 - len(commLoops))]
        
        levelSizes = [outBlockSize]
        for commLoop in commLoops:
            levelSizes.append(commLoop * levelSizes[-1])


        # gen logic place & weightDict & multicast
        weightDict = self.genWeightDict()
        s2LocalPlaces = list()
        self.buildSLogicPlace(
            s2LocalPlaces, commLoops, levelSizes, 0, 
            0, len(commLoops) - 1, bitWidth, LCN, copyNum
        )
        for s2LocalPlace in s2LocalPlaces:
            s2LocalPlace.multicast()

        inputNum = inputSize[0]
        outputNum = outputSize[0]
        biasDict = dict()
        for i in range(outputNum):
            biasDict[i] = self.bias[i]
        
        genSoftwareNetwork(   
            softwareNetwork, s2LocalPlaces, self.name, bitWidth, groupId, 
            LCN, inputNum, outputNum, inputWidth, outputWidth, self.isOffline
        )
        lowerMem = -(1 << 6) if self.bitWidth == 1 else -(1 << 31)
        groupId, weightMapping = genLocalPlace(
            s2LocalPlaces, weightDict, biasDict, self.resetMode, self.threshold, 
            self.bitTrunc, groupId,  computeGroup, self.mode, self.pool, self.isOffline,
            # below are online core parameters
            np.zeros([60]),  # LUT
            0,               # resetMem
            0,               # lateral_inhi_val
            -(1 << 7),       # lowerWeight
            (1 << 7) - 1,    # upperWeight
            0,               # weightDecay
            lowerMem,        # lowerMem
            0                # learnMode
        )

        return groupId, weightMapping

class TransConv2dInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "transposedConv2d"
        self.kernelSize = [1,1,1,1]
        self.inputSize =  [1,1,1]
        self.outputSize = [1,1,1]
        self.padding = [0,0]
        self.outputPadding = [0,0]
        self.stride = [1,1]
        self.groups = 1
        self.bias = None
        self.weight = None
        
        return

    def load(self, info):
        super().load(info)
        self.kernelSize = info['kernelSize']
        self.padding = info['padding']
        self.outputPadding = info['outputPadding']
        self.stride = info['stride']
        self.dilation = info['dilation']
        self.groups = info['groups']
        self.bias = info['bias']
        self.weight = info['weight']

    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__

    def inputLen(self):
        inputLen = math.ceil(self.kernelSize[2] / self.stride[0])
        inputLen *= math.ceil(self.kernelSize[3] / self.stride[1])
        inputLen *= self.kernelSize[0]
        inputLen /= self.groups
        return inputLen
    
    def selectCoreLevel(self,  bitWidth, maxCopyNum, LCN, hardwareType):
        inputSize  = self.inputSize[0]
        outputSize = self.outputSize
        
        # ic: input channel
        # oc: output channel
        icStride =  inputSize[0] // self.groups
        ocStride =  outputSize[0] // self.groups

        # bow: block output width
        # boh: block output height
        resBow = 0
        resBoh = 0
        resBoc = 0
        # resLCN = LCN
        resCopy = 1
        minCores = 1<<20

        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", self.isOffline)
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxNeuronSize = hardwareNeuronNum * inputWidth
        if hardwareType == 'v2':
            if inputWidth == 8:
                inputCombination = LCN * bitWidth
                if inputCombination == 1:
                    maxNeuronSize = 1888
                elif inputCombination == 2:
                    maxNeuronSize = 2728
                elif inputCombination == 4:
                    maxNeuronSize = 3504
                else:
                    maxNeuronSize = 4096
        maxAxonSize = hardwareAxonNum // inputWidth


        # while LCN <= Hardware.MAXLCN:
        realNeuronNum = maxNeuronSize // (LCN * bitWidth)
        realAxonNum = int(maxAxonSize * LCN)
        for boc in range(1, outputSize[0] + 1, 1):
            if boc % ocStride != 0 or ocStride % boc != 0:
                continue
            for boh in range(1, outputSize[1] + 1):
                for bow in range(1, outputSize[2] + 1):
                    for copy in range(1, maxCopyNum + 1):
                        if copy * boc * boh * bow > realNeuronNum:
                            break
                        c = math.ceil(boc / ocStride) * icStride
                        h = math.ceil((boh + self.kernelSize[2] - 1) / self.stride[0])
                        w = math.ceil((bow + self.kernelSize[3] - 1) / self.stride[1])
                        if c * h * w > realAxonNum:
                            break
                        blkNum = math.ceil(outputSize[1] / boh) * \
                                    math.ceil(outputSize[2] / bow) * \
                                    math.ceil(outputSize[0] / boc) * \
                                    math.ceil(maxCopyNum / copy)
                        if minCores > blkNum:
                            minCores = blkNum
                            resBoc = boc
                            resBoh = boh
                            resBow = bow
                            resCopy = copy
                            # resLCN = LCN
            # LCN *= 2
        
        print(f"MIN_CORE: select block size = {(resBoc,resBoh,resBow)}, block_num = {minCores}\n")
        return (resBoc, resBoh, resBow, resCopy)

    def selectCommLevel(self, outBaseSize, loops, commLoops):
        selections = [
                (1,1,1,4), (1,1,2,2), (1,1,4,1), (1,2,1,2),(1,2,2,1), 
                (1,4,1,1), (2,1,1,2), (2,1,2,1),(2,2,1,1), (4,1,1,1)
            ]
        if loops[0] == 1 and loops[1] == 1 and loops[2] == 1 and loops[3] == 1:
            return
        levelId = len(commLoops)
        maxLevelId = Hardware.getAttr("NOCLEVEL", self.isOffline)
        if levelId >= maxLevelId:
            commLoops.append([loops[0], loops[1], loops[2], loops[3]])
            return
        NoCX = Hardware.getAttr("NoCLevelsX", self.isOffline)[levelId]
        NoCY = Hardware.getAttr("NoCLevelsY", self.isOffline)[levelId]
        NoCNum = NoCX * NoCY
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        ocStride = outputSize[0] // self.groups
        icStride = inputSize[0] // self.groups
        inBaseC = math.ceil(outBaseSize[0] / ocStride) * icStride
        inBaseH = math.ceil((outBaseSize[1] + self.kernelSize[2] - 1) / self.stride[0])
        inBaseW = math.ceil((outBaseSize[2] + self.kernelSize[3] - 1) / self.stride[1])
        
        # overlapH = 1
        # overlapW = 1
        overlapH = max(0, (self.kernelSize[2] - 1) - (self.stride[0] - 1))
        overlapW = max(0, (self.kernelSize[3] - 1) - (self.stride[1] - 1))
        if outBaseSize[0] % ocStride == 0:
            overlap1 = 0
        else:
            overlap1 = icStride * inBaseH * inBaseW 
        overlap2 = math.ceil(inBaseC * overlapH * inBaseW / self.stride[1])
        overlap3 = math.ceil(inBaseC * overlapW * inBaseH / self.stride[0])
        overlap4 = inBaseC * inBaseH * inBaseW 
        # center = inBaseC * overlapH * overlapW

        commLoop = [1,1,1,1]
        overlap = 0
        maxOutNeuron = 0
        for selection in selections:
            tmpOverlap = 0
            if loops[0] == 1 and selection[0]>1:
                continue
            if loops[1] == 1 and selection[1]>1:
                continue
            if loops[2] == 1 and selection[2]>1:
                continue
            if loops[3] == 1 and selection[3]>1:
                continue

            realC = min(loops[0], selection[0])
            realH = min(loops[1], selection[1])
            realW = min(loops[2], selection[2])
            realB = min(loops[3], selection[3])
            num = realC * realH * realW * realB
            if num > NoCNum:
                continue
            tmpOverlap += overlap2 * (realH - 1) * realW 
            tmpOverlap += overlap3 * (realW - 1) * realH
            tmpOverlap += overlap1 * (realC - 1) * realH * realW
            tmpOverlap += overlap4 * (realB - 1) * realC * realH * realW

            if tmpOverlap > overlap or (tmpOverlap == overlap and maxOutNeuron < num):
                overlap = tmpOverlap
                maxOutNeuron = num
                commLoop = np.array([realC, realH, realW, realB])
        newLoops = [
            math.ceil(loops[0] / commLoop[0]),
            math.ceil(loops[1] / commLoop[1]),
            math.ceil(loops[2] / commLoop[2]),
            math.ceil(loops[3] / commLoop[3])
        ]
        newBaseSize = [
            outBaseSize[0] * commLoop[0],
            outBaseSize[1] * commLoop[1],
            outBaseSize[2] * commLoop[2],
            outBaseSize[3] * commLoop[3],
        ]
        commLoops.append(commLoop)
        self.selectCommLevel(newBaseSize, newLoops, commLoops)
        return

    def output2input(self, oc, oh, ow, ocStride, icStride):
        inputSize = self.inputSize[0]
        icBeg = (oc // ocStride) * icStride
        icEnd = icBeg + icStride
        padding = [self.kernelSize[2] - self.padding[0] - 1, self.kernelSize[3] - self.padding[1] - 1]
        ihBeg = oh - padding[0]
        ihEnd = ihBeg + self.kernelSize[2]
        ihEnd = min(ihEnd, inputSize[1] * self.stride[0])
        ihStride = 1

        iwBeg = ow - padding[1]
        iwEnd = iwBeg + self.kernelSize[3]
        iwEnd = min(iwEnd, inputSize[2] * self.stride[1])
        iwStride = 1

        i = 0 
        j = 0 
        k = 0
        o = oc % ocStride
        weightDict = dict()
        for ic in range(icBeg, icEnd, 1):
            for ih in  range(ihBeg, ihEnd, ihStride):
                if ih < 0 or ih % self.stride[0] != 0:
                    j += 1
                    continue
                for iw in range(iwBeg, iwEnd, iwStride):
                    if iw < 0 or iw % self.stride[1] != 0:
                        k += 1
                        continue
                    inputPos = ic * (inputSize[1] * inputSize[2]) + ih * inputSize[2] + iw
                    weightDict[inputPos] = self.weight[
                        ic, o, self.kernelSize[2] - 1 -j, self.kernelSize[3] - 1 - k,:]
                    k += 1
                k = 0
                j += 1
            k = 0
            j = 0
            i += 1
        return weightDict

    def buildSLogicCore(self, s2LocalPlace, coreId, baseLocation, outBlockSize):
        outputSize = self.outputSize
        inputSize = self.inputSize[0]
        ocSize = outBlockSize[0]
        ohSize = outBlockSize[1]
        owSize = outBlockSize[2]
        obSize = outBlockSize[3]
        baseOc = baseLocation[0]
        baseOh = baseLocation[1]
        baseOw = baseLocation[2]
        ocStride = outputSize[0] // self.groups
        icStride = inputSize[0] // self.groups
        endOc = min(outputSize[0], baseOc + ocSize)
        endOh = min(outputSize[1], baseOh + ohSize)
        endOw = min(outputSize[2], baseOw + owSize)
        weightDict = dict()
        neurons = list()
        axons = set()
        for oc in range(baseOc, endOc):
            for oh in range(baseOh, endOh):
                for ow in range(baseOw, endOw):
                    pos = oc * (outputSize[1] * outputSize[2]) + oh * outputSize[2] + ow
                    neurons += [pos] * obSize
                    weightDict[pos] = self.output2input(oc,oh,ow,ocStride,icStride)
                    axons |= set(weightDict[pos].keys())

        s2LocalPlace.addCore(coreId, axons, neurons)
        return weightDict

    def buildSLogicPlace(
        self, s2LocalPlaces, commLoops, levelSizes, baseLoop, baseCoreId,
        levelId, bitWidth, LCN
    ):
        oc = commLoops[levelId][0]
        oh = commLoops[levelId][1]
        ow = commLoops[levelId][2]
        b  = commLoops[levelId][3]
        direction = 0
        weightDict = dict()
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)  
        for i in range(oc):
            for j in range(oh):
                for k in range(ow):
                    for t in range(b):
                        if levelId == hardwareNoCLevel:
                            s2LocalPlaces.append(
                                S2LocalPlace(inputWidth, outputWidth, bitWidth, LCN, self.isOffline)
                            )
                        if levelId < hardwareNoCLevel:
                            # coreXId = (direction >> 1) << (hardwareCoreXBit + (levelId & hardwareNoCMask))
                            # coreYId = ((direction & 1) << (levelId & hardwareNoCMask))
                            # coreId = baseCoreId | coreXId | coreYId
                            coreId = getCore(direction, baseCoreId, levelId, hardwareCoreYBit, self.isOffline)
                        else:
                            coreId = 0
                        baseLocation = levelSizes[levelId] * np.array([i,j,k,t]) + baseLoop
                        if levelId == 0:
                            tmpWeightDict = self.buildSLogicCore(
                                s2LocalPlaces[-1], coreId, baseLocation, levelSizes[0]
                            )
                            weightDict.update(tmpWeightDict)
                        else:
                            tmpWeightDict = self.buildSLogicPlace(
                                s2LocalPlaces, commLoops, levelSizes, baseLocation, coreId, 
                                levelId - 1, bitWidth, LCN
                            )
                            weightDict.update(tmpWeightDict)
                        direction += 1
        return weightDict

    def minLCN(self):
        inputNum = self.kernelSize[0] * self.kernelSize[2] * self.kernelSize[3] // self.groups
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth
        LCN =  math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN =  tmpLCN
        return LCN
    
    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        
        outBlockSize = self.selectCoreLevel(bitWidth, copyNum, LCN, hardwareType)
        loops = [
            math.ceil(outputSize[0] / outBlockSize[0]), 
            math.ceil(outputSize[1] / outBlockSize[1]), 
            math.ceil(outputSize[2] / outBlockSize[2]),
            math.ceil(copyNum / outBlockSize[3])
        ]
        
        commLoops = list()
        self.selectCommLevel(outBlockSize, loops, commLoops)
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        if len(commLoops) <= hardwareNoCLevel:
            commLoops += [np.array([1,1,1,1])] * \
                                (hardwareNoCLevel + 1 - len(commLoops))
        
        levelSizes = [np.array(outBlockSize)]
        for i, commLoop in enumerate(commLoops):
            levelSizes.append(levelSizes[-1] * commLoop)
        s2LocalPlaces = list()
        weightDict = self.buildSLogicPlace(
            s2LocalPlaces, commLoops, levelSizes, np.array([0,0,0,0]), 
            0, len(commLoops) - 1, bitWidth, LCN)
        
        for s2LocalPlace in s2LocalPlaces:
            s2LocalPlace.multicast()
        
        inputNum = np.prod(inputSize)
        outputNum = np.prod(outputSize)
        oneChannel = outputSize[1] * outputSize[2]
        biasDict = dict()
        for i in range(outputNum):
            biasDict[i] = self.bias[i // oneChannel]
        
        
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        
        genSoftwareNetwork(   
            softwareNetwork, s2LocalPlaces, self.name, bitWidth, groupId, 
            LCN, inputNum, outputNum, inputWidth, outputWidth, self.isOffline
        )
        lowerMem = -(1 << 6) if self.bitWidth == 1 else -(1 << 31)
        groupId, weightMapping = genLocalPlace(
            s2LocalPlaces, weightDict, biasDict, self.resetMode, self.threshold, 
            self.bitTrunc, groupId, computeGroup, self.mode, self.pool, self.isOffline,
            # below are online core parameters
            np.zeros([60]),  # LUT
            0,               # resetMem
            0,               # lateral_inhi_val
            -(1 << 7),       # lowerWeight
            (1 << 7) - 1,    # upperWeight
            0,               # weightDecay
            lowerMem,        # lowerMem
            0                # learnMode
        )
   
        return groupId, weightMapping

class AddInfo(BasicOpInfo):
    
    def __init__(self):
        super().__init__()
        self.kind = "add"
        self.inputNum = 2
        self.inputSize = [1,1,1]
        self.weight = None
        self.bias = None

    def load(self, info):
        super().load(info)
        self.inputNum = info['inputNum']
        self.weight = info['weight']
        self.bias = info['bias']

    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__

    def inputLen(self):
        return (self.inputNum)
    
    def genWeightDict(self):
        inputNum = np.prod(self.inputSize[0])
        inputLen = len(self.inputSize)
        weightDict = dict()
        xySize = np.prod(self.inputSize[1:])
        for i in range(inputNum):
            weightDict[i] = dict()
            channelId = i // xySize
            for j in range(inputLen):
                weightDict[i][i + j * inputNum] = self.weight[channelId,j,:]
        return weightDict

    def buildSLogicCore(self, s2LocalPlace, coreId, baseLocation, outBlockSize, copyNum):
        outputNum = np.prod(self.outputSize)
        outputNumAll = outputNum * copyNum
        inputLen = len(self.inputSize)
        if baseLocation >= outputNumAll:
            return 
        axons = set()
        endNum = min(baseLocation + outBlockSize, outputNumAll)
        neurons = (np.arange(baseLocation, endNum) // copyNum).tolist()
        neuronArray = np.array(list(set(neurons)))
        for i in range(inputLen):
            newSet = set(neuronArray + outputNum * i)
            axons |= newSet
        s2LocalPlace.addCore(coreId, axons, neurons)

    def buildSLogicPlace(
        self, s2LocalPlaces, commLoops, levelSizes, baseLoop, 
        baseCoreId, levelId, bitWidth, LCN, copyNum
    ):
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        commLoop = commLoops[levelId]
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)
        for i in range(commLoop):
            if levelId == hardwareNoCLevel:
                s2LocalPlaces.append(
                    S2LocalPlace(inputWidth, outputWidth, bitWidth, LCN, self.isOffline)
                )
            if levelId < hardwareNoCLevel:
                # coreXId = (i >> 1) << (hardwareCoreYBit + levelId)
                # coreYId = ((i & 1) << levelId)
                # coreId = baseCoreId | coreXId | coreYId
                coreId = getCore(i, baseCoreId, levelId, hardwareCoreYBit, self.isOffline)
            else:
                coreId = 0
            baseLocation = levelSizes[levelId] * i + baseLoop
            if levelId == 0:
                self.buildSLogicCore(
                    s2LocalPlaces[-1], coreId, baseLocation, levelSizes[0], copyNum
                )
            else:
                self.buildSLogicPlace(
                    s2LocalPlaces, commLoops, levelSizes, baseLocation, coreId,
                    levelId - 1, bitWidth, LCN, copyNum
                )
    
    def minLCN(self):
        #compute LCN
        inputLen = len(self.inputSize)
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth
        LCN = math.ceil(inputLen / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN = tmpLCN
        return LCN

    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        assert len(self.inputSize[0]) == 3
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        inputLen = len(self.inputSize)
        inputNum = np.prod(inputSize)

        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1

        #compute outBlockSize
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", self.isOffline)
        maxNeuronSize = hardwareNeuronNum * inputWidth
        if hardwareType == 'v2':
            if inputWidth == 8:
                inputCombination = LCN * bitWidth
                if inputCombination == 1:
                    maxNeuronSize = 1888
                elif inputCombination == 2:
                    maxNeuronSize = 2728
                elif inputCombination == 4:
                    maxNeuronSize = 3504
                else:
                    maxNeuronSize = 4096
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        vldNeuron = ((hardwareAxonNum // inputWidth) // inputLen) * (bitWidth * LCN)
        vldNeuron = math.ceil(min(maxNeuronSize,vldNeuron) / (bitWidth * LCN))
        

        assert vldNeuron >= copyNum
        neuronNum = math.ceil(vldNeuron / copyNum) * copyNum

        blockNum = math.ceil(inputNum * copyNum / neuronNum)
        outBlockSize = neuronNum
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        # loop scheduling
        outputNum = blockNum
        commLoops = list()
        while outputNum > 1:
            if len(commLoops) >= hardwareNoCLevel:
                commLoops.append(outputNum)
                outputNum = 1
                break
            levelId = len(commLoops)
            NocX = Hardware.getAttr("NoCLevelsX", self.isOffline)[levelId]
            NocY = Hardware.getAttr("NoCLevelsY", self.isOffline)[levelId]
            NocNum = NocX * NocY
            if outputNum > NocNum:
                outputNum = math.ceil(outputNum / NocNum)
                commLoops.append(NocNum)
            else:
                commLoops.append(outputNum)
                outputNum = 1

        if len(commLoops) <= hardwareNoCLevel:
            commLoops += [1 for i in range(hardwareNoCLevel + 1 - len(commLoops))]
        
        levelSizes = [outBlockSize]
        for commLoop in commLoops:
            levelSizes.append(commLoop * levelSizes[-1])
        
        # gen logic place & weightDict & multicast
        weightDict = self.genWeightDict()
        s2LocalPlaces = list()
        self.buildSLogicPlace(
            s2LocalPlaces, commLoops, levelSizes, 0, 
            0, len(commLoops) - 1, bitWidth, LCN, copyNum
        )
        for s2LocalPlace in s2LocalPlaces:
            s2LocalPlace.multicast()

        outputNum = np.prod(outputSize)
        oneChannel = outputSize[1] * outputSize[2]
        biasDict = dict()
        for i in range(outputNum):
            biasDict[i] = self.bias[i // oneChannel]
    
        genSoftwareNetwork(   
            softwareNetwork, s2LocalPlaces, self.name, bitWidth, groupId, 
            LCN, inputNum * inputLen, outputNum, inputWidth, outputWidth, self.isOffline
        )
        lowerMem = -(1 << 6) if self.bitWidth == 1 else -(1 << 31)
        groupId, weightMapping = genLocalPlace(
            s2LocalPlaces, weightDict, biasDict, self.resetMode, self.threshold, 
            self.bitTrunc, groupId, computeGroup, self.mode, self.pool, self.isOffline,
            # below are online core parameters
            np.zeros([60]),  # LUT
            0,               # resetMem
            0,               # lateral_inhi_val
            -(1 << 7),       # lowerWeight
            (1 << 7) - 1,    # upperWeight
            0,               # weightDecay
            lowerMem,        # lowerMem
            0                # learnMode
        )     
        return groupId, weightMapping


#these operators can be set as 'ann'
class Maxpool2dInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "maxpool2d"
        self.kernelSize = [2,2]
        self.inputSize =  [1,2,2]
        self.outputSize = [1,1,1]
        self.padding = [0,0]
        self.stride = [1,1]
        self.dilation = [1,1]
        return
    
    def load(self, info):
        super().load(info)
        self.kernelSize = info['kernelSize']
        self.padding = info['padding']
        self.stride = info['stride']
        self.dilation = info['dilation']
    
    def store(self):
        info = self.__dict__
        return self.__dict__
    
    def inputLen(self):
        return self.kernelSize[0] * self.kernelSize[1]
    
    def minLCN(self):
        # compute LCN
        inputNum = self.kernelSize[0] * self.kernelSize[1]
        assert self.mode == 'ann', self.mode
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM",self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth
        LCN = math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN = tmpLCN
        return LCN

    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        assert self.isOffline
        conv2dInfo = Conv2dInfo()
        conv2dInfo.name = self.name

        conv2dInfo.kernelSize = np.array(
            [
                1, 
                1,
                self.kernelSize[0],
                self.kernelSize[1]
            ]
        )
        conv2dInfo.padding = self.padding
        conv2dInfo.stride = self.stride
        conv2dInfo.dilation = self.dilation
        conv2dInfo.groups = self.inputSize[0][0]
        conv2dInfo.weight = np.ones([
            self.outputSize[0], 
            1, 
            self.kernelSize[0], 
            self.kernelSize[1],
            1]
        )
        conv2dInfo.bias = np.zeros([self.outputSize[0]])
        conv2dInfo.bitWidth = 1
        conv2dInfo.mode = self.mode
        conv2dInfo.bitTrunc = self.bitTrunc
        conv2dInfo.threshold = self.kernelSize[0] * self.kernelSize[1]
        conv2dInfo.memPotential = 0
        conv2dInfo.scale = self.scale
        conv2dInfo.resetMode = self.resetMode

        # conv2dInfo.inputNames = self.inputNames
        # conv2dInfo.outputNames = self.outputNames
        conv2dInfo.inputSize = self.inputSize
        conv2dInfo.outputSize = self.outputSize
        conv2dInfo.pool = self.pool
        conv2dInfo.isOffline = self.isOffline
        groupId, weightMapping = conv2dInfo.transform(
            computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
        )
        return groupId, weightMapping

'''------------------online learning layers defination--------------------'''

class STDPFcInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "STDPfc"
        self.inputSize = [1]
        self.outputSize = [1]
        self.bias = None
        self.weight = None 

        # online learning parameters
        self.isOffline = False
        self.learnMode = True
        self.LUT = None
        self.resetMem = 0
        self.lowerMem = 0
        self.prohibation = 0
        self.lowerWeight = 0
        self.upperWeight = 0
        self.weightDecay = 0
        
    def load(self, info):
        raise NotImplementedError()
        return 
    
    def store(self):
        raise NotImplementedError()
        return
    
    def inputLen(self):
        return self.kernelSize[1]
    
    def genWeightDict(self):
        inputNum = self.inputSize[0][0]
        outputNum = self.outputSize[0]
        weightDict = dict()
        for i in range(outputNum):
            weightDict[i] = dict()
            for j in range(inputNum):
                weightDict[i][j] = self.weight[i,j,:]
        return weightDict

    def buildSLogicCore(self, s2LocalPlace, coreId, baseLocation, outBlockSize, copyNum):
        outputNum = self.outputSize[0]
        inputNum = self.inputSize[0][0]
        outputNumAll = outputNum * copyNum
        if baseLocation >= outputNumAll:
            return 
        axons = set(range(inputNum))
        endNum = min(baseLocation + outBlockSize, outputNumAll)
        neurons = (np.arange(baseLocation, endNum) // copyNum).tolist()
        s2LocalPlace.addCore(coreId, axons, neurons)

    def minLCN(self):
        # compute LCN
        inputSize = self.inputSize[0]
        inputNum = inputSize[0]
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth 
        LCN = math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN = tmpLCN
        return LCN    

    '''------------------------------------------------------------'''
    '''   'mergeSLogicPlaces' is designed for only online cores    '''
    '''------------------------------------------------------------'''
    def mergeSLogicPlaces(self, s2LocalPlaces):
        if not self.learnMode:
            return
        maxNoCLevel   = Hardware.getAttr("MAXNOCLEVEL", self.isOffline)
        maxNoCX = Hardware.getAttr("MAXNoCLevelsX", self.isOffline)
        maxNoCY = Hardware.getAttr("MAXNoCLevelsY", self.isOffline)
        
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL",    self.isOffline)
        hardwareNocX     = Hardware.getAttr("NoCLevelsX", self.isOffline)
        hardwareNocY     = Hardware.getAttr("NoCLevelsY", self.isOffline)
        

        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)
        hardwareCoreBit = Hardware.getAttr("COREBIT", self.isOffline)
        
        def getBase(coreIds, levelId, coreId):
            if levelId < hardwareNoCLevel - 1:
                coreIds.append(coreId)
                return
            elif levelId > hardwareNoCLevel - 1:
                capX = maxNoCX[levelId]
                capY = maxNoCY[levelId]
            else:
                capX = maxNoCX[levelId] // hardwareNocX[levelId]
                capY = maxNoCY[levelId] // hardwareNocY[levelId]

            for x in range(capX):
                for y in range(capY):
                    newCoreId = coreId | (x << (levelId + hardwareCoreYBit)) | (y << (levelId))
                    getBase(coreIds, levelId - 1, newCoreId)

        coreIds = list()
        getBase(coreIds, maxNoCLevel - 1, 0)

        assert len(coreIds) >= len(s2LocalPlaces), f"online cores from a layer cannot be put on a chip"

        newLocalPlace = None
        baseCoreId = -1
        starId = 0
        for i, s2LocalPlace in enumerate(s2LocalPlaces):
            if i == 0:
                inputWidth = s2LocalPlace.inputWidth
                outputWidth = s2LocalPlace.outputWidth
                bitWidth = s2LocalPlace.bitWidth
                LCN = s2LocalPlace.LCN
                newLocalPlace = S2LocalPlace(
                    inputWidth, outputWidth, bitWidth, LCN, self.isOffline
                )
            baseId = coreIds[i]
            for coreId, core in s2LocalPlace.cores.items():
                axons = list(core.axons.keys())
                neurons = core.neurons
                newCoreId = coreId | baseId
                newLocalPlace.addCore(newCoreId, axons, neurons)
                if baseCoreId < 0:
                    baseCoreId = newCoreId
                starId |= (baseCoreId ^ newCoreId)
        
        allCores = multiCast(baseCoreId, starId, hardwareCoreBit, None)
        
        # allCores = set()
        # allCores.add(baseCoreId)
        # for i in range(hardwareCoreBit):
        #     if starId & (1 << i) == 1:
        #         tmpCores = deepcopy(allCores)
        #         star = 1 << i
        #         for core in tmpCores:
        #             allCores.insert(core ^ star)
        
        for coreId in allCores:
            if coreId not in newLocalPlace.cores:
                newLocalPlace.addCore(coreId, [], [])

        return [newLocalPlace]

    def buildSLogicPlace(
        self, s2LocalPlaces, commLoops, levelSizes, baseLoop,
        baseCoreId, levelId, bitWidth, LCN, copyNum
    ):
        inputWidth = 1
        outputWidth = 1

        commLoop = commLoops[levelId]
        hardwareNoCLevel = Hardware.getAttr("NOCLEVEL", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)

        for i in range(commLoop):
            if levelId == hardwareNoCLevel:
                s2LocalPlaces.append(
                    S2LocalPlace(inputWidth, outputWidth, bitWidth, LCN, self.isOffline)
                )
            if levelId < hardwareNoCLevel:
                coreId = getCore(i, baseCoreId, levelId, hardwareCoreYBit, self.isOffline)
            else:
                coreId = 0
            baseLocation = levelSizes[levelId] * i + baseLoop
            if levelId == 0:
                self.buildSLogicCore(
                    s2LocalPlaces[-1], coreId, baseLocation, levelSizes[0], copyNum
                )
            else:
                self.buildSLogicPlace(
                    s2LocalPlaces, commLoops, levelSizes, baseLocation, coreId,
                    levelId - 1, bitWidth, LCN, copyNum
                )

    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN
    ):
        inputSize = self.inputSize[0]
        outputSize = self.outputSize
        if self.mode == 'ann':
            inputWidth = 8
            outputWidth = 8
        else:
            inputWidth = 1
            outputWidth = 1
        
        #compute outBlockSize
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM",   self.isOffline)
        hardwareNoCLevel  = Hardware.getAttr("NOCLEVEL",    self.isOffline)
        maxNoCLevel       = Hardware.getAttr("MAXNOCLEVEL", self.isOffline)
        maxNoCLevelsX = Hardware.getAttr("MAXNoCLevelsX", self.isOffline)
        maxNoCLevelsY = Hardware.getAttr("MAXNoCLevelsY", self.isOffline)
        maxCoreNum = 1
        for i, j in zip(maxNoCLevelsX, maxNoCLevelsY):
            maxCoreNum *= i * j

        vldNeuron = math.ceil(hardwareNeuronNum / (LCN * bitWidth)) * inputWidth
        blockNum = math.ceil((outputSize[0] * copyNum) / vldNeuron)
        outBlockSize = math.ceil((outputSize[0] * copyNum) / blockNum)
        
        # loop scheduling
        outputNum = blockNum
        commLoops = list()

        while outputNum > 1:
            if len(commLoops) >= hardwareNoCLevel:
                commLoops.append(outputNum)
                outputNum = 1
                break
            levelId = len(commLoops)
            NocX = Hardware.getAttr("NoCLevelsX", self.isOffline)[levelId]
            NocY = Hardware.getAttr("NoCLevelsY", self.isOffline)[levelId]
            NocNum = NocX * NocY
            if outputNum > NocNum:
                outputNum = math.ceil(outputNum / NocNum)
                commLoops.append(NocNum)
            else:
                commLoops.append(outputNum)
                outputNum = 1
        
        realBlockNum = 1
        for i in commLoops:
            realBlockNum *= i
        assert realBlockNum <= maxCoreNum, \
            f"One STDP FC layer should be put on a single chip.\n"

        if len(commLoops) <= hardwareNoCLevel:
            commLoops += [1 for i in range(hardwareNoCLevel + 1 - len(commLoops))]
        
        levelSizes = [outBlockSize]
        for commLoop in commLoops:
            levelSizes.append(commLoop * levelSizes[-1])


        # gen logic place & weightDict & multicast
        weightDict = self.genWeightDict()
        s2LocalPlaces = list()

        self.buildSLogicPlace(
            s2LocalPlaces, commLoops, levelSizes, 0, 
            0, len(commLoops) - 1, bitWidth, LCN, copyNum
        )
        if self.learnMode:
            s2LocalPlaces = self.mergeSLogicPlaces(s2LocalPlaces)

        for s2LocalPlace in s2LocalPlaces:
            s2LocalPlace.multicast()

        inputNum = inputSize[0]
        outputNum = outputSize[0]
        biasDict = dict()
        for i in range(outputNum):
            biasDict[i] = self.bias[i]
        
        genSoftwareNetwork(   
            softwareNetwork, s2LocalPlaces, self.name, bitWidth, groupId, 
            LCN, inputNum, outputNum, inputWidth, outputWidth, self.isOffline

        )

        groupId, weightMapping = genLocalPlace(
            s2LocalPlaces, weightDict, biasDict, self.resetMode, self.threshold, 
            self.bitTrunc, groupId,  computeGroup, self.mode, self.pool, self.isOffline,
            # below are online core parameters
            self.LUT, self.resetMem, self.prohibation, self.lowerWeight,
            self.upperWeight, self.weightDecay, self.lowerMem, self.learnMode,
        )
   
        return groupId, weightMapping


'''-----------------------------------------------------------------------'''
'''not used now, the avgpool2d is transformed to conv2d'''
'''-----------------------------------------------------------------------'''

#these operators can be set as 'ann', 'snn'
class Avgpool2dInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "avgpool2d"
        self.kernelSize = [2,2]
        self.inputSize =  [1,2,2]
        self.outputSize = [1,1,1]
        self.padding = [0,0]
        self.stride = [1,1]
        self.dilation = [1,1]
        return
    
    def load(self, info):
        super().load(info)
        self.kernelSize = info['kernelSize']
        self.padding = info['padding']
        self.stride = info['stride']
        self.dilation = info['dilation']
    
    def store(self):
        info = self.__dict__
        return self.__dict__
    
    def inputLen(self):
        return self.kernelSize[0] * self.kernelSize[1]
    
    def minLCN(self):
        # compute LCN
        inputNum = self.kernelSize[0] * self.kernelSize[1]
        if self.mode == 'ann':
            inputWidth = 8
        else:
            inputWidth = 1
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxAxonSize = hardwareAxonNum // inputWidth
        LCN = math.ceil(inputNum / maxAxonSize)
        tmpLCN = 1
        while tmpLCN < LCN:
            tmpLCN <<= 1
        LCN = tmpLCN
        return LCN

    def transform(
        self, computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
    ):
        conv2dInfo = Conv2dInfo()
        conv2dInfo.name = self.name
        conv2dInfo.kernelSize = np.array(
            [
                1, 
                1,
                self.kernelSize[0],
                self.kernelSize[1]
            ]
        )
        conv2dInfo.padding = self.padding
        conv2dInfo.stride = self.stride
        conv2dInfo.dilation = self.dilation
        conv2dInfo.groups = self.inputSize[0][0]
        conv2dInfo.weight = np.ones([
            self.outputSize[0], 
            1, 
            self.kernelSize[0], 
            self.kernelSize[1],
            1]
        )
        conv2dInfo.bias = np.zeros([self.outputSize[0]])
        conv2dInfo.bitWidth = 1
        conv2dInfo.mode = self.mode
        conv2dInfo.bitTrunc = self.bitTrunc
        conv2dInfo.threshold = self.kernelSize[0] * self.kernelSize[1]
        conv2dInfo.memPotential = 0
        conv2dInfo.scale = self.scale
        conv2dInfo.resetMode = self.resetMode
        conv2dInfo.inputSize = self.inputSize
        conv2dInfo.outputSize = self.outputSize
        conv2dInfo.isOffline = self.isOffline
        groupId, weightMapping = conv2dInfo.transform(
            computeGroup, softwareNetwork, bitWidth, copyNum, groupId, LCN, hardwareType
        )
        return groupId, weightMapping


'''-----------------------------------------------------------------------'''
'''the classes below are not used, as these layers are not put on chip'''
'''-----------------------------------------------------------------------'''

# not used now
class ViewInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "view"
        
    def load(self, info):
        super().load(info)
        
    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__

# now used now
class ConcatInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "view"
        
    def load(self, info):
        super().load(info)
        
    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__

# not used now
class ReluInfo(BasicOpInfo):
    def __init__(self):
        super().__init__()
        self.kind = "relu"
        
    def load(self, info):
        super().load(info)
        
    def store(self):
        info = self.__dict__
        if self.weight is not None:
            info['weight'] = self.weight.tolist()
        if self.bias is not None:
            info['bias'] = self.bias.tolist()
        return self.__dict__
    



