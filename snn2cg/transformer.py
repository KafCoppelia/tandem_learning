from collections import OrderedDict
from copy import deepcopy
import numpy as np

from .hwConfig import Hardware
from .HardwareNet import HardwareNetwork, LocalPlace, ComputeCore
from .softwareNet import SoftwareNetwork
from .utils import getStar

class OnChipNetwork:
    def __init__(self, coreType):
        self.coreType = coreType
        self.inputList = list()
        self.tensorSizes = dict()
        self.outputDict = dict()
        self.ops = OrderedDict()
        self.output2Ops = dict()
        self.input2Ops = dict()
        self.opInputs = dict()
        self.opOutputs = dict()
        self.outputShapes = dict()
        self.outputScale = dict()
        self.minLCNs = dict()
    
    # get input/output info
    def getIoInfo(self):
        shapeDict = dict()
        scaleDict = dict()
        outputSet = set()
        for name, outputs in self.outputDict.items():
            if name not in self.tensorSizes:
                shapeDict[name] = list(self.outputShapes[name])
            else:
                shapeDict[name] = list(self.tensorSizes[name])
            outputSet.add(name)
            for output in outputs:
                shapeDict[output] = list(self.tensorSizes[output])
                outputSet.add(output)
        for output in outputSet:
            if output not in self.output2Ops:
                if isinstance(self.outputScale[output], float) or \
                    isinstance(self.outputScale[output], int):
                    scaleDict[output] = self.outputScale[output]
                else:
                    scaleDict[output] = self.outputScale[output].tolist()[0]
            else:
                for opName in self.output2Ops[output]:
                    if isinstance(self.ops[opName].scale, float) or \
                        isinstance(self.ops[opName].scale, int):
                        scaleDict[output] = self.ops[opName].scale
                    else:
                        scaleDict[output] = self.ops[opName].scale.tolist()[0]
        return deepcopy(self.outputDict), shapeDict, scaleDict, self.inputList
    
    def genConnection(self):
        connectionTo = OrderedDict()
        connectionFrom = OrderedDict()
        for opName in self.ops.keys():
            inputs = self.opInputs[opName]
            for tensorName in inputs:
                if opName not in connectionFrom:
                    connectionFrom[opName] = list()
                if tensorName not in self.output2Ops:
                    connectionFrom[opName].append(tensorName)
                    continue
                connectionFrom[opName] += self.output2Ops[tensorName]
            outputs = self.opOutputs[opName]
            for tensorName in outputs:
                if tensorName not in self.input2Ops:
                    continue
                if opName not in connectionTo:
                    connectionTo[opName] = list()
                connectionTo[opName] += self.input2Ops[tensorName]
        return connectionTo, connectionFrom

    def selectLCN(self):
        if len(self.minLCNs)>0:
            return self.minLCNs
        minLCNs = dict()
        for opName, op in self.ops.items():
            minLCNs[opName] = op.minLCN()
        
        # find the layers that must have the same LCNs
        equals = list()
        if self.coreType == 'offline':

            for inputTensor in self.inputList:
                tmpEqual = set()
                for opName in self.input2Ops[inputTensor]:
                    for equal in equals:
                        if opName in equal:
                            tmpEqual |= equal
                            equals.remove(equal)
                            break
                    tmpEqual.add(opName)
                equals.append(tmpEqual)
            connectionTo, connectionFrom = self.genConnection()
            for opName, ops in connectionTo.items():
                tmpEqual = set()
                for op in ops:
                    for equal in equals:
                        if opName in equal:
                            tmpEqual |= equal
                            equals.remove(equal)
                            break
                    tmpEqual.add(opName)
                equals.append(tmpEqual)
        else:
            equal = list(self.ops.keys())
            equals.append(equal)
                    
        for equal in equals:
            maxLCN = 0
            for opName in equal:
                maxLCN = max(maxLCN, minLCNs[opName])
            for opName in equal:
                minLCNs[opName] = maxLCN

        self.minLCNs = minLCNs
        return minLCNs

    def genInputInfo(self):
        inputInfo = dict()
        tensors = dict()
        for tensorName, tensorShape in self.tensorSizes.items():
            tensors[tensorName] = np.prod(tensorShape)
        for inName in self.inputList:
            inputInfo[inName] = {'parameter': [1], 'opInfo':list()} 
            # the first parameter in 'parameter' is to control input copy number 
            for opName in self.input2Ops[inName]:
                basePos = 0
                for opInput in self.opInputs[opName]:
                    if opInput == inName:
                        break
                    basePos += tensors[opInput]
                if len(inputInfo[inName]['parameter']) == 1:
                    inputInfo[inName]['parameter'].append(self.minLCNs[opName])
                    if self.ops[opName].mode == 'ann':
                        inputWidth = 8
                    else:
                        inputWidth = 1
                    
                    if self.coreType == "offline":
                        isOffline = True
                    else:
                        isOffline = False

                    inputInfo[inName]['parameter'].append(inputWidth)
                    inputInfo[inName]['parameter'].append(tensors[inName])
                    inputInfo[inName]['parameter'].append(isOffline)
                inputInfo[inName]['opInfo'].append([opName, basePos])
        return inputInfo

    def getTensorUseTimes(self):
        outputTensors = set()
        tensorUseTimes = dict()
        for outputName, outs in self.outputDict.items():
            outputTensors |= set(outs)
        for tensorName in self.tensorSizes.keys():
            useTime = int(tensorName in outputTensors)
            if tensorName in self.input2Ops:
                useTime += len(self.input2Ops[tensorName])
            tensorUseTimes[tensorName] = useTime
        return tensorUseTimes

    def getShapes(self):
        shapeDict = dict()
        for opName, op in self.ops.items():
            shapeDict[opName] = op.inputSize
        return shapeDict

'''
Transformer Helper Class, only used in layer transform
'''
class S2LogicCore:
    def __init__(self, inputWidth, LCN, isOffline):
        hardwareAxonNum = Hardware.getAttr("AXONNUM", isOffline)
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", isOffline)
        self.maxCompleteAxonSize = (hardwareAxonNum * LCN) // inputWidth
        # self.maxNeuronSize = hardwareNeuronNum * inputWidth
        self.inputWidth = inputWidth
        # self.neuronUnitNum = neuronUnitNum
        self.axons = dict()
        self.neurons = list()
        self.axonNum = 0
        self.mask = 0
        self.realNum = 0
        self.restPlace = \
            set(range(self.maxCompleteAxonSize))
    def addAxons(self, axons):
        self.axons.clear()
        self.axons.update(zip(axons,[list() for i in range(len(axons))]))
        self.axonNum = len(axons)
        assert self.axonNum <= self.maxCompleteAxonSize
    def addNeurons(self, neurons):
        self.neurons.clear()
        self.neurons += neurons
    def canPlaceAxon(self, axon):
        if axon in self.axons and len(self.axons[axon]) == 0:
            return True
        can = (self.axonNum + 1 <= self.maxCompleteAxonSize)
        return can
    def needPlace(self, axon):
        if axon in self.axons and len(self.axons[axon]) > 0:
            return False
        return True
    def placeAxon(self, axon, position):
        self.realNum += 1
        if not(axon in self.axons and len(self.axons[axon]) == 0):
            self.axonNum += 1
        if axon not in self.axons:
            self.axons[axon] = list()
        self.axons[axon].append(position)
        self.restPlace.remove(position)

'''
Transformer Helper Class, only used in layer transfome
'''
class S2LocalPlace:
    def __init__(self, inputWidth, outputWidth, bitWidth, LCN, isOffline):
        self.inputWidth = inputWidth
        self.outputWidth = outputWidth
        self.bitWidth = bitWidth
        self.isOffline = isOffline
        self.LCN = LCN
        self.cores = dict()
        self.axons = dict()
        self.dests = dict()
        self.stars = dict()

    def addCore(self, coreId, axons, neurons):
        core = S2LogicCore(
            self.inputWidth, self.LCN, self.isOffline
        )
        assert coreId not in self.cores, coreId
        core.addAxons(axons)
        core.addNeurons(neurons)
        self.cores[coreId] = core
        for axon in axons:
            if axon not in self.axons:
                self.axons[axon] = list()
            self.axons[axon].append(coreId)
        return
    
    def multicast(self):
        axonNums = list()
        for axonId, coreList in self.axons.items():
            axonNums.append([axonId, len(coreList)])
            assert len(coreList) > 0
        axonNums.sort(key=lambda x: x[1], reverse=True)
        hardwareCoreXBit = Hardware.getAttr("COREXBIT", self.isOffline)
        hardwareCoreYBit = Hardware.getAttr("COREYBIT", self.isOffline)
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        for (axonId, coreNum) in axonNums:
            coreList = self.axons[axonId]
            base = coreList[0]
            star = 0
            for core in coreList:
                star |= (base ^ core)
            coreSet = {base}
            for i in range(hardwareCoreXBit + hardwareCoreYBit):
                mask = 1 << i
                if star & mask:
                    tmpCoreSet = deepcopy(coreSet)
                    for coreId in tmpCoreSet:
                        coreSet.add(coreId ^ mask)
            
            canPlace = len(coreSet) < 1.5 * len(coreList)
            selectedAxon = -1
            for coreId in coreSet:
                if coreId in self.cores and not self.cores[coreId].canPlaceAxon(axonId):
                    canPlace = False
                    break
            if canPlace:
                axonSet = None
                for coreId in coreSet:
                    if coreId not in self.cores:
                        tmpAxonSet = set(
                            range(hardwareAxonNum * self.LCN // self.inputWidth)
                        )
                    else:
                        tmpAxonSet = deepcopy(self.cores[coreId].restPlace)
                    if axonSet is None:
                        axonSet = tmpAxonSet
                    else:
                        axonSet &= tmpAxonSet
                if axonSet is None or len(axonSet) == 0:
                    canPlace = False
                else: 
                    selectedAxon = min(axonSet)
            if axonId not in self.dests:
                self.dests[axonId] = list()
                self.stars[axonId] = list()
            if canPlace:
                for coreId in coreSet:
                    if coreId in self.cores:
                        self.cores[coreId].placeAxon(axonId, selectedAxon)
                    else:
                        self.addCore(coreId, [], [])
                        self.cores[coreId].placeAxon(axonId, selectedAxon)
                fullId = Hardware.getfullId2(base, selectedAxon)
                self.dests[axonId].append(fullId)
                self.stars[axonId].append(star)
            else:
                for coreId in coreList:
                    assert axonId in self.cores[coreId].axons
                    if self.cores[coreId].needPlace(axonId):
                        position = min(self.cores[coreId].restPlace)
                        self.cores[coreId].placeAxon(axonId, position)
                        fullId = Hardware.getfullId2(coreId, position)
                        self.dests[axonId].append(fullId)
                        self.stars[axonId].append(0)
        self.check()

    def check(self):
        axonsPlaced = dict()
        for name, dests in self.dests.items():
            stars = self.stars[name]
            for dest, star in zip(dests, stars):
                if star == 0:
                    assert dest not in axonsPlaced or axonsPlaced[dest] == name
                    axonsPlaced[dest] = name
                    continue
                else:
                    s = {Hardware.getCoreId(dest)}
                    for j in range(10):
                        if (star >> j) & 1:
                            starId = 1 << j
                            tmpS = deepcopy(s)
                            for tmp in tmpS:
                                s.add(tmp ^ starId)
                    for tmpCoreId in s:
                        axonId = Hardware.getfullId2(
                            tmpCoreId, 
                            Hardware.getComAxonId(dest), 
                        )
                        assert axonId not in axonsPlaced or axonsPlaced[axonId] == name, \
                            f"{axonId}, {name} {axonsPlaced[axonId]}"
                        axonsPlaced[axonId] = name

def genSoftwareNetwork(
    softwareNetwork, s2LocalPlaces, layerName, bitWidth, groupId, 
    LCN, inputNum, outputNum, inputWidth, outputWidth, isOffline
):
    tmpGroupId = groupId
    softwareNetwork.addLayer(
        layerName, bitWidth, LCN, inputNum, outputNum, inputWidth, outputWidth, isOffline
    )

    for s2LocalPlace in s2LocalPlaces:
        for axonId, dests in s2LocalPlace.dests.items():
            stars = s2LocalPlace.stars[axonId]
            globalDests = [Hardware.addGroupId(tmpGroupId, dest) for dest in dests]
            softwareNetwork.addInputs(layerName, axonId, globalDests, stars)
        coreNames = list(s2LocalPlace.cores.keys())
        coreNames.sort()
        for coreId, s2Core in s2LocalPlace.cores.items():
            fullId = Hardware.getfullId(tmpGroupId, coreId, 0)
            for i, neuronId in enumerate(s2Core.neurons):   
                softwareNetwork.addOutputs(
                    layerName,
                    neuronId,
                    [fullId]
                ) 
                fullId += 1

        tmpGroupId += 1

'''
    weightMapping Info:
    {
        "LCN": LCN,
        "bitWidth": bitWidth
        "weight":{
            neuron_1: [
                globalNeuronId_1,
                { axon_1: axonPos_1, axon_2: axonPos_2}
            ]
        }
    }
'''

def storeWeightMapping(
    neurons, axonMapping, weightDict, neuronSet, 
    groupId, coreId, LCN, bitWidth
):
    mapping = dict()
    globalCoreId = Hardware.getgPlusCoreId2(groupId, coreId)
    for i, neuron in enumerate(neurons):
        assert neuron not in neuronSet
        oneWeight = dict()
        for axon in weightDict[neuron]:
            axonPos = axonMapping[axon][0]
            assert len(axonMapping[axon]) == 1
            oneWeight[axon] = axonPos
        mapping[neuron] = [
            Hardware.getfullId2(globalCoreId, i),
            oneWeight 
        ]
    return mapping

def genLocalPlace(
    s2LocalPlaces, weightDict, biasDict, resetMode, threshold, 
    bitTrunc, groupId, computeGroup, mode, pool, isOffline, *onlineParameters,
):
    if mode == 'ann':
        SNNEN = 0
    else:
        SNNEN = 1
    
    dumpWeight = False
    if not isOffline:
        if onlineParameters[7]:
            dumpWeight = True

    if not isOffline:
        onlineParameters = list(onlineParameters)
        onlineParameters.append(0)

    neuronSet = set()
    weightMapping = dict()

    #for dumping weight
    if dumpWeight:
        tmpGroupId = deepcopy(groupId)
        axonNum = Hardware.getAttr("AXONNUM", isOffline)
        for s2LocalPlace in s2LocalPlaces:
            if 'LCN' not in weightMapping:
                weightMapping['LCN'] = s2LocalPlace.LCN
                weightMapping['bitWidth'] = s2LocalPlace.bitWidth
                weightMapping['axonNum'] = axonNum
                weightMapping['weight'] = dict()
            for coreId, s2Core in s2LocalPlace.cores.items():
                mapping = storeWeightMapping(
                    s2Core.neurons, s2Core.axons, weightDict, 
                    neuronSet, tmpGroupId, coreId, s2LocalPlace.LCN, 
                    s2LocalPlace.bitWidth
                )
                weightMapping['weight'].update(mapping)
            tmpGroupId += 1
                    
    for s2LocalPlace in s2LocalPlaces:
        localPlace = LocalPlace(isOffline)
        if not isOffline:
            if onlineParameters[7] == 1:
                inhiCoreStar = getStar(s2LocalPlace.cores.keys())
            else:
                inhiCoreStar = 0
            onlineParameters[-1] = inhiCoreStar
        for coreId, s2Core in s2LocalPlace.cores.items():
            computeCore = ComputeCore(
                s2LocalPlace.LCN, s2LocalPlace.bitWidth, SNNEN,
                s2LocalPlace.inputWidth, s2LocalPlace.outputWidth,
                int(pool), s2Core.axons, isOffline,
                *onlineParameters
                )
            
            for neuronId, neuron in enumerate(s2Core.neurons):
                assert neuron in weightDict,f"{neuronId} : {neuron}"
                computeCore.addNeuron(
                    neuronId, weightDict[neuron], biasDict[neuron], 
                    resetMode, threshold, bitTrunc
                )

            # fullCoreId = Hardware.addBaseCoreId(coreId, isOffline)
            fullCoreId = coreId
            localPlace.addCore(fullCoreId, computeCore)
        computeGroup.addLocalPlace(groupId,localPlace)
        groupId += 1
    return groupId, weightMapping

def transform(onChipNet, bitWidth, inputCopyNum, timeStep, hardwareType):
    hardwareNetwork = HardwareNetwork()
    softwareNetwork = SoftwareNetwork(onChipNet, timeStep)

    groupId = 0
    tensorUseTimes = onChipNet.getTensorUseTimes()
    minLCNs = onChipNet.minLCNs
    weightMappings = dict()
    for opName, op in onChipNet.ops.items():
        outputTensor = onChipNet.opOutputs[opName][0]
        useTimes = tensorUseTimes[outputTensor]
        groupId, weightMapping = op.transform(
            hardwareNetwork.computeGroup, 
            softwareNetwork, bitWidth, 
            useTimes, groupId, minLCNs[opName], hardwareType
        )
        if len(weightMapping) > 0:
            weightMappings[opName] = weightMapping

    hardwareNetwork.beginRelay()

    softwareNetwork.setInputLayers()
    softwareNetwork.setOutputLayers(hardwareNetwork.computeGroup)
    softwareNetwork.setBegTime(inputCopyNum, hardwareNetwork.computeGroup)

    softwareNetwork.connect(
        hardwareNetwork.computeGroup, 
        hardwareNetwork.relayGroup,
        timeStep
    )
    # softwareNetwork.setOutput(hardwareNetwork.computeGroup)
    # hardwareNetwork.store(timeStep)

    return hardwareNetwork, softwareNetwork, weightMappings
