from collections import OrderedDict
import numpy as np
import math
from copy import deepcopy
from .hwConfig import Hardware, DataFrame
from .frame import Frame
from .utils import multiCast

class SoftwareInput:
    def __init__(
        self, inputCopyNum, targetLCN, inputWidth, inputNum, timeStep, isOffline
        ):
        self.inputCopyNum = inputCopyNum
        self.inputWidth = inputWidth
        self.targetLCN = targetLCN
        self.timeStep = timeStep
        self.isOffline = isOffline
        self.begTime = 0

        self.axons = [[] for i in range(inputNum)]
        self.stars = [[] for i in range(inputNum)]
        self.neuronPos = np.zeros(inputNum)
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", self.isOffline)
        # steps = math.ceil(self.timeStep / hardwareSlotNum) 
        # self.multiTimeNeurons = [[[] for i in range(inputNum)] for j in range(steps)]
        steps = math.ceil((self.timeStep - hardwareSlotNum//self.targetLCN) / hardwareSlotNum) + 1
        self.multiTimeAxons = [[[] for i in range(inputNum)] for j in range(steps)]
        self.multiTimeStars = [[[] for i in range(inputNum)] for j in range(steps)]
        self.multiTimeTargetLCNs = [ 0 for j in range(steps)]

    def setInputs(self, axons, stars): 
        for i in range(len(axons)):
            self.axons[i] += axons[i]
            self.stars[i] += stars[i]
    
    def needRelay(self):
        # if len(self.multiTimeNeurons) > 1:
        #     return True
        c = [len(axon) > self.inputCopyNum for axon in self.axons]
        for i in c:
            if i:
                return True
        return False
    
    def setBegTime(self, begTime):
        self.begTime = begTime

    def relayInput(self, relayGroup):
        steps = len(self.multiTimeAxons)
        connectionNum = np.zeros(steps)
        if self.begTime > 0:
            relayGroup.newCore()
            for i, axons in enumerate(self.axons):
                if len(axons) > 0:
                    axonFullId, neuronFullIds = relayGroup.relayNeuron(
                         axons[0], len(axons), self.begTime, 1, self.targetLCN, 
                        self.inputWidth, self.isOffline
                    )
                    # self.multiTimeNeurons[0][i] += neuronFullIds
                    self.multiTimeAxons[0][i] += [axonFullId]
                    self.multiTimeStars[0][i] += [0]
                    self.multiTimeTargetLCNs[0] = 1
                    connectionNum[0] += len(axons)
                    for j in range(len(axons)):
                        axonId = axons[j]
                        starId = self.stars[i][j]
                        relayGroup.connect(neuronFullIds[j], axonId, starId)
        else:
            for i, axons in enumerate(self.axons):
                connectionNum[0] += len(axons)
                self.multiTimeAxons[0][i] += axons
                self.multiTimeStars[0][i] += self.stars[i]
                self.multiTimeTargetLCNs[0] = self.targetLCN

        begTime = max(self.begTime, 1)
        for i in range(1, steps):
            relayGroup.newCore()
            targetLCN = self.multiTimeTargetLCNs[i-1]
            
            for j, axons in enumerate(self.multiTimeAxons[i-1]):
                if len(axons) > 0:
                    axonFullId, neuronFullIds = relayGroup.relayNeuron(
                        axons[0], len(axons), begTime, 1, targetLCN, 
                        self.inputWidth, self.isOffline
                    )
                    # self.multiTimeNeurons[i][j] += neuronFullIds
                    self.multiTimeAxons[i][j] += [axonFullId]
                    self.multiTimeTargetLCNs[i] = 1
                    self.multiTimeStars[i][j] += [0]
                    connectionNum[i] += len(axons)
                    assert len(axons) == len(self.multiTimeStars[i-1][j])
                    for k in range(len(axons)):
                        axonId = axons[k]
                        starId = self.multiTimeStars[i-1][j][k]
                        relayGroup.connect(neuronFullIds[k], axonId, starId)

        print(f"relay input steps {steps * self.begTime}")
        for i in range(steps):
            print(f"    {i}: connectionNum = {connectionNum[i]}")

    def relayInput2(self, relayGroup):
        steps = len(self.multiTimeNeurons)
        connectionNum = np.zeros(steps)
        if self.begTime > 0:
            relayGroup.newCore()
            for i, axons in enumerate(self.axons):
                if len(axons) > 0:
                    axonFullId, neuronFullIds = relayGroup.relayNeuron(
                         axons[0], len(axons), self.begTime, 1, self.targetLCN, 
                        self.inputWidth, self.isOffline
                    )
                    self.multiTimeNeurons[0][i] += neuronFullIds
                    self.multiTimeAxons[0][i] += [axonFullId]
                    connectionNum[0] += len(axons)
                    for j in range(len(axons)):
                        axonId = axons[j]
                        starId = self.stars[i][j]
                        relayGroup.connect(neuronFullIds[j], axonId, starId)
            for i in range(1, steps):
                relayGroup.newCore()
                for j, axons in enumerate(self.multiTimeAxons[i-1]):
                    if len(axons) > 0:
                        axonFullId, neuronFullIds = relayGroup.relayNeuron(
                            axons[0], len(axons), self.begTime, 1, 1, 
                            self.inputWidth, self.isOffline
                        )
                        self.multiTimeNeurons[i][j] += neuronFullIds
                        self.multiTimeAxons[i][j] += [axonFullId]
                        connectionNum[i] += len(axons)
                        for k in range(len(axons)):
                            axonId = axons[k]
                            starId = 0
                            relayGroup.connect(neuronFullIds[k], axonId, starId)
        else:
            for i, axons in enumerate(self.axons):
                connectionNum[0] += len(axons)
                self.multiTimeAxons[0][i] += axons
        print(f"relay input steps {steps * self.begTime}")
        for i in range(steps):
            print(f"    {i}: connectionNum = {connectionNum[i]}")

    def store(self):
        inputNum = len(self.axons)
        spikeInputs = [[[] for i in range(inputNum)] for j in range(self.timeStep)]
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", self.isOffline)
        relaySteps = 0
        base = 0
        for i in range(self.timeStep):
            stepLen =  hardwareSlotNum // self.multiTimeTargetLCNs[relaySteps]
            if i >= base + stepLen:
                relaySteps += 1
                base += stepLen
            innerStep = i - base
            LCNBase = self.multiTimeTargetLCNs[relaySteps] * innerStep
            for j in range(inputNum):
                for k, pos in enumerate(self.multiTimeAxons[relaySteps][j]):
                    slotId = (Hardware.getSlotId(pos, self.inputWidth, self.isOffline) + LCNBase) % hardwareSlotNum
                    axonId = Hardware.getAxonId(pos,self.inputWidth, self.isOffline) * self.inputWidth
                    coreId = Hardware.getgPlusCoreId(pos)
                    # starId = 0
                    starId = self.multiTimeStars[relaySteps][j][k]
                    spikeInputs[i][j].append(
                        Frame.makePosFrame(coreId, starId, axonId, slotId)
                    )
                    # spikeInputs[i][j].append(DataFrame.genFakeFrame(coreId, starId, axonId, slotId))
    
            
        return spikeInputs

    def store2(self):
        inputNum = len(self.axons)
        spikeInputs = [[[] for i in range(inputNum)] for j in range(self.timeStep)]
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", self.isOffline)
        if self.begTime:
            for i in range(self.timeStep):
                relaySteps = i // hardwareSlotNum
                if relaySteps == 0:
                    LCNBase = self.targetLCN * i
                else:
                    LCNBase = i
                for j in range(inputNum):
                    for k, pos in enumerate(self.multiTimeAxons[relaySteps][j]):
                        slotId = (Hardware.getSlotId(pos, self.inputWidth, self.isOffline) + LCNBase) % hardwareSlotNum
                        axonId = Hardware.getAxonId(pos,self.inputWidth, self.isOffline) * self.inputWidth
                        coreId = Hardware.getgPlusCoreId(pos)
                        # starId = 0
                        starId = self.multiTimeStars[relaySteps][j][k]
                        spikeInputs[i][j].append(
                            Frame.makePosFrame(coreId, starId, axonId, slotId)
                        )
                        # spikeInputs[i][j].append(DataFrame.genFakeFrame(coreId, starId, axonId, slotId))
        
        else:
            for i in range(self.timeStep):
                LCNBase = self.targetLCN * i
                for j in range(inputNum):
                    for posId, starId in zip(self.axons[j], self.stars[j]):
                        slotId = (Hardware.getSlotId(posId, self.inputWidth, self.isOffline) + LCNBase) % hardwareSlotNum
                        axonId = Hardware.getAxonId(posId, self.inputWidth, self.isOffline)
                        coreId = Hardware.getgPlusCoreId(posId)
                        spikeInputs[i][j].append(
                            Frame.makePosFrame(coreId, starId, axonId, slotId)
                            # DataFrame.genFakeFrame(coreId, starId, axonId, slotId)
                        )
            
        return spikeInputs

class SoftwareOutputs:
    def __init__(self, outputDict):
        self.outputDict = outputDict
        self.outputs = OrderedDict()
        self.LCNs = OrderedDict()
        self.bitWidths = OrderedDict()
    def addOutput(self, outputName, outputNeurons, bitWidth, LCN):
        self.outputs[outputName] = outputNeurons
        self.LCNs[outputName] = LCN
        self.bitWidths[outputName] = bitWidth
    def setOutput(self, computeGroup):
        offset = 0
        for outputName, outputs in self.outputs:
            for output in outputs:
                computeGroup.setOutput(output, offset)
                offset += 1
    def store(self, computeGroup):
        # outputPos = dict()
        # for name, tensorNames in self.outputDict.items():
        #     outputNeurons = list()
        #     for tensorName in tensorNames:
        #         outputNeurons += self.outputs[tensorName]
        #     outputPos[name] = outputNeurons
        # return outputPos
        resOutputs = dict()
        for outputName, outputs in self.outputs.items():
            resOutputs[outputName] = {
                'axons': [computeGroup.getRealOutput(output) for output in outputs],
                'neurons' : [computeGroup.getOutNeuron(output) for output in outputs],
                'bitWidth': self.bitWidths[outputName],
                'LCN': self.LCNs[outputName]
            }
        return resOutputs

class NewSoftwareLayer:
    def __init__(
        self, bitWidth, LCN, inputNum, outputNum, inputWidth, outputWidth, isOffline
    ):
        self.relayInputAxons = [[[] for i in range(inputNum)]] 
        self.relayInputNeurons = [[[] for i in range(inputNum)]] 
        # self.relayOutputs = list()
        self.axons = [[] for i in range(inputNum)]
        self.stars = [[] for i in range(inputNum)]
        
        self.neurons = [[]for i in range(outputNum)]
        self.bitWidth = bitWidth
        self.LCN = LCN
        self.inputWidth = inputWidth
        self.outputWidth = outputWidth
        self.isOffline = isOffline
        self.begTime = 0
    
    def needRelay(self, inputCopyNum):
        c = [inputCopyNum < len(axon) for axon in self.axons]
        for i in c:
            if i:
                return True
        return False

    def setBegTime(self, begTime, computeGroup):
        self.begTime = begTime
        for neuronIds in self.neurons:
            for neuronId in neuronIds:
                computeGroup.setBegTime(neuronId, begTime)

    def connectInputs(self, inputNeuronGroups, inputTimes, computeGroup, relayGroup):
        neuronNum = len(self.neurons)
        minInputTime = min(inputTimes)
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", self.isOffline)
        maxDiff = hardwareSlotNum // self.LCN 
        relaySlot = (self.begTime - minInputTime - maxDiff)
        relayTimes = math.ceil(relaySlot / maxDiff)

        if relaySlot > 0:
            self.relayInputAxons = [deepcopy(self.relayInputAxons[0]) for i in range(relayTimes)]
            self.relayInputNeurons = [deepcopy(self.relayInputNeurons[0]) for i in range(relayTimes)]
        neuronPos = 0
        relayGroup.newCore()
        sneedRelay = False
        connectDict = dict()
        valideNeurons = dict()
        for i in range(len(inputNeuronGroups)):
            inputTime = inputTimes[i]
            inputRelaySlot = (self.begTime - inputTime - maxDiff)
            inputRelayTimes = math.ceil(inputRelaySlot / maxDiff)
            if inputTime == 0: #means input layer
                neuronPos += len(inputNeuronGroups[i])
                continue
            for j in range(len(inputNeuronGroups[i])):
                neuronIds = inputNeuronGroups[i][j]
                needRelay = len(neuronIds) < len(self.axons[neuronPos])
                if needRelay:
                    sneedRelay = True
                if needRelay:
                    relayAxon, relayNeurons = relayGroup.relayNeuron(
                        neuronIds[0], 
                        len(self.axons[neuronPos]) - len(neuronIds) + 1, 
                        inputTime + 1, 
                        self.LCN, 
                        self.LCN, 
                        self.inputWidth,
                        #TODO: maybe need to change in future?
                        self.isOffline
                    )
                    self.relayInputAxons[0][neuronPos] += [relayAxon]
                    self.relayInputNeurons[0][neuronPos] += relayNeurons
                    if neuronIds[0] not in connectDict:
                        connectDict[neuronIds[0]] = 1
                    else:
                        connectDict[neuronIds[0]] += 1
                    
                    computeGroup.connect(neuronIds[0], relayAxon, 0)
                if inputRelaySlot > 0:
                    for neuronId in neuronIds[needRelay:]:
                        relayAxon, relayNeurons = relayGroup.relayNeuron(
                            neuronId, 
                            1, 
                            inputTime + 1, 
                            self.LCN, 
                            self.LCN, 
                            self.inputWidth
                        )
                        self.relayInputAxons[0][neuronPos] += [relayAxon]
                        self.relayInputNeurons[0][neuronPos] += relayNeurons
                        if neuronId not in connectDict:
                            connectDict[neuronId] = 1
                        else:
                            connectDict[neuronId] += 1
                        computeGroup.connect(neuronId, relayAxon, 0)
                        
                    for k in range(1, inputRelayTimes):
                        for neuronId in self.relayInputNeurons[k-1][neuronPos]:
                            relayAxon, relayNeurons = relayGroup.relayNeuron(
                                neuronId, 
                                1, 
                                inputTime + 1, 
                                self.LCN, 
                                self.LCN, 
                                self.inputWidth
                            )
                            self.relayInputAxons[k][neuronPos] += [relayAxon]
                            self.relayInputNeurons[k][neuronPos] += relayNeurons
                            if neuronId not in connectDict:
                                connectDict[neuronId] = 1
                            else:
                                connectDict[neuronId] += 1
                            relayGroup.connect(neuronId, relayAxon, 0, self.LCN)
                    for t, (neuronId, axonId) in enumerate(zip(
                        self.relayInputNeurons[inputRelayTimes - 1][neuronPos], 
                        self.axons[neuronPos]
                    )):
                        if neuronId not in connectDict:
                            connectDict[neuronId] = 1
                        else:
                            connectDict[neuronId] += 1
                        relayGroup.connect(neuronId, axonId, self.stars[neuronPos][t])
                else:
                    for t, (neuronId, axonId) in enumerate(zip(
                        self.relayInputNeurons[0][neuronPos], 
                        self.axons[neuronPos]
                    )):
                        relayGroup.connect(neuronId, axonId, self.stars[neuronPos][t])
                        if neuronId not in connectDict:
                            connectDict[neuronId] = 1
                        else:
                            connectDict[neuronId] += 1
                    num = len(self.relayInputNeurons[0][neuronPos])
                    for t, (neuronId, axonId) in enumerate(zip(
                        neuronIds[needRelay:], 
                        self.axons[neuronPos][num:]
                    )):
                        computeGroup.connect(neuronId, axonId, self.stars[neuronPos][t + num])
                        if neuronId not in connectDict:
                            connectDict[neuronId] = 1
                        else:
                            connectDict[neuronId] += 1
                neuronPos += 1
        validAxonNum = 0
        for axon in self.axons:
            validAxonNum+=len(axon)
        print(f"{len(inputNeuronGroups[0])} / {validAxonNum}")
        print(f"{sneedRelay}: {len(connectDict)}/{sum(connectDict.values())}")
        print("---------------------------------------------------------------")

    def addInputs(self, softAxonId, inputAxons, stars):
        self.axons[softAxonId] += inputAxons
        self.stars[softAxonId] += stars
        return
    
    def addOutputs(self, softNeuronId, outputNeurons):
        self.neurons[softNeuronId] += outputNeurons
        return
    
    def printNeuron(self, softNeuronId):
        print(f"{self.neurons[softNeuronId]}")

    def getOutputs(self, neuronCopyPos, neuronCopyNum):
        neuronCopyEnd = neuronCopyPos + neuronCopyNum
        neurons = [neuron[neuronCopyPos:neuronCopyEnd] for neuron in self.neurons]
        begTime = self.begTime
        return neurons, begTime

    def reBuild(self, computeGroup):
        groupIds = set()
        allInfos = dict()
        
        for neuronIds in self.neurons:
            for neuronId in neuronIds:
                groupId = Hardware.getGroupId(neuronId)
                if groupId in groupIds:
                    continue
                groupIds.add(groupId)
                infos = computeGroup.reBuild(groupId)
                allInfos.update(infos)
        
        axonDict = dict()
        for i, (axons,stars) in enumerate(zip(self.axons, self.stars)):
            for axon, star in zip(axons, stars):
                if star == 0:
                    assert axon not in axonDict or axonDict[axon] == i
                    axonDict[axon] = i
                else:
                    axonSet = {Hardware.getgPlusCoreId(axon)}
                    for j in range(10):
                        if (star >> j) & 1:
                            starId = 1 << j
                            tmpSet = deepcopy(axonSet)
                            for tmp in tmpSet:
                                axonSet.add(tmp^starId)
                    for tmpCoreId in axonSet:
                        axonId = Hardware.getfullId2(tmpCoreId, Hardware.getComAxonId(axon))
                        assert axonId not in axonDict or axonDict[axonId] == i, \
                            f"axonId = {axonId}, i = {i}, axonDict[axonId] = {axonDict[axonId]}, {axonSet} {axon} {star}"
                        axonDict[axonId] = i
        neuronParams = list()
        connections = dict()
        for i, neuronIds in enumerate(self.neurons):
            tmpParam = [
                -1, # leak
                -1, # threshold_pos
                -1, # bitTrunc
                -1, # SNNEN
                -1, # outputWidth
                -1  # bitTrunc
            ]
            connection = dict()
            first = True
            for neuronId in neuronIds:
                gPlusCoreId = Hardware.getgPlusCoreId(neuronId)
                nId = Hardware.getNeuronId(neuronId)
                neuronConfig = allInfos[gPlusCoreId][0][nId]
                if first:
                    tmpParam[0] = neuronConfig[1]
                    tmpParam[1] = neuronConfig[2]
                    tmpParam[2] = neuronConfig[3]
                    tmpParam[3] = allInfos[gPlusCoreId][1]
                    tmpParam[4] = allInfos[gPlusCoreId][2]
                    tmpParam[5] = allInfos[gPlusCoreId][3]
                    for axonId, weight in neuronConfig[0].items():
                        fullAxonId = Hardware.getfullId2(gPlusCoreId, axonId)
                        originAxon = axonDict[fullAxonId]
                        connection[originAxon] = weight
                    first = False
                else:
                    assert tmpParam[0] == neuronConfig[1]
                    assert tmpParam[1] == neuronConfig[2]
                    assert tmpParam[2] == neuronConfig[3]
                    assert tmpParam[3] == allInfos[gPlusCoreId][1]
                    assert tmpParam[4] == allInfos[gPlusCoreId][2]
                    assert tmpParam[5] == allInfos[gPlusCoreId][3]
                    tmpConnection = dict()
                    for axonId, weight in neuronConfig[0].items():
                        fullAxonId = Hardware.getfullId2(gPlusCoreId, axonId)
                        originAxon = axonDict[fullAxonId]
                        tmpConnection[originAxon] = weight
                    assert tmpConnection == connection
            neuronParams.append(tmpParam)
            connections[i] = connection
        return neuronParams, connections


def compute(inputs, connections, parameters, timeStep):
    inputs = inputs.reshape(timeStep, -1)
    membranes = np.zeros(len(connections), dtype=int)
    spikes = np.zeros([timeStep, len(connections)], dtype=int)

    for i in range(timeStep):
        
        for j, (neuron, connection) in enumerate(connections.items()):
            tmp = 0
            if not parameters[neuron][5]: #maxPool
                for axonId, weight in connection.items():
                    tmp = tmp + weight * inputs[i,axonId]
                    assert weight != 0
            else:
                tmp = -(1<<20)
                for axonId, weight in connection.items():
                    tmp = max(tmp, inputs[i,axonId])
            membranes[neuron] = membranes[neuron] + tmp + parameters[neuron][0] #0: leakage

            if parameters[neuron][4] == 1: #outputWidth
                if membranes[neuron] >= parameters[neuron][1]:
                    membranes[neuron] -= parameters[neuron][1]
                    spikes[i,neuron] = 1
            else:
                if membranes[neuron] > 0:
                    spikes[i,neuron] = membranes[neuron] >> (parameters[neuron][2] - 8)
                    mask = (1 << 8) - 1
                    if spikes[i,neuron] > mask:
                        spikes[i, neuron] = mask
                    else:
                        spikes[i, neuron] = spikes[i,neuron] & mask
            if not parameters[neuron][3]: #SNNEN
                membranes[neuron] = 0
    return spikes


def checkOneLayer(net, opOutputs, tensorSizes, inputSizes, softwareNetwork, hardwareNetwork, timeStep):
    from spike_tensor import SpikeTensor
    import torch
    for opName in softwareNetwork.layers.keys():
        inputLen = 0
        inputs = list()
        for inputSize in inputSizes[opName]:
            if net.ops[opName]['op'].mode == 'snn':
                inputs.append(SpikeTensor(torch.ones(timeStep, *inputSize), timeStep,1))
            else:
                inputs.append(torch.ones(timeStep, *inputSize))
                inputs[0].scale = 1
            inputLen += np.prod(inputSize)

        inputs2 = np.ones([timeStep, inputLen])
        outputs = net.ops[opName]['op'](*inputs).data
        
        neuronParams, connections = softwareNetwork.reBuildOneLayer(
            opName, hardwareNetwork.computeGroup
        )
        spikes = torch.tensor(compute(inputs2, connections, neuronParams, timeStep))
        print(f"{opName}(before): {outputs.sum()}")
        print(f"{opName}(before): {outputs.shape}")
        print(f"{opName}(after): {spikes.sum()}")
        print(f"{opName}(after): {spikes.shape}")
        
        outputs = outputs.view(timeStep,-1)
        assert (spikes == outputs).int().sum() == np.prod(tensorSizes[opOutputs[opName][0]]) * timeStep, \
            f"{outputs.sum()} != {spikes.sum()}"

class SoftwareNetwork:
    def __init__(self, onChipNetwork, timeStep):
        self.tensorSizes = onChipNetwork.tensorSizes
        self.inputList = onChipNetwork.inputList
        self.outputDict = onChipNetwork.outputDict
        self.output2Ops = onChipNetwork.output2Ops
        self.input2Ops = onChipNetwork.input2Ops
        self.opInputs = onChipNetwork.opInputs
        self.opOutputs = onChipNetwork.opOutputs
        self.minLCNs = onChipNetwork.selectLCN()

        self.tensorUseTimes = onChipNetwork.getTensorUseTimes()
        self.inputInfo = onChipNetwork.genInputInfo()

        connectionTo, connectionFrom = onChipNetwork.genConnection()
        self.connectionTo = connectionTo
        self.connectionFrom = connectionFrom
        self.timeStep = timeStep

        self.layers = OrderedDict()
        self.inputs = OrderedDict()
        self.outputs = None
    
    def setInputLayers(self):
        for inName in self.inputList:
            infoPara = self.inputInfo[inName]['parameter']
            ops = self.inputInfo[inName]['opInfo']
            inputCopyNum = infoPara[0]
            targetLCN = infoPara[1]
            inputWidth = infoPara[2]
            inputNum = infoPara[3]
            isOffline = infoPara[4]
            softwareInput = SoftwareInput(
                inputCopyNum, targetLCN, inputWidth, 
                inputNum, self.timeStep, isOffline
            )
            for opName, inputPos in ops:
                softwareInput.setInputs(
                    self.layers[opName].axons[inputPos : inputPos + inputNum],
                    self.layers[opName].stars[inputPos : inputPos + inputNum]
                )
            self.inputs[inName] = softwareInput

    def setOutputLayers(self, computeGroup):
        self.outputs = SoftwareOutputs(self.outputDict)
        offset = 0
        for outputName, outputs in self.outputDict.items():
            for output in outputs:
                opName = self.output2Ops[output][0]
                neurons = [neuron[-1] for neuron in self.layers[opName].neurons]
                LCN = self.layers[opName].LCN
                bitWidth = self.layers[opName].bitWidth
                self.outputs.addOutput(output, neurons, bitWidth, LCN)
                for neuronId in neurons:
                    computeGroup.setOutput(neuronId, offset)
                    offset += 1

    def setOutput(self, computeGroup):
        assert self.outputs is not None
        self.outputs.setOutput(computeGroup)

    def setBegTime(self, inputCopyNum, computeGroup):
        maxTime = OrderedDict()
        needRelay = OrderedDict()
        for opName in self.connectionFrom.keys():
            needRelay[opName] = self.layers[opName].needRelay(1)
        for inputName, softInputs in self.inputs.items():
            if softInputs.needRelay():
                maxTime[inputName] = 1
            else:
                maxTime[inputName] = 0
        for layerId, froms in self.connectionFrom.items():
            for inputTensor in froms:
                if inputTensor in self.inputs:
                    maxTime[layerId] = max(maxTime.get(layerId,0), maxTime[inputTensor] + 1)
        for opName in self.connectionFrom.keys():
            maxTime[opName] = maxTime.get(opName, 0)
        for opName, ops in self.connectionTo.items():
            for op in ops:
                maxTime[op] = max(maxTime[op], maxTime[opName] + 1 + needRelay[op])
        for layerId, begTime in maxTime.items():
            if layerId in self.inputs:
                self.inputs[layerId].setBegTime(begTime)
            else:
                self.layers[layerId].setBegTime(begTime, computeGroup)
        print("-----------------------set layer begTime---------------------------")
        nameLen = len(max(maxTime.keys(), key= lambda x:len(x))) + 1
        for name, t in maxTime.items():
            if name in needRelay:
                n = str(needRelay[name])
            else:
                n = "-"
            print(f"{name + ' ' * (nameLen - (len(name)))} | relay {n + ' ' * (6 - len(n))} | beginTime {t}")
        print("-------------------------------------------------------------------")
        return

    def addLayer(
        self, layerId, bitWidth, LCN, inputNum, outputNum, 
        inputWidth, outputWidth, isOffline
    ):
        self.layers[layerId] = NewSoftwareLayer(
            bitWidth, LCN, inputNum, outputNum, inputWidth, outputWidth, isOffline
        )

    def addInputs(self, layerName, softAxonId, inputAxons, stars):
        self.layers[layerName].addInputs(softAxonId, inputAxons, stars)

    def addOutputs(self, layerName, softNeuronId, outputNeurons):
        self.layers[layerName].addOutputs(softNeuronId, outputNeurons)
    
    def print(self, opName, pos):
        self.layers[opName].printNeuron(pos)

    def connect(self, computeGroup, relayGroup, timeSteps):
        neuronPos = dict()
        for layerName in self.layers.keys():
            neuronPos[layerName] = 0
        for inputName, onChipInput in self.inputs.items():
            onChipInput.relayInput(relayGroup)
        for layerName, layer in self.layers.items():
            inputNeuronGroups = list()
            inputTimes = list()
            neuronNum = 0
            for opName in self.connectionFrom[layerName]:
                if opName not in self.inputs:
                    neurons, begTime = self.layers[opName].getOutputs(
                        neuronPos[opName], 1
                    )
                    neuronPos[opName] += 1
                else:
                    neurons = [[] for i in range(len(self.inputs[opName].axons))]
                    begTime = 0
                inputNeuronGroups.append(neurons)
                neuronNum += len(neurons)
                inputTimes.append(begTime)
            print(f"connecting : {layerName}, input Num {neuronNum}, begTime = {inputTimes}")
            self.layers[layerName].connectInputs(
                inputNeuronGroups, inputTimes, computeGroup, relayGroup
            )

    def store(self, computeGroup):
        inputs = dict()
        for name in self.inputList:
            inputs[name] = self.inputs[name].store()

        return inputs, self.outputs.store(computeGroup)

    def reBuildOneLayer(self, layerName, computeGroup):
        return self.layers[layerName].reBuild(computeGroup)

    # def store(self, computeGroup, )

    # def addLayer(self, layerId, layerStatus, LCN):
    #     self.layers[layerId] = SoftwareLayer(layerStatus, LCN)

    # def addNeuron(self, layerId, softNeuronId):
    #     self.layers[layerId].addNeuron(softNeuronId)

    # def addInputs(self, layerId, softNeuronId, destIds, starIds, targetLCNs):
    #     self.layers[layerId].addInputs(softNeuronId, destIds, starIds, targetLCNs)

    # def addOutputs(self, layerId, softNeuronId, neuronIds):
    #     self.layers[layerId].addOutputs(softNeuronId, neuronIds)

    # def connect(self, computeGroup, relayGroup, timeSteps, connections, neededTimes):
    #     needRelay = OrderedDict()
    #     LCNs = OrderedDict()
    #     inputLayers = set()
    #     for layerId, layer in self.layers.items():
    #         needRelay[layerId] = layer.extendWoRelay(computeGroup, relayGroup)
    #         LCNs[layerId] = layer.LCN
    #         if layer.status == "input":
    #             inputLayers.add(layerId) 
        
    #     for layerId, layer in self.layers.items():
    #         layer.connect(computeGroup, relayGroup, timeSteps)
    #     return

