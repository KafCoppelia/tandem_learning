import torch
import numpy as np
import os
import json
import pickle
from copy import deepcopy
from collections import OrderedDict
from multiprocessing import Pool

from hwConfig import Hardware
from spike_tensor import SpikeTensor
from frame import Frame, FrameKind
from hwSim import Simulator, GLOBAL_CORE_ID, STARID
from hwSim import ISSYNC, ISEND

'''-----------------------------------------------------------------------'''
'''                             RUN NETWORK                               '''
'''-----------------------------------------------------------------------'''

def runPreNet(preNet, inputNames, *data):
    if len(preNet.inputs_nodes) > 0:
        preNet(*data)
        return preNet.nodes
    else:
        outData = dict()
        for i in range(len(inputNames)):
            outData[inputNames[i]] = data[i]
        return outData

def runPostNet(postNet, dataDict):
    data = [dataDict[name] for name in postNet.inputs_nodes]
    if len(postNet.inputs_nodes) > 0:
        postNet(*data)
        return postNet.nodes
    else:
        return dataDict

def runOnChipNetwork(simulator, dataPath, outputPath, testFrameDir):
    dataFrames = getData(dataPath)
    dataFrames = [Frame.toInt(frame) for frame in dataFrames]
    frameList, helpInfo = framePartition(dataFrames)
    
    dataNum = 0
    for h in helpInfo:
        dataNum += h
    assert dataNum == 1, dataNum

    outputFrames = list()
    print("run")
    for frames, h in zip(frameList, helpInfo):
        print("run")
        simulator.setInputs(frames)
        print("run")
        if testFrameDir is not None:
            simulator.setDebug(testFrameDir)
        print("run")
        simulator.begin()
        if h:
            outputFrames.append(
                [Frame.toString(frame) for frame in simulator.outputBuffer]
            )
    for outputFrame in outputFrames:
        storeData(outputFrame, outputPath)
    return


'''-----------------------------------------------------------------------'''
'''        load files to help:                                            '''
'''            1. transform tensors to frames                             '''
'''            2. transform frames to tensors                             '''
'''            3. map weights on chip to those in network                 '''
'''-----------------------------------------------------------------------'''

def loadInputFormats(baseDir):
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(formatDir, "formats.pkl"),"rb") as formatInputFile:
        inputFormats = pickle.load(formatInputFile)
    with open(os.path.join(formatDir, "numbers.pkl"),"rb") as numberFile:
        numbers = pickle.load(numberFile)
    with open(os.path.join(formatDir, "inputNames.pkl"),"rb") as nameInputFile:
        inputNames = pickle.load(nameInputFile)
    return inputFormats, numbers, inputNames

def loadOutputFormats(baseDir):
    infoDir = os.path.join(baseDir, "info")
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(infoDir, "outDict.json"),"r") as f:
        outDict = json.load(f)
    with open(os.path.join(infoDir, "shape.json"),"r") as f:
        shapeDict = json.load(f)
    with open(os.path.join(infoDir, "scale.json"),"r") as f:
        scaleDict = json.load(f)
    with open(os.path.join(formatDir, "mapper.txt"),"r") as f:
        mapper = json.load(f)
    intMapper = dict()
    for name, [tensorName, pos] in mapper.items():
        intMapper[int(name)] = [tensorName, int(pos)]
    return outDict, shapeDict, scaleDict, intMapper

def loadWMapFormats(baseDir):
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(formatDir, "weightMapping.json"),'r') as f:
        weightMaps = json.load(f)
    intWeightMaps = dict()
    for opName, weightMap in weightMaps.items():
        intWeightMaps[opName] = dict()
        for neuron, neuronInfo in weightMap.items():
            intNeuron = int(neuron)
            intWeightMaps[opName][intNeuron] = []
            intWeightMaps[opName][intNeuron].append(int(neuronInfo[0]))
            tmpMapping = dict()
            for axon, axonPos in neuronInfo[1].items():
                tmpMapping[int(axon)] = [
                    int(axonPos[0]),  # complete neuron id
                    int(axonPos[1]),  # LCN id
                    int(axonPos[2])   # axon id
                ]
            intWeightMaps[opName][intNeuron].append(tmpMapping)
    return intWeightMaps

'''-----------------------------------------------------------------------'''
'''               encodeDataFrame:  tensor --> frame                      '''
'''               decodeDataFrame:  frame --> tensor                      '''
'''-----------------------------------------------------------------------'''

def encodeDataFrame(dataDict, frameFormats, frameNums, nameLists, filePath):

    with open(filePath, 'w') as f:
    
        # for init frames
        initFrames = "\n".join(frameFormats[:frameNums[0]]) + "\n"
        base = frameNums[0]
        f.write(initFrames)

        # for data frames
        for name in nameLists:
            if hasattr(dataDict[name], "timesteps"):
                data = dataDict[name].data
                data = data.reshape(-1)
                isSNN = True
            else:
                data = dataDict[name].reshape(-1)
                isSNN = False
            for spike, number in zip(data, frameNums[1:]):
                if spike == 0:
                    base += number
                    continue
                if spike < 0:
                    spike += (1 << 8)
                spike = int(spike)
                dataStr = "{:08b}\n".format(spike)
                dataStrframes = dataStr.join(frameFormats[base : base + number]) 
                
                f.write(dataStrframes+dataStr)
                base += number
        
         # for sync frames
        syncFrames = "\n".join(frameFormats[base:]) + "\n"
        if (len(syncFrames) > 1):
            f.write(syncFrames)

    print(f"[generate] Generate INPUT frames in [{filePath}]")

def decodeDataFrame(
    dataFrames, outputDict, shapeDict, 
    scaleDict, mapper, timeStep, coreType):
    dataDict = dict()
    shapeLen = dict()
    timeSteps = dict()
    for name, shape in shapeDict.items():
        shapeLen[name] = np.prod(shape)
        dataDict[name] = torch.zeros(shapeLen[name] * timeStep)
        timeSteps[name] = 0
    
    #for both online and offline  
    hardwareAxonBit = Hardware.getAttr("AXONBIT", True)
    if coreType == 'offline':
        timeSlotNum = 256
    else:
        timeSlotNum = 8
    for frameId, frame in enumerate(dataFrames):
        if frame == '':
            break
        pos = (int(frame[4:24],2) << hardwareAxonBit) + int(frame[37:48],2)
        data = int(frame[-8:],2)
        newTimeStep = int(frame[48:56],2)
        name, tensorPos = mapper[pos]
        if timeSteps[name] % timeSlotNum <= newTimeStep:
            timeSteps[name] += (newTimeStep - timeSteps[name] % timeSlotNum)
            if dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]]:
                timeSteps[name] += (timeSlotNum - (timeSteps[name] % timeSlotNum)) + newTimeStep
        else:
            timeSteps[name] += (timeSlotNum - (timeSteps[name] % timeSlotNum)) + newTimeStep
        assert dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]] ==0, f"{timeSteps[name]}"
        dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]] = data

    for name, outputs in outputDict.items():
        pos = 0
        if len(outputs) == 1 and outputs[0] == name:
            continue
        for output in outputs:
            dataDict[name][pos: (pos + shapeLen[output] * timeStep)] = dataDict[output][:]
            pos += shapeLen[output] * timeStep
    realDataDict = dict()
    realDataSpikeDict = dict()
    for name in outputDict.keys():
        realDataDict[name] = dataDict[name].reshape(timeStep, *shapeDict[name])
    for name in outputDict.keys():
        shape = shapeDict[name]
        scale = np.array(scaleDict[name])
        realDataSpikeDict[name] = realDataDict[name].reshape(timeStep, *shape)
        realDataDict[name] = realDataDict[name].reshape(timeStep, *shape).mean(0) * scale
    return realDataDict, realDataSpikeDict


'''-----------------------------------------------------------------------'''
'''                        load & store frames                            '''
'''-----------------------------------------------------------------------'''

def getData(dataPath):
    with open(dataPath,'r') as f:
        frames = f.readlines()
    frames = [frame.strip() for frame in frames if not frame.startswith("0000")]
    return [frame.strip() for frame in frames]

def storeData(frames, outputPath):
    with open(outputPath,'w') as f:
        f.write("\n".join(frames))
        f.write("\n")
    print(f"[generate] Generate OUTPUT frames in [{outputPath}]")


'''-----------------------------------------------------------------------'''
'''       gen input frames in batch (only called in genInput mode)        '''
'''-----------------------------------------------------------------------'''

def genInputFrames(baseDir, inputMode, preNet, timeSteps):
    inputDir = os.path.join(baseDir, "input")
    frameDir = os.path.join(baseDir, "frames")
    frameFormats, frameNums, inputNames = loadInputFormats(baseDir)
    files = os.listdir(inputDir)
    for dataFile in files:
        if dataFile.startswith(".") or not dataFile.endswith(".pth"):
            continue
        fullFileName = os.path.join(inputDir, dataFile)
        with open(fullFileName, 'rb') as f:
            data = torch.load(f).float()
        
        data = data.unsqueeze(0).expand(timeSteps,*data.shape)
        if inputMode == 'snn':
            data = SpikeTensor(data, timeSteps, 1)
        else:
            data.scale = 1
        dataDict = runPreNet(preNet, inputNames, *[data])
        filePath = os.path.join(frameDir, dataFile[:-4] + "_in.txt")
        encodeDataFrame(dataDict, frameFormats, frameNums, inputNames, filePath)
    return

def single_gen(dataFile, inputDir, timeSteps, inputMode, preNet, inputNames, frameDir, frameFormats, frameNums):
    if dataFile.startswith(".") or not dataFile.endswith(".pth"):
        return
    fullFileName = os.path.join(inputDir, dataFile)
    with open(fullFileName, 'rb') as f:
        data = torch.load(f).float()
    
    data = data.unsqueeze(0).expand(timeSteps,*data.shape)
    if inputMode == 'snn':
        data = SpikeTensor(data, timeSteps, 1)
    else:
        data.scale = 1
    dataDict = runPreNet(preNet, inputNames, *[data])
    filePath = os.path.join(frameDir, dataFile[:-4] + "_in.txt")
    encodeDataFrame(dataDict, frameFormats, frameNums, inputNames, filePath)

def genInputFrames_multiproc(baseDir, inputMode, preNet, timeSteps):
    inputDir = os.path.join(baseDir, "input")
    frameDir = os.path.join(baseDir, "frames")
    frameFormats, frameNums, inputNames = loadInputFormats(baseDir)
    files = os.listdir(inputDir)
    args = []
    for dataFile in files:
        args.append((dataFile, inputDir, timeSteps, inputMode, preNet, inputNames, frameDir, frameFormats, frameNums))
    core_num = 8
    p = Pool(core_num)
    p.map(single_gen, args)
    return

def genOutputFrames(preNet, postNet, baseDir, fullNet, timeSteps, inputMode, coreType):
    inputDir = os.path.join(baseDir, "input")
    frameDir = os.path.join(baseDir, "frames")
    files = os.listdir(inputDir)
    files.sort()
    inputs = list()
    configPath = os.path.join(baseDir, "frames/config.txt")
    simulator = setOnChipNetwork(configPath)
    outDict, shapeDict, scaleDict, mapper = loadOutputFormats(baseDir)
    caseId = 0
    for dataFile in files:
        if dataFile.startswith(".") or not dataFile.endswith(".pth"):
            continue


        fullFileName = os.path.join(inputDir, dataFile)
        with open(fullFileName, 'rb') as f:
            data = torch.load(f).float()
        data = data.unsqueeze(0).expand(timeSteps,*data.shape)
        if inputMode == 'snn':
            data = SpikeTensor(data, timeSteps, 1)
        else:
            data.scale = 1


        inputPath = os.path.join(frameDir, dataFile[:-4] + "_in.txt")
        outputPath = os.path.join(frameDir, dataFile[:-4] + "_out.txt")
        runOnChipNetwork(simulator, inputPath, outputPath, None)

        dataFrames = getData(outputPath)
        
        newDataDict, newDataSpikeDict = decodeDataFrame(
            dataFrames, outDict, shapeDict, scaleDict, mapper, timeSteps, coreType
        )

        weightMaps = loadWMapFormats(baseDir)
        weightSims = dumpWeight(simulator, weightMaps)
        print(f"case --> {dataFile[:-4]}")
        check(fullNet, newDataDict, newDataSpikeDict, scaleDict, weightSims, caseId, *[data.squeeze(0)])
        caseId += 1
        

'''-----------------------------------------------------------------------'''
'''     parse config frames (support both online and offline cores)       '''
'''-----------------------------------------------------------------------'''

def parseConfig(configPath):


    
    '''---------------------------------------------------------------'''
    '''                           offline                             '''
    '''---------------------------------------------------------------'''
    def parseConfig1(frameGroup, coreId):
        return []

    def parseConfig2(frameGroup, coreId):
        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame,2) & ((1 << 30) - 1)
            load = (load << 30) + payLoad
        load = load >> 23
        config = [0] * 11
        dataLen = [2,4,1,1,13,1,15,15,1,4,10]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        for i in range(10, -1, -1):
            config[i] = load & dataMask[i]
            load >>= dataLen[i]
        return config
    
    def parseConfig3(frameGroup, coreId):
        intFrame0 = int(frameGroup[0],2)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        load = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        memBase = 0

        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 4 == 3:
                tmpConfig = [0] * 21
                for j in range(20, -1, -1):
                    tmpConfig[j] = load & dataMask[j]
                    if signedData[j] and (tmpConfig[j] & (1 << (dataLen[j] - 1)) != 0):
                        tmpConfig[j] -= 1 << dataLen[j]
                    load >>= dataLen[j]
                config[neuronId] = tmpConfig
                neuronId += 1
                memBase = 0
        return  config
    
    def parseConfig4_param(frameGroup, coreId):
        intFrame0 = int(frameGroup[0],2)
        # neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        neuronId = 0
        memBase = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                for j in range(5):
                    tmpConfig = [0] * 21
                    for k in range(len(dataLen) - 1, -1, -1):
                        tmpConfig[k] = load & dataMask[k]
                        if signedData[k] and (tmpConfig[k] & (1 << (dataLen[k] - 1)) != 0):
                            tmpConfig[k] -= 1 << dataLen[k]
                        load >>= dataLen[k]
                    config[neuronId] = tmpConfig
                    neuronId += 1
                    # load >>= 214

        return config
    
    def parseConfig4_weight(frameGroup, coreId, isSNN):
        intFrame0 = int(frameGroup[0],2)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        if not isSNN:
            neuronId *= 8
        frameNum = intFrame0 & ((1 << 19) - 1)
        assert starId == 0
        memBase = 0
        config = OrderedDict()
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                if isSNN:
                    weight = np.zeros(1152)
                    for i in range(1152):
                        weight[i] = load & 1
                        load >>= 1
                    config[neuronId] = weight
                    neuronId += 1
                else:
                    for j in range(8):
                        weight = np.zeros(144)
                        for i in range(144):
                            weight[i] = load & 1
                            load >>= 1
                        config[neuronId] = weight
                        neuronId += 1
                load = 0
        return config

    '''---------------------------------------------------------------'''
    '''                            online                             '''
    '''---------------------------------------------------------------'''

    def parseConfigLUT(frameGroup, coreId):

        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame[-30:], 2)
            load = (load << 30) + payLoad
        LUT = np.zeros(60, dtype=int)
        mask = (1 << 8) - 1
        for i in range(59,-1,-1):
            LUT[i] = int(load & mask)
            if LUT[i] & (1 << 7):
                LUT[i] -= (1 << 8)
            load >>= 8
        return  LUT
    
    def parseConfig2ON(frameGroup, coreId):

        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame,2) & ((1 << 30) - 1)
            load = (load << 30) + payLoad
        load = load >> 30
        config = [0] * 18

        dataLen = [
             2,  2, 32, 8,  8,  8, 
            10, 10,  5, 5, 15, 15, 
            60,  1,  1, 1, 10, 16
        ]

        dataMask = [(1 << (i)) - 1 for i in dataLen]

        signedData = [
            0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ]

        for i in range(len(config) - 1, -1, -1):
            config[i] = load & dataMask[i]
            load >>= dataLen[i]
            if i == 13:
                config[i] = load & dataMask[i]
                load >>= 1
            if signedData[i]:
                if config[i] & ((1 << (dataLen[i] - 1))):
                    config[i] -= (1 << dataLen[i])
        return config

    def parseConfig3ON(frameGroup, coreId, bitWidth):
        intFrame0 = Frame.toInt(frameGroup[0])
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        load = 0
        config = dict()

        if bitWidth == 1:
            dataLen = [
                15, 15, 7, 6, 6, 15,  3,  5,
                5,  5, 5, 5, 5, 11, 10, 10,
            ]
        else:
            dataLen = [ 
                32, 32, 32, 32, 32, 32,  3,  5,
                5,  5,  5,  5,  5, 11, 10, 10
            ]

        signedData = [
            1, 1, 1, 1, 1, 1, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        dataMask = [(1 << (i)) - 1 for i in dataLen]

        if bitWidth == 1:
            frameUnit = 2
            frameEnd = 1
            neuronUnit = 1
        else:
            frameUnit = 4
            frameEnd = 3
            neuronUnit = 2

        for i, frame in enumerate(frameGroup[1:]):
            frame = Frame.toInt(frame)
            payLoad = frame & ((1 << 64) - 1)
            load = (load<<64) + payLoad

            if i % frameUnit == frameEnd:
                tmpConfig = [0] * len(dataLen)
                for j in range(len(tmpConfig) - 1, -1, -1):
                    tmpConfig[j] = load & dataMask[j]
                    if signedData[j] and (tmpConfig[j] & (1 << (dataLen[j] - 1)) != 0):
                        tmpConfig[j] -= 1 << dataLen[j]
                    load >>= dataLen[j]
                config[neuronId] = tmpConfig
                neuronId += neuronUnit
        return config
    
    def parseConfig4ON(frameGroup, coreId, bitWidth):
        # intFrame0 = int(frameGroup[0],2)
        intFrame0 = Frame.toInt(frameGroup[0])
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)

        frameNum = intFrame0 & ((1 << 19) - 1)
        config = dict()
        load = 0
        weightVec = list()
        OneComplete = 16 * bitWidth
        weightMask = (1 << bitWidth) - 1
        for i, frame in enumerate(frameGroup[1:]):
            # frame = int(frame,2)
            frame = Frame.toInt(frame)
            payLoad = frame & ((1 << 64) - 1)
            load = (load << 64) + payLoad
            if i % 16 == 15:
                vecNum = 1024 // bitWidth
                for j in range(vecNum):
                    shiftLen = (vecNum - 1 -j) * bitWidth
                    oneWeight = (load >> shiftLen) & weightMask
                    if bitWidth > 1 and oneWeight & (1<<(bitWidth - 1)):
                        oneWeight -= 1 << bitWidth
                    weightVec.append(oneWeight)
                load = 0
            if i % OneComplete == OneComplete - 1:
                weight = np.array(weightVec)
                config[neuronId] = weight
                weightVec.clear()
                neuronId += bitWidth
        return config

    '''--------------------------------------------------------------'''
    '''                         parse frames                         '''
    '''--------------------------------------------------------------'''

    with open(configPath,'r') as f:
        frames = f.readlines()
    frameNum = len(frames)
    configs =  dict()
    i = 0
    
    
    while i < frameNum:
        frame = frames[i].strip()
        # intFrame = int(frame,2)
        intFrame = Frame.toInt(frame)
        frameHead = intFrame >> 60
        coreId = GLOBAL_CORE_ID(intFrame)
        starId = STARID(intFrame)
        assert starId == 0, f"{i}: {frame}"
        # assert coreId & 31 <= 15, coreId
        # assert (coreId >> 5) <= 15, coreId
        # assert coreId & int("1110011100",2) != int("1110011100",2), coreId
        # assert frameHeadM <= frameHead, f"{frameHeadM}: {frames[i]}"
        # frameHeadM = frameHead
        if frameHead == 0:
            k = 0
            while i + k < frameNum:
                tmpIntFrame = Frame.toInt(frames[i+k])
                tmpFrameHead = tmpIntFrame >> 60
                if tmpFrameHead == 0 and GLOBAL_CORE_ID(tmpIntFrame) == coreId:
                    k += 1
                else:
                    break
            if k == 3: # offline
                LUT = parseConfig1(frames[i:i+k], coreId)
            else:
                LUT = parseConfigLUT(frames[i:i+k], coreId)
            if coreId in configs:
                configs[coreId]['LUT'] = LUT
            else:
                configs[coreId] = {'LUT':LUT}
            i += k

        elif frameHead == 1:
            assert coreId in configs, f"{coreId}\n {sorted(configs.keys())}"
            if len(configs[coreId]['LUT']) == 0:
                end = i + 3
            else:
                end = i + 8
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if len(configs[coreId]['LUT']) == 0:
                config = parseConfig2(frameGroup, coreId)
            else:
                config = parseConfig2ON(frameGroup, coreId)
            configs[coreId]['core'] = config
            i = end

        elif frameHead == 2:
            num = intFrame & ((1 << 19) - 1)
            end = i + num + 1
            neuronNum = configs[coreId]['core'][4]
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()

            if len(configs[coreId]['LUT']) == 0:
                config = parseConfig3(frameGroup,coreId)
                isSNN = (configs[coreId]['core'][2] == 0)
                neuronUnit = (1 << configs[coreId]['core'][0]) * (1 << configs[coreId]['core'][1])
                if not isSNN:
                    for neuronId, neuronConfig in config.items():
                        if neuronId * neuronUnit >= neuronNum:
                            continue
                        for j in range(neuronUnit):
                            if neuronId * neuronUnit + j not in configs[coreId]['neuron']:
                                configs[coreId]['neuron'][neuronId * neuronUnit + j] = {
                                    'parameter':None, 
                                    'weight':None
                                }
                            configs[coreId]['neuron'][neuronId * neuronUnit + j]['parameter'] = neuronConfig
                else:
                    for neuronId, neuronConfig in config.items():
                        if neuronId  not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId] = {'parameter':None, 'weight':None}
                        configs[coreId]['neuron'][neuronId]['parameter'] = neuronConfig
            else:
                bitWidth = 1 << configs[coreId]['core'][0]
                config = parseConfig3ON(frameGroup, coreId, bitWidth)
                for neuronId, neuronConfig in config.items():
                    if neuronId % bitWidth != 0:
                        continue
                    for j in range(bitWidth):
                        newId = neuronId + j
                        if newId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][newId] = {
                                'parameter': None,
                                'weight': None
                            }
                        configs[coreId]['neuron'][newId]['parameter'] = neuronConfig

            

            i = end

        elif frameHead == 3:
            num = intFrame & ((1 << 19) - 1)
            neuronId = (intFrame >> 20) & ((1 << 10) - 1)
            end = i + num + 1
            frameGroup = [frames[j].strip() for j in range(i, end)]
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()
            if len(configs[coreId]['LUT']) == 0:
                isSNN = (configs[coreId]['core'][2] == 0)
                neuronUnit = configs[coreId]['core'][0] * (1 << configs[coreId]['core'][1])
                neuronNum = configs[coreId]['core'][4]
                if isSNN:
                    config = parseConfig4_weight(frameGroup, coreId, isSNN)
                    for neuronId, neuronConfig in config.items():
                        if neuronId >= neuronNum:
                            continue
                        if neuronId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                        configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
                else:
                    if neuronId == 0:
                        config = parseConfig4_weight(frameGroup, coreId, isSNN)
                        for neuronId, neuronConfig in config.items():
                            if neuronId >= neuronNum:
                                continue
                            if neuronId not in configs[coreId]['neuron']:
                                configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                            configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
                    else:
                        # assert False
                        config = parseConfig4_param(frameGroup, coreId)
                        neuronUnit = 1<< configs[coreId]['core'][0]     #bitWidth
                        neuronUnit *= (1 << configs[coreId]['core'][1]) #LCN
                        neuronBase = int(512 * neuronUnit)
                        for neuronId, neuronConfig in config.items():
                            base = neuronBase + neuronId * neuronUnit
                            for i in range(neuronUnit):
                                if base >= neuronNum:
                                    continue
                                if base not in configs[coreId]['neuron']:
                                    configs[coreId]['neuron'][base] = {'paramter':None, 'weight': None}
                                configs[coreId]['neuron'][base]['parameter'] = neuronConfig
                                base += 1
            else:
                bitWidth = 1 << configs[coreId]['core'][0]
                config = parseConfig4ON(frameGroup, coreId, bitWidth)
                for neuronId, neuronConfig in config.items():
                    for i in range(bitWidth):
                        newId = neuronId + i
                        # newConfig = (neuronConfig >> (bitWidth - i - 1)) & 1
                        if newId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][newId] = {
                                'parameter': None,
                                'weight': None
                            }
                        configs[coreId]['neuron'][newId]['weight'] = neuronConfig
            i = end
        
        else:
            assert False, frameHead
    
    return configs




'''-----------------------------------------------------------------------'''
'''     partition frames in multiple groups:                              '''
'''         if a group has True helpInfo, it has data frames              '''
'''         else it is just end frames for online cores                   '''
'''-----------------------------------------------------------------------'''

def framePartition(frames):
    beforeSync = True
    onSync = False
    afterSync = False
    preFrames = list()
    postFrames = list()
    frameList = list()
    helpInfo = list()
    for i, frame in enumerate(frames):
        if ISSYNC(frame):
            beforeSync = False
            onSync = True
        else:
            if onSync:
                onSync = False
                afterSync = True
                frameList.append(preFrames)
                helpInfo.append(1)
                preFrames = list()
        if afterSync:
            if ISEND(frame):
                postFrames.append(frame)
            else:
                frameList.append(postFrames)
                helpInfo.append(0)
                postFrames = list()
                beforeSync = True
                onSync = False
                afterSync = False
        else:
            preFrames.append(frame)
    if len(preFrames) > 0:
        frameList.append(preFrames)
        helpInfo.append(1)
    if len(postFrames) > 0:
        frameList.append(postFrames)
        helpInfo.append(0)
    return frameList, helpInfo

def setOnChipNetwork(configPath):
    configs = parseConfig(configPath)
    simulator = Simulator()
    simulator.setConfig(configs)
    return simulator

def dumpWeight(simulator, weightMaps):
    weightDict = dict()
    for opName, weightMap in weightMaps.items():
        weightDict[opName] = dict()
        for neuron, neuronInfo in weightMap.items():
            weightDict[opName][neuron] = dict()
            globalCoreId = neuronInfo[0]
            for axon, axonPos in neuronInfo[1].items():
                weightDict[opName][neuron][axon] = simulator.dumpWeight(
                    globalCoreId, *axonPos
                )
    return weightDict

'''-----------------------------------------------------------------------'''
'''      compare the results from the simulator with groudTrue            '''
'''        1. online cores with learn mode on: check output & weight      '''
'''        2. others                         : check output               '''
'''-----------------------------------------------------------------------'''

def checkOutput(fullNet, dataDict, dataSpikeDict, scaleDict, caseId):
    for name, outData in dataDict.items():
        outDataSpike = dataSpikeDict[name]
        if hasattr(fullNet.nodes[name], "timesteps"):
            gt = fullNet.nodes[name].to_float()
            gt_spike = fullNet.nodes[name].data
            # print("[res]"gt.argmax())
        else:
            gt = fullNet.nodes[name].detach() * np.array(scaleDict[name])
            gt_spike = None
        length = np.prod(gt.shape)
        if gt_spike is not None:
            timePlusLength = np.prod(gt_spike.shape)
        else:
            timePlusLength = None
        print(f"---------------------- [case {caseId}] check result --------------------------")
        print("[info] output tensor name: ", name)
        print("---- GroundTruth result  ---------")
        print(f"shape = {list(gt.shape)}")
        if length <= 10:
            print(gt[0])
            if gt_spike is not None:
                print("spike num: ")
                print(gt_spike.sum(0))
        else:
            print(gt.sum())

        print("-----  onCHIP  result  ---------")
        print(f"shape = {list(outData.shape)}")
        if length <= 10:
            print(outData)
            if gt_spike is not None:
                print("spike num: ")
                print(outDataSpike.sum(0))
        else:
            print(outData.sum())
        # for i in range(100):
        #     for j in range(512):
        #         if gt[i,0, j] != outData[i,0,j]:
        #             assert False, f"{i}, {j}, {gt[i,0, j]} {outData[i,0,j]}"
        assert (gt == outData).int().sum() == length
        if gt_spike is not None:
            assert (gt_spike == outDataSpike).int().sum() == timePlusLength
    print(f"[check] case_{caseId}: output checking pass")

# only support fc layer now
#TODO: support more online learning layers
def checkWeight(fullNet, weightSims, caseId):
    avgWeight = 0
    num = 0
    for opName, weightSim in weightSims.items():
        assert opName in fullNet.ops
        weightGT = fullNet.ops[opName]['op'].weight
        for neuron, neuronInfo in weightSim.items():
            for axon, weight in neuronInfo.items():
                avgWeight += weight
                num += 1
                if weightGT[neuron, axon] != weight:
                    print(
                        f"weight at position [{neuron},{axon}] doesn't match, \
                            GT = {weightGT[neuron, axon]}, sim = {weight}"
                        )
                    assert False
    if num != 0:
        avgWeight /= num
        print(f"weight num = {num}, avg weight = {avgWeight}")
        print(f"[check] case_{caseId}: weight checking pass.")

def check(fullNet, dataDict, dataSpikeDict, scaleDict, weightSims, caseId, *data):
    if fullNet is None:
        return
    # fullNet.ops['fc1']['op'].switch_learn(False)
    # print(data[0].data.shape)
    fullNet(*data)
    checkOutput(fullNet, dataDict, dataSpikeDict, scaleDict, caseId)
    checkWeight(fullNet, weightSims, caseId)
    # checkOutput(fullNet, dataDict, scaleDict, caseId)
    print(f"[check] case_{caseId}: all results checking passed.\n")

def reorgData(dataDict, timeStep):
    dataNum = -1
    newDataDicts = list()
    assert isinstance(dataDict, dict)

    newDataDicts = list()
    dataNum = 0
    for dataName, data in dataDict.items():
        if hasattr(data, "data"):
            dataSize = data.data.size()[1:]
            dataPos = list(np.arange(len(dataSize)) + 2)
            newData = data.data.view(timeStep, -1, *dataSize).permute(1,0,*dataPos)
            dataNum = newData.shape[0]
        else:
            dataSize = data.size()[1:]
            dataPos = list(np.arange(len(dataSize)) + 2)
            newData = data.view(timeStep, -1, *dataSize).permute(1,0,*dataPos)
            dataNum = newData.shape[0]
        if len(newDataDicts) == 0:
            newDataDicts = [dict() for i in range(dataNum)]
        for i in range(dataNum):
            if hasattr(data, "data"):
                newDataDicts[i][dataName] = SpikeTensor(newData[i,...], timeStep, data.scale_factor)
            else:
                newDataDicts[i][dataName] = newData[i,...]
    return dataNum, newDataDicts


def runFullApp(
    preNet, postNet, baseDir, batchedData, fullNet, timeStep, coreType
):
    frameFormats, frameNums, inputNames = loadInputFormats(baseDir)
    dataNum = len(batchedData)
    configPath = os.path.join(baseDir, "frames/config.txt")
    simulator = setOnChipNetwork(configPath)
    outDict, shapeDict, scaleDict, mapper = loadOutputFormats(baseDir)
    for i, data in enumerate(batchedData):
        # input(data[0].data.sum())
        dataDict = runPreNet(preNet, inputNames, *data)
        inputPath = os.path.join(baseDir, f"frames/input_{i}.txt")
        encodeDataFrame(dataDict, frameFormats, frameNums, inputNames, inputPath)

        outputPath = os.path.join(baseDir, f"frames/simuOut_{i}.txt")
        runOnChipNetwork(simulator, inputPath, outputPath, None)
        
        dataFrames = getData(outputPath)
        
        newDataDict, newDataSpikeDict = decodeDataFrame(
            dataFrames, outDict, shapeDict, scaleDict, mapper, timeStep, coreType
        )
        # outdataDict = runPostNet(postNet, newDataDict)

        weightMaps = loadWMapFormats(baseDir)
        weightSims = dumpWeight(simulator, weightMaps)
        check(fullNet, newDataDict, newDataSpikeDict, scaleDict, weightSims, i, *data)

    return
