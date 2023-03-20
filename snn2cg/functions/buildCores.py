import sys
sys.path.append("..")
sys.path.append(".")
from HardwareNet import *
from genConfig import *

from hwConfig import Hardware
from utils import multiCast
from runtime import setOnChipNetwork, runOnChipNetwork
import numpy as np
from tqdm import tqdm
from time import time
import math
import os
import json

import argparse

'''
config:
{
    type: 'FPGA' or 'ASIC',
    offline: [0****_0****, 00***_11***],
    online: [111**_111**],
    neuron_offline: 16,
    neuron_online: 4096,
    destChip: 10000_00000,
    Vthr: 1, 
    inputWidth: 8,
    outputWidth: 1,
    mode: ann,
    timestep: 1,
    connect: "all"
}
'''

def getDestOffline1(neuronId, coreId, offlineAxonMAX, destChip):
    groupId = neuronId // offlineAxonMAX
    assert groupId < 2
    groupId = destChip + (groupId << (Hardware.COREBIT - 2))
    axonId = neuronId % offlineAxonMAX
    fullAxonId = Hardware.getfullId(groupId, coreId, axonId)
    return fullAxonId

def getDestOffline2(neuronId, coreId, offlineAxonNum, destChip, connection_rate):
    groupId = destChip
    if connection_rate > 0:
        axonId = (neuronId % int(round(offlineAxonNum * connection_rate)))
        axonId += offlineAxonNum
    else:
        axonId = offlineAxonNum
    fullAxonId = Hardware.getfullId(groupId, coreId, axonId)
    return fullAxonId

def getDestOnline1(destChip, coreId, neuronId):
    fullAxonId = Hardware.getfullId(destChip, coreId, neuronId)
    return fullAxonId

def getDestOnline2(destChip, coreId, neuronId, onlineAxonNum, connection_rate):
    if connection_rate > 0:
        axonId = neuronId % int(round(onlineAxonNum * connection_rate))
    else:
        axonId = 0
    fullAxonId = Hardware.getfullId(destChip, coreId, axonId) + onlineAxonNum
    return fullAxonId

connection_patterns = {
    0            : (2 << 5) + 2,
    (1 << 5)     : (3 << 5) + 2,
    (2 << 5)     : 2,
    3 << 5       : (1 << 5) + 2,
    1            : (2 << 5) + 3,
    (1 << 5) + 1 : (3 << 5) + 3,
    (2 << 5) + 1 : 3,
    (3 << 5) + 1 : (1 << 5) + 3,
}


def getDestOffline3(neuronId, coreId, offlineAxonNum):
    global connection_patterns
    if len(connection_patterns) == 8:
        tmp_connection_patterns = dict()
        for key,val in connection_patterns.items():
            tmp_connection_patterns[key] = val
            tmp_connection_patterns[val] = key
        connection_patterns = tmp_connection_patterns
    coreId_id = (coreId & (3 << 5)) + (coreId & 3)
    destCore = connection_patterns[coreId_id]
    destCore = int(destCore + (coreId - coreId_id))
    axonId = neuronId + offlineAxonNum
    fullAxonId = Hardware.getfullId(0, destCore, axonId)
    return fullAxonId

def getDestOnline3(neuronId, coreId, onlineAxonNum):
    coreId_id = (coreId & (3 << 5)) + (coreId & 3)
    destCore = connection_patterns[coreId_id]
    destCore = int(destCore + (coreId - coreId_id))
    axonId = neuronId + onlineAxonNum
    fullAxonId = Hardware.getfullId(0, destCore, axonId)
    return fullAxonId
    

def readConfig(configPath):
    with open(configPath) as f:
        config = json.load(f)
    return config

def parseConfig(config, num):
    onlineCores = set()
    offlineCores = set()
    excludeCores = set()

    if 'exclude' in config:
        for core in config['exclude']:
            addrs = core.split("_")
            coreX = addrs[0]
            coreAddr = 0
            star = 0
            for i in range(len(coreX)):
                if coreX[i] == '0' or coreX[i] == '1':
                    coreAddr |= int(coreX[i]) << (Hardware.COREX - 1 - i + Hardware.COREY)
                else:
                    star |= 1 << (Hardware.COREX - 1 - i + Hardware.COREY)

            coreY = addrs[1]
            for i in range(len(coreY)):
                if coreY[i] == '0' or coreY[i] == '1':
                    coreAddr |= int(coreY[i]) << (Hardware.COREY - 1 - i)
                else:
                    star |= 1 << (Hardware.COREY - 1 - i)

            cores = multiCast(coreAddr, star, Hardware.COREBIT, None)
            excludeCores |= cores

    if num is not None:
        if num < 1024:
            online_core_strs = []
        else:
            online_core_strs = ["111**_111**"]
    else:
        online_core_strs = config['online']
    for core in online_core_strs:
        addrs = core.split("_")
        coreX = addrs[0]
        coreAddr = 0
        star = 0
        for i in range(len(coreX)):
            if coreX[i] == '0' or coreX[i] == '1':
                coreAddr |= int(coreX[i]) << (Hardware.COREX - 1 - i + Hardware.COREY)
            else:
                star |= 1 << (Hardware.COREX - 1 - i + Hardware.COREY)

        coreY = addrs[1]
        for i in range(len(coreY)):
            if coreY[i] == '0' or coreY[i] == '1':
                coreAddr |= int(coreY[i]) << (Hardware.COREY - 1 - i)
            else:
                star |= 1 << (Hardware.COREY - 1 - i)

        cores = multiCast(coreAddr, star, Hardware.COREBIT, None)
        onlineCores |= (cores - excludeCores)

    if num is not None:
        if num ==1:
            offline_core_strs = ["00000_00000"]
        elif num == 4:
            offline_core_strs = ["0000*_0000*"]
        elif num == 16:
            offline_core_strs = ["000**_000**"]
        elif num == 32:
            offline_core_strs = ["00***_000**"]
        elif num == 64:
            offline_core_strs = ["00***_00***"]
        elif num == 128:
            offline_core_strs = ["0****_00***"]
        elif num == 256:
            offline_core_strs = ["0****_0****"]
        elif num == 512:
            offline_core_strs = ["*****_0****"]
        elif num == 768:
            offline_core_strs = ["*****_0****", "0****_1****"]
        elif num == 1024:
            offline_core_strs = ["*****_*****"]
        else:
            offline_core_strs = ["111**_111**"]
    else:
        offline_core_strs = config['offline']

    for core in offline_core_strs:
        addrs = core.split("_")
        coreX = addrs[0]
        coreAddr = 0
        star = 0
        for i in range(len(coreX)):
            if coreX[i] == '0' or coreX[i] == '1':
                coreAddr |= int(coreX[i]) << (Hardware.COREX - 1 - i + Hardware.COREY)
            else:
                star |= 1 << (Hardware.COREX - 1 - i + Hardware.COREY)

        coreY = addrs[1]
        for i in range(len(coreY)):
            if coreY[i] == '0' or coreY[i] == '1':
                coreAddr |= int(coreY[i]) << (Hardware.COREY - 1 - i)
            else:
                star |= 1 << (Hardware.COREY - 1 - i)

        cores = multiCast(coreAddr, star, Hardware.COREBIT, None)
        newCores = cores - onlineCores - excludeCores
        offlineCores |= newCores
    
    destChipStrs = config['destChip'].split("_")
    destChip = (int(destChipStrs[0], 2) << Hardware.COREY) + int(destChipStrs[1],2)



    SNNEN = config['mode'] == 'snn'
    inputWidth = int(config['inputWidth'])
    outputWidth = int(config['outputWidth'])
    offlineNeuronNum = int(config['neuron_offline'])
    onlineNeuronNum = int(config['neuron_online'])
    if config['Vthr'] == "MAX":
        threshold = "MAX"
    else:
        threshold = int(config['Vthr'])
    connect = config['connect']
    timeStep = int(config['timestep'])
    LCN = int(config['LCN'])

    offlineCores = list(offlineCores)
    onlineCores = list(onlineCores)
    offlineCores.sort()
    onlineCores.sort()

    # parsedConfig = {
    #     'type': config['type'],
    #     "offlineCores": offlineCores,
    #     "onlineCores": onlineCores,
    #     "mode": SNNEN,
    #     "inputWidth": inputWidth,
    #     "outputWidth": outputWidth,
    #     "offlineNeuronNum": offlineNeuronNum,
    #     "onlineNeuronNum": onlineNeuronNum,
    #     "threshold": threshold,
    #     "timestep": timeStep,
    #     "destChip": destChip,
    #     "connect": connect,
    # }

    # return parsedConfig

    return config['type'], offlineCores, onlineCores, SNNEN, inputWidth, outputWidth ,\
        offlineNeuronNum, onlineNeuronNum, threshold, timeStep, destChip, connect, LCN

def getWeight(axonNum, rate):
    if rate >= 1.0:
        weight = np.ones(axonNum).astype(int)
    elif rate > 0:
        weight = (np.random.rand(axonNum) < rate).astype(int)
    else:
        weight = np.zeros(axonNum)
    if rate > 0 and weight.sum() == 0:
        weight[0] = 1
    return dict(zip(range(axonNum), weight))

def buildCores(
    offlineCores, onlineCores, SNNEN, inputWidth, outputWidth,
    offlineNeuronNum, onlineNeuronNum, threshold, timeStep, destChip, rate, LCN, connection_rate
):
    if connection_rate > 1.0:
        connection_rate = 1.0
    offlineAxonNum = (Hardware.getAttr("AXONNUM", True) // inputWidth)
    offlineAxonNum_LCN = offlineAxonNum * LCN
    onlineAxonNum  = Hardware.getAttr("AXONNUM", False) 
    onlineAxonNum_LCN = onlineAxonNum * LCN
    offlineAxonPlace = dict(
        zip(
            range(offlineAxonNum_LCN), 
            [[i] for i in range(offlineAxonNum_LCN)]
        )
    )
    onlineAxonPlace = dict(
        zip(
            range(onlineAxonNum_LCN), 
            [[i] for i in range(onlineAxonNum_LCN)]
        )
    )
    # offlineWeights = dict(zip(range(offlineAxonNum), [1] * offlineAxonNum))
    # onlineWeights = dict(zip(range(onlineAxonNum), [1] * onlineAxonNum))

    offlineAxonMAX = (1 << Hardware.getAttr("AXONBIT", True))
    if threshold == "MAX":
        offlineThreshold = (1 << 28) - 1
        onlineThreshold = (1 << 14) - 1
    else:
        offlineThreshold = threshold
        onlineThreshold = threshold

    onlineParameters = [
        [],
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0
    ]

    onlineParameters2 = [
        np.zeros(60, dtype=int),
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        0
    ]
    
    localPlace = LocalPlace(True)
    for core in tqdm(offlineCores,"build offline cores"):
        computeCore = ComputeCore(
            LCN, 1, SNNEN, inputWidth, outputWidth, 0, offlineAxonPlace, True,
            *onlineParameters
        )
        for i in range(offlineNeuronNum):
            offlineWeights = getWeight(offlineAxonNum_LCN, rate)
            # computeCore.addNeuron(
            #     i, offlineWeights, 0, "subtraction", offlineThreshold, 8,
            # )
            computeCore.addNeuron(
                i, offlineWeights, 0, "reset", offlineThreshold, 8,
            )
        localPlace.addCore(core, computeCore)
        for i in range(offlineNeuronNum):
            # groupId = i // offlineAxonMAX
            # assert groupId < 2
            # groupId = destChip + (groupId << (Hardware.COREBIT - 2))
            # axonId = i % offlineAxonMAX
            # fullAxonId = Hardware.getfullId(groupId, core, axonId)
            # fullAxonId = getDestOffline1(i, core, offlineAxonMAX, destChip)
            if destChip != 0:
                fullAxonId = getDestOffline1(i, core, offlineAxonMAX, destChip)
            else:
                # fullAxonId = getDestOffline2(i, core, offlineAxonNum, destChip,connection_rate)
                fullAxonId = getDestOffline3(i, core, offlineAxonNum)
            localPlace.connect(core, i, fullAxonId, 0)
            localPlace.setBegTime(core, 1)
    
    for core in tqdm(onlineCores, "build online cores"):
        onlineLCN = min(LCN, 8)
        computeCore = ComputeCore(
            onlineLCN, 1, SNNEN, 1, 1, 0, onlineAxonPlace, False,
            *onlineParameters2
        )

        for i in range(onlineNeuronNum):
            onlineWeights = getWeight(onlineAxonNum_LCN, rate)
            computeCore.addNeuron(
                i, onlineWeights, 0, "subtraction", onlineThreshold, 0
            )
        localPlace.addCore(core, computeCore)
        for i in range(onlineNeuronNum):
            # fullAxonId = Hardware.getfullId(destChip, core, i)
            if destChip != 0:
                fullAxonId = getDestOnline1(destChip, core, i)
            else:
                # fullAxonId = getDestOnline2(destChip, core, i, onlineAxonNum, connection_rate)
                fullAxonId = getDestOnline3(i, core, onlineAxonNum)
            localPlace.connect(core, i, fullAxonId, 0)
            localPlace.setBegTime(core, 1)
    
    localPlaceInfo, coreIds = localPlace.store(0, timeStep)
    print("[info] info generated")
    info = {
        'localPlace': {0:localPlaceInfo},
        'relay': {}
    }
    return info

def genInputFrames(
    chipMaxTime, baseOnlineCores, lateralStars, onlineModes, offlineCores, 
    onlineCores, connect, inputWidth, frameDir, inputData, LUT, connection_rate, hardwareType
):
    destinations = list()
    frames = list()
    if connection_rate > 1.0:
        connection_rate = 1.0
    if connect == 'all':
        offlineAxonNum = int(round((Hardware.getAttr("AXONNUM", True) // inputWidth) * connection_rate))
        onlineAxonNum = int(round(Hardware.getAttr("AXONNUM", False) * connection_rate))
    else:
        offlineAxonNum = 1
        onlineAxonNum = 1
    for core in offlineCores:
        if inputWidth == 1:
            destinations += [[core, 0, i, j, inputData] for i in range(offlineAxonNum) for j in range(LUT)]
        else:
            destinations += [
                [core, 0, int(inputWidth * i), j, inputData] for i in range(offlineAxonNum) for j in range(LUT)
            ]
    for core in onlineCores:
        destinations += [
            [core, 0, i, j, 1] for i in range(onlineAxonNum) for j in range(LUT)
        ]
    
    # cores = onlineCores + offlineCores
    init_hardwareType = hardwareType
    if hardwareType == 'ASIC':
        init_hardwareType = 'v2'
    initFrames = genInitFrame(chipMaxTime, offlineCores, init_hardwareType)
    frames += [Frame.toString(intFrame) for intFrame in initFrames]

    # clearFrames = genClearFrame(chipMaxTime)
    # frames += [Frame.toString(intFrame) for intFrame in clearFrames]

    startFrames = genStartFrame(baseOnlineCores, lateralStars)
    frames += [Frame.toString(intFrame) for intFrame in startFrames]

    dataFrames = genDataFrame(destinations)
    frames += [Frame.toString(intFrame) for intFrame in dataFrames]
 
    syncFrames = genSyncFrame(chipMaxTime)
    frames += [Frame.toString(intFrame) for intFrame in syncFrames]

    endFrames = genEndFrame(baseOnlineCores, lateralStars, onlineModes)
    frames += [Frame.toString(intFrame) for intFrame in endFrames]   

    inputFramePath = os.path.join(frameDir, "input_0.txt")
    with open(inputFramePath,'w') as f:
        f.write("\n".join(frames))

    return

def genTestFrames(offlineCores, onlineCores, offlineNeuronNum, onlineNeuronNum, hardwareType, inputWidth, LCN, frameDir):

    configs = list()

    if hardwareType == 'FPGA':
        offlineFrame1 = offlineNeuronNum* 4
        offlineFrame1_2 = 0
        offlineBeg2 = 0
        offlineFrame2 = 0        
    elif inputWidth == 1:
        offlineFrame1 = (offlineNeuronNum - 1)* 4
        offlineFrame1_2 = 4
        offlineBeg2 = 0
        offlineFrame2 = 0
    else:
        weightNeuronNum = max(0, offlineNeuronNum - 512)
        neuronNeuronNum = min(512,offlineNeuronNum)
        offlineFrame1 = (neuronNeuronNum - 1)* 4
        offlineFrame1_2 = 4
        offlineFrame2 = math.ceil(weightNeuronNum / 5) * 18
        offlineBeg2 = math.ceil(offlineNeuronNum * LCN / 8)
    

    for core in offlineCores:
        configs.append([3,core, 0, 0, offlineFrame1])
        if offlineFrame1_2 > 0:
            configs.append([3, core, 0, 1, offlineFrame1_2])
        if offlineFrame2 > 0:
            configs.append([4, core, 0, offlineBeg2, offlineFrame2])
    
    for core in onlineCores:
        configs.append([3, core, 0, 0, 2 * onlineNeuronNum])
    
    frames = genNeuParaTestFrame(configs)
    frames = [Frame.toString(intFrame) for intFrame in frames]

    framePath = os.path.join(frameDir, "test3.txt")
    with open(framePath, 'w') as f:
        f.write("\n".join(frames))

def genRealOut(offlineCores, onlineCores, offlineNeuronNum, onlineNeuronNum, destChip, frameDir):
    destinations = list()

    offlineAxonMAX = (1 << Hardware.getAttr("AXONBIT", True))
    destCores = ((np.arange(offlineNeuronNum) // offlineAxonMAX) << (Hardware.COREBIT - 2)) + destChip
    destChips = [Hardware.getgPlusCoreId2(destCore, 0) for destCore in destCores]
    destAxons = np.arange(offlineNeuronNum) % offlineAxonMAX

    baseGlobalCoreId = Hardware.getgPlusCoreId2(destChip, 0)

    for core in offlineCores:
        destinations += [[int(dest + core), 0, destAxon, 0] for dest, destAxon in zip(destChips, destAxons)]

    for core in onlineCores:
        destinations += [[int(baseGlobalCoreId + core), 0, i, 0] for i in range(onlineNeuronNum)]

    frames = genDataFrame(destinations, 1)
    frames = [Frame.toString(intFrame) for intFrame in frames]
    framePath = os.path.join(frameDir, "tmpOut.txt")
    with open(framePath, 'w') as f:
        f.write("\n".join(frames))
    return

def runSimulator(frameDir, timeStep):
    configPath = os.path.join(frameDir, "config.txt")
    inputPath = os.path.join(frameDir, "input_0.txt")
    outputPath = os.path.join(frameDir, "output_0.txt")
    simulator = setOnChipNetwork(configPath)
    if timeStep >= 0:
        runOnChipNetwork(simulator, inputPath, outputPath, None)

def timeProfile(func, desc, *args):
    beg = time()
    res = func(*args)
    end = time()
    print(f"[info] {desc} done ({end - beg} s)")
    return res

def merge1(axon, relative, isOffline):
    newDestAxonId = relative * Hardware.getAttr("AXONNUM", isOffline) + axon
    return newDestAxonId, 0

def merge2(axon, relative, isOffline):
    return axon, relative

def myGenConfig(info, mapping, destChip):

    coreNum = 0
    if mapping is not None:
        cores = list(mapping.values())
        cores.sort()
    else:
        cores = list()
        for localPlaceId, placeConfig in info['localPlace'].items():
            cores += list(placeConfig.keys())
        cores += list(info['relay'].keys())
        cores.sort()
    print(cores)

    configs = dict()

    for coreId in cores:
        configs[coreId] = dict()
        configs[coreId]['core'] = None
        configs[coreId]['LUT'] = list()
        configs[coreId]['neuron'] = dict()
    mask = (1 << 5) - 1
    for localPlaceId, placeConfig in info['localPlace'].items():
        for coreId, coreConfig in placeConfig.items():
            isOffline = len(coreConfig['LUT']) == 0
            if mapping is not None:
                coreAddr = int(mapping[coreId])
            else:
                coreAddr = coreId
            coreY = coreAddr & mask
            coreX = (coreAddr >> 5) & mask
            chipY = (coreAddr >> 10) & mask
            chipX = (coreAddr >> 15) & mask
            configs[coreAddr]['core'] = coreConfig['core']
            configs[coreAddr]['LUT'] = coreConfig['LUT']
            # if isOffline:
            #     maxTime[coreAddr] = coreConfig['core'][6] + coreConfig['core'][7] - 1
            # else:
            #     maxTime[coreAddr] = coreConfig['core'][10] + coreConfig['core'][11] - 1
            assert coreConfig['core'] is not None

            if not isOffline:
                lateralStar = coreConfig['core'][8]
                lateralCores = multiCast(coreId, lateralStar, Hardware.COREBIT, mapping)
                star = getStar(lateralCores)
                configs[coreAddr]['core'][8] = (star >> Hardware.COREY) & mask
                configs[coreAddr]['core'][9] = star & mask

            for neuronId, neuronConfig in coreConfig['neuron'].items():
                para = deepcopy(neuronConfig['parameter'])
                destAxonId = para[1]
                relative = para[0]

                if destChip != 0:
                    newDestAxonId, newRe = merge1(destAxonId, relative, isOffline)
                else:
                    newDestAxonId, newRe = merge2(destAxonId, relative, isOffline)

                para[0] = newRe
                para[1] = newDestAxonId

                destCoreId = para[2]
                destStarId = para[3]
                isOutput = False
                if destCoreId < 0:
                    destCoreId = -destCoreId
                    isOutput = True
                if not isOutput:
                    destCores = multiCast(destCoreId, destStarId, Hardware.COREBIT, mapping)
                    if mapping is not None:
                        baseCoreAddr = mapping[destCoreId]
                    else:
                        baseCoreAddr = destCoreId
                    star = getStar(destCores)
                    if mapping is not None:
                        destCoreAddr = mapping[destCoreId]
                    else:
                        destCoreAddr = destCoreId
                else:
                    baseCoreAddr = destCoreId
                    star = 0

                para[2] = (baseCoreAddr >> 5) & mask
                para[3] = baseCoreAddr & mask
                para[4] = (star >> 5) & mask
                para[5] = star & mask
                para[6] = (baseCoreAddr >> 15) & mask
                para[7] = (baseCoreAddr >> 10) & mask
                
                configs[coreAddr]['neuron'][neuronId] = {
                    'parameter': para,
                    'weight' : neuronConfig['weight']
                }

    return configs

def parseArg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=int, default=1,
        help = 'input to the chip'
    )

    parser.add_argument(
        "--rate", type=float, default=1.005,
        help = 'connection rate in the chip'
    )

    parser.add_argument(
        "--config", type=str, default="ASIC_One",
        help = 'config name'
    )

    parser.add_argument(
        "--connection_rate", type=float, default=1.0,
        help = 'axon connection rate'
    )


    parser.add_argument(
        "--num", type=int,
        help = 'core num'
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # configName = "ASIC_None"
    # configName = "ASIC_All"
    # configName = "FPGA_None"
    # configName = "ASIC_One"
    # configName = "FPGA_All"
    # configName = "FPGA_One"
    # configName = "ASIC_All2"
    # configName = "ASIC_All3"
    # configName = "ASIC_LongRun"
    args = parseArg()
    inputData = args.input
    rate = args.rate
    configName = args.config
    assert isinstance(rate, float)
    print(f"[info] config = {configName}, input  = {inputData}, rate = {rate}, connection_rate = {args.connection_rate}")
    # configName = "FPGA_One2"
    frameDir = os.path.join("./output", configName+f"_{inputData}_{int(rate * 100)}_connect_{int(args.connection_rate * 100)}")
    if args.num is not None:
        frameDir+= "_" + str(args.num)
    configPath = os.path.join("./functions/config", configName + ".json")
    if not os.path.exists(frameDir):
        os.makedirs(frameDir)
    
    config = readConfig(configPath)

    hardwareType, offlineCores, onlineCores, SNNEN, inputWidth, outputWidth,\
        offlineNeuronNum, onlineNeuronNum, threshold, \
            timeStep, destChip, connect, LCN = timeProfile(parseConfig, "generate config",config, args.num)

    info = timeProfile(buildCores, "build cores",
        offlineCores, onlineCores, SNNEN, inputWidth, outputWidth,
        offlineNeuronNum, onlineNeuronNum, threshold, timeStep, destChip, rate, LCN, args.connection_rate
    )

    if hardwareType == 'FPGA':
        neuronNum = 4096
    else:
        neuronNum = 512

    
    
    configs = timeProfile(myGenConfig, "generate configs", info, None, destChip)

    offlineCores, onlineCores, chipMaxTime, baseOnlineCores, lateralStars, onlineModes = \
        timeProfile(genConfigFramesRaw, "generate config frames", configs, neuronNum, frameDir)
    timeProfile(genInputFrames, "generate input frames",
        chipMaxTime, baseOnlineCores, lateralStars, onlineModes, offlineCores, 
        onlineCores, connect, inputWidth, frameDir, inputData, LCN, args.connection_rate, hardwareType
    )

    timeProfile(genTestFrames,"generate test frames",
        offlineCores, onlineCores, offlineNeuronNum, onlineNeuronNum, hardwareType, 
        inputWidth, LCN, frameDir
    )

    # timeProfile(genRealOut, "generate simple output frames",
    #     offlineCores, onlineCores, offlineNeuronNum, onlineNeuronNum, destChip, frameDir
    # )
    timeProfile(runSimulator, "run simulator",frameDir, timeStep)





    
        





