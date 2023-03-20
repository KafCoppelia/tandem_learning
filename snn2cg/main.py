from parser import parse
import argparse
import os
import json
import torch

from genConfig import loadMapperJson, genConfigFrames
from parser import parse
from transformer import transform
from genConfig import loadMapInfo, genConfigFrames
from genConfig import genInputFormat, genOutputFormat,genWeightFormat
from runtime import runFullApp, genInputFrames, genOutputFrames

from spike_tensor import SpikeTensor
from softwareNet import checkOneLayer
# from simpleConvert import genSimpleMapper
from simpleConvert import gen16Core as genSimpleMapper

def parseArg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen", action="store_true", default=False,
        help = 'generate config frames'
    )

    parser.add_argument(
        "--dir", default = "./output", type = str,
        help = 'the directory to store output results'
    )

    parser.add_argument(
        "--net", default = "./net/net.pth", type = str,
        help = 'the path of network'
    )

    parser.add_argument(
        "--shape", default = "3,32,32", type = str,
        help = 'shape of inputs'
    )

    parser.add_argument(
        "--time", default = 16, type = int,
        help = 'time steps'
    )

    parser.add_argument(
        '--direct',action="store_true",default=False,
        help='select the modules'
    )

    parser.add_argument(
        '--bit', default=8, type=int,
        help='bitWidth'
    )

    parser.add_argument(
        '--input', action="store_true", default=False, 
        help='genInputFrames'
    )

    parser.add_argument(
        '--mode', default='snn', type=str,
        help='input mode'
    )

    parser.add_argument(
        '--core', default='offline', type=str, 
        choices=['online', 'offline'], 
        help='use online learning cores or offline cores'
    )

    parser.add_argument(
        '--rand', default=None, type=str, 
        help='Dir to store random inputs'
    )

    parser.add_argument(
        '--basePos', default="0000000000", type=str, 
        help='no use'
    )

    parser.add_argument(
        '--hardware', default="v2", type=str, choices=['v2', 'FPGA'],
        help='hardware, [ v2 = ASIC, FPGA = FPGA]'
    )

    args = parser.parse_args()

    return args

def mkdirs(baseDir):
    infoPath = os.path.join(baseDir, "info")
    auxNetPath = os.path.join(baseDir,"auxNet")
    framePath = os.path.join(baseDir, "frames")
    formatPath = os.path.join(baseDir, "formats")
    if not os.path.exists(infoPath):
        os.makedirs(infoPath)
    if not os.path.exists(auxNetPath):
        os.makedirs(auxNetPath)
    if not os.path.exists(framePath):
        os.makedirs(framePath)
    if not os.path.exists(formatPath):
        os.makedirs(formatPath)

def genData(dataShape, timeStep, inputMode, baseDir):
    data = torch.ones(timeStep, *dataShape)
    # data[:,:500] = 0
    # data = torch.rand(timeStep, *dataShape)
    data = (data > 0.5).float()
    torch.save(data, os.path.join(baseDir, "input_0.pth"))
    if inputMode == 'snn':
        data = SpikeTensor(data, timeStep, 1)
    else:
        data = data * 64
        data.scale = 1
    return [[data]]

def getMultiRandomData(
     timeStep, inputMode, multiDataDir
):
    dataList = list()
    assert os.path.exists(multiDataDir)
    files = os.listdir(multiDataDir)
    for fileName in files:
        if fileName == '.' or fileName == '..':
            continue
        fullFileName = os.path.join(multiDataDir, fileName)
        data = torch.load(fullFileName)
        if inputMode == 'snn':
            data = SpikeTensor(data, timeStep, 1)
        else:
            data.scale = 1
        dataList.append([data])

    return dataList
            

def loadNet(netPath):
    net = torch.load(netPath, map_location=torch.device("cpu"))
    # net = net["model_state_dict"]
    # net.eval()
    return net

def storeNetInfo(onChipNet, infoDir):
    outputDict, shapeDict, scaleDict, inputList = onChipNet.getIoInfo()
    outDictPath = os.path.join(infoDir, "outDict.json")
    shapePath = os.path.join(infoDir, "shape.json")
    scalePath = os.path.join(infoDir, "scale.json")
    inputPath = os.path.join(infoDir, "inputList.json")
    with open(outDictPath, 'w') as f:
        json.dump(outputDict, f, indent = 4)
    with open(shapePath, 'w') as f:
        json.dump(shapeDict, f, indent = 4)
    with open(scalePath, 'w') as f:
        json.dump(scaleDict, f, indent = 4)
    with open(inputPath, 'w') as f:
        json.dump(inputList, f, indent = 4)
    return 

def storeMappingInfo(hardwareNetwork, softwareNetwork, weightMappings, infoDir, timeStep):
    info = hardwareNetwork.store(args.time)
    inputMapping, outputMapping = softwareNetwork.store(hardwareNetwork.computeGroup)

    netInfoPath = os.path.join(infoDir, "info.json")
    inMappingPath = os.path.join(infoDir, "inputMapping.json")
    outMappingPath = os.path.join(infoDir, "outputMapping.json")
    weightMappingPath = os.path.join(infoDir, "weightMapping.json")
    
    with open(netInfoPath,'w') as f:
        json.dump(info, f, indent=4)
    with open(inMappingPath, 'w') as f:
        json.dump(inputMapping, f, indent = 4)
    with open(outMappingPath,'w') as f:
        json.dump(outputMapping, f, indent = 4)
    with open(weightMappingPath, 'w') as f:
        json.dump(weightMappings, f, indent = 4)

def loadFormatInputInfo(infoDir):
    inMappingPath = os.path.join(infoDir, "inputMapping.json")
    inputPath = os.path.join(infoDir, "inputList.json")
    with open(inMappingPath, 'r') as f:
        inputMapping = json.load(f)
    with open(inputPath, 'r') as f:
        inputList = json.load(f)
    intInputMapping = dict()
    for name, oneInMapping in inputMapping.items():
        newInMapping = [
            list() for i in range(len(oneInMapping))
        ]
        for i, mapping in enumerate(oneInMapping):
            for positions in mapping:
                newInMapping[i].append([int(position) for position in positions])
        intInputMapping[name] = newInMapping
    return inputList, intInputMapping

def loadFormatOutputInfo(infoDir):
    outMappingPath = os.path.join(infoDir, "outputMapping.json")
    with open(outMappingPath,'r') as f:
        outputMapping = json.load(f)
    intOutputMapping = dict()
    for name, infos in outputMapping.items():
        intOutputMapping[name] = {
            'axons': [int(pos) for pos in infos['axons']],
            'neurons': [[int(neuron) for neuron in neurons] for neurons in infos['neurons']],
            'bitWidth': int(infos['bitWidth']),
            'LCN': int(infos['LCN'])
        }
    return intOutputMapping

def loadAuxNet(auxNetDir):
    preNetPath = os.path.join(auxNetDir, "preNet.pth")
    postNetPath = os.path.join(auxNetDir, "postNet.pth")
    return loadNet(preNetPath), loadNet(postNetPath)

def tmpSimpleMapper(infoDir, basePos):
    netInfoPath = os.path.join(infoDir, "info.json")
    with open(netInfoPath, 'r') as f:
        config = json.load(f)
    mapper = genSimpleMapper(config, basePos)
    mapperPath = os.path.join(infoDir, "mapping_result.txt")
    with open(mapperPath,'w') as f:
        json.dump(mapper, f, indent = 4) 

if __name__ == "__main__":

    args = parseArg()
    dataShape = [int(s) for s in args.shape.split(",")]
    mkdirs(args.dir)
    infoDir = os.path.join(args.dir, "info")
    frameDir = os.path.join(args.dir, "frames")
    formatDir = os.path.join(args.dir, "formats")
    auxNetDir = os.path.join(args.dir, "auxNet")

    net = loadNet(args.net)
    if args.rand is None:
        inputs = genData(dataShape, args.time, args.mode, args.dir)
    else:
        inputs = getMultiRandomData(args.time, args.mode, args.rand)
    

    if args.input:
        preNet, postNet = loadAuxNet(auxNetDir)
        genInputFrames(args.dir, args.mode, preNet, args.time)
        genOutputFrames(preNet, postNet, args.dir, net, args.time, args.mode, args.core)
    elif args.gen:
        # tmpSimpleMapper(infoDir, int(args.basePos,2))
        info, mapper, weightMapping = loadMapInfo(infoDir)

        if args.hardware == 'v2':
            para3MaxNeuron = 512
        elif args.hardware == 'FPGA':
            para3MaxNeuron = 4096
        else:
            assert False
        offlineCores, onlineCores, chipMaxTime, baseOnlineCores, lateralStars, onlineModes \
             = genConfigFrames(info, mapper, para3MaxNeuron, frameDir)
        
        inputList, inputMapping = loadFormatInputInfo(infoDir)
        outputMapping = loadFormatOutputInfo(infoDir)

        # store input, output and weight mapping info
        genInputFormat(
            offlineCores, args.hardware, inputMapping, formatDir, mapper, 
            inputList, args.time, chipMaxTime, 
            baseOnlineCores, lateralStars, onlineModes
        )
        genOutputFormat(outputMapping, formatDir, mapper)
        genWeightFormat(weightMapping, formatDir, mapper)
        
        preNet, postNet = loadAuxNet(auxNetDir)
        runFullApp(preNet, postNet, args.dir, inputs, net, args.time, args.core)
    else:
        onChipNet = parse(
            net, args.bit, args.time, args.direct, args.dir, 
            args.core, *(inputs[0])
        )
        from hwConfig import Hardware
        Hardware.setNoCLevel([1], [1], True)
        # Hardware.setNoCLevel([2,2], [2,2], True)
        # Hardware.setNoCLevel([1], [1], False)
        hardwareNetwork, softwareNetwork, weightMappings = transform(
            onChipNet, args.bit, 1, args.time, args.hardware
        )
        # softwareNetwork.print("conv4",3* 36 + 2 * 6 + 5)
        # inputSizes = onChipNet.getShapes()
        # checkOneLayer(
        #     net, onChipNet.opOutputs, 
        #     onChipNet.tensorSizes,inputSizes, softwareNetwork, 
        #     hardwareNetwork, args.time
        # )
        storeNetInfo(onChipNet, infoDir)
        storeMappingInfo(hardwareNetwork, softwareNetwork, weightMappings, infoDir, args.time)
        