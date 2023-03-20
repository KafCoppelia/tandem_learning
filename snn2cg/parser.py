from .operators import *
from copy import deepcopy
from .transformer import OnChipNetwork
from .spike_dag import *
# from SNN.spike_dag import *
from .spike_tensor import SpikeTensor
# from SNN.spike_tensor import SpikeTensor
import os

'''--------------------------------------------------------------------------'''
'''          helper class to record a layer if is onchip or offchip          '''
'''--------------------------------------------------------------------------'''
class Status:
    INVALID = 1
    PREPROCESS = 2
    INPUT = 4
    ONCHIP = 8
    OUTPUT = 16
    POSTPROCESS = 32
    @staticmethod
    def addStatus(status, newStatus):
        return status | newStatus
    @staticmethod
    def isInput(status):
        return (status & Status.INPUT) != 0
    @staticmethod
    def isOutput(status):
        return (status & Status.OUTPUT) != 0
    @staticmethod
    def isOnChip(status):
        return (status & (Status.INPUT | Status.ONCHIP | Status.OUTPUT)) != 0
    @staticmethod
    def isPre(status):
        return (status & Status.PREPROCESS) != 0
    @staticmethod
    def isPost(status):
        return (status & Status.POSTPROCESS) != 0
    @staticmethod
    def isInvalid(status):
        return (status & Status.INVALID) != 0

class NetworkInfo:
    def __init__(self, net):
        self.opStatus = OrderedDict()
        self.deleteLayers = dict()
        self.input2Ops = dict()
        self.output2Ops = dict()
        self.opInputs = dict()
        self.opOutputs = dict()
        self.netInputs = deepcopy(net.inputs_nodes)
        self.netOutputs = deepcopy(net.outputs_nodes)
        self.outputDict = dict()
        self.inputList = list()
        for opName, op in net.ops.items():
            self.opStatus[opName] = Status.INVALID
            self.opInputs[opName] = deepcopy(op['in_nodes'])
            self.opOutputs[opName] = deepcopy(op['out_nodes'])
            for tensorName in self.opInputs[opName]:
                if tensorName not in self.input2Ops:
                    self.input2Ops[tensorName] = list()
                self.input2Ops[tensorName].append(opName)
            for tensorName in self.opOutputs[opName]:
                if tensorName not in self.output2Ops:
                    self.output2Ops[tensorName] = list()
                self.output2Ops[tensorName].append(opName)
            if op['op'].__class__.__name__.endswith("SpikeConv2d"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeAvgPool2d"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeMaxPool2d"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeLinear"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeConvTranspose2d"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeAdd"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeLinearSTDP"):
                self.deleteLayers[opName] = False
            elif op['op'].__class__.__name__.endswith("SpikeReLU"):
                self.deleteLayers[opName] = True
            elif op['op'].__class__.__name__.endswith("DAGViewOp"):
                self.deleteLayers[opName] = True
            elif op['op'].__class__.__name__.endswith("ConcatOp"):
                self.deleteLayers[opName] = True
            else:
                assert False, f"the layer kind {op['op'].__class__.__name__} is not supported now"
    
    def setInputStatus(self, inputOps):

        if len(inputOps) > 0:
            for inputOp in inputOps:
                self.opStatus[inputOp] = Status.INPUT
        else:
            networkInputs  = set(self.netInputs)
            networkOutputs = set(self.netOutputs)
            for opName in self.opStatus.keys():
                inputs = set(self.opInputs[opName])
                isInput = len(networkInputs & inputs) != 0
                if isInput:
                    self.opStatus[opName] = Status.INPUT
    
    def setPreStatus(self):
        opNames = list(self.opStatus.keys())
        opNames.reverse()
        networkOutputs = set(self.netOutputs)
        for opName in opNames:
            opOutputs = self.opOutputs[opName]
            assert len(opOutputs) == 1 #now each network layer can only have one output tensor
            for opOutput in opOutputs:
                if opOutput in networkOutputs:
                    break
                isPre = False
                for outOpName in self.input2Ops[opOutput]:
                    status = self.opStatus[outOpName]
                    if Status.isInput(status) or Status.isPre(status):
                        self.opStatus[opName] = Status.PREPROCESS
                        isPre = True
                        break
                if isPre:
                    break
        return
    
    def setOutputStatus(self, outputOps):
        networkOutputs = set(self.netOutputs)
        if len(outputOps) > 0:
            for outputOp in outputOps:
                assert outputOp in self.opStatus
                status = self.opStatus[outputOp]
                assert Status.isInvalid(status) or Status.isInput(status)
                if Status.isInvalid(status):
                    self.opStatus[outputOp] = Status.OUTPUT
                else:
                    self.opStatus[outputOp] = Status.addStatus(status, Status.OUTPUT)
        else:
            for opName, status in self.opStatus.items():
                if len(set(self.opOutputs[opName]) & networkOutputs) > 0:
                    if Status.isInput(status):
                        self.opStatus[opName] = Status.addStatus(status, Status.OUTPUT)
                    else:
                        self.opStatus[opName] = Status.OUTPUT
            '''
            for opName, status in opStatus.items():
                if Status.isPre(status):
                    continue
                inputs = opInputs[opName]
                isOnChip = True
                for opInput in inputs:
                    inOpName = network.output2Module[opInput]
                    if network.ops[inOpName].status == Status.INVALID:
                        inOnChip = False
                        break
                if isOnChip:
                    for opInput in opInputs:
                        inOpName = network.output2Module[opInput]
                        inOpStatus = network.ops[inOpName].status
                        if inOpStatus | Status.PREPROCESS:
                            continue
                        if (inOpStatus | Status.INPUT) == 0:
                            network.ops[inOpName].status = Status.ONCHIP
                        elif inOpStatus | Status.OUTPUT:
                            network.ops[inOpName].status = Status.INPUT
                    if opStatus | Status.INPUT:
                        network.ops[opName].status |= Status.OUTPUT
                    else:
                        network.ops[opName].status = Status.OUTPUT
            '''

    def setOnChipStatus(self):
        networkInputs = set(self.netInputs)
        networkOutputs = set(self.netOutputs)
        for opName,status in self.opStatus.items():
            if Status.isInput(status) or Status.isPre(status) or Status.isOutput(status):
                continue
            isOnChip = True
            for opInput in self.opInputs[opName]:
                if opInput in networkInputs:
                    isOnChip = False
                    break
                inOpName = self.output2Ops[opInput][0]
                inOpStatus = self.opStatus[inOpName]
                if Status.isInvalid(inOpStatus): # may be some bugs
                    isOnChip = False
                    break
            if isOnChip:
                self.opStatus[opName] = Status.ONCHIP
        
        opNames = list(self.opStatus.keys())
        opNames.reverse()
        for opName in opNames:
            status = self.opStatus[opName]
            if Status.isPre(status) or Status.isInput(status) \
                or Status.isOutput(status) or Status.isPost(status):
                continue
            outputs = self.opOutputs[opName]
            isOnChip = False
            for opOutput in outputs:
                if opOutput in networkOutputs:
                    if Status.isOutput(status):
                        isOnChip = True
                    break
                outOpNames = self.input2Ops[opOutput]
                for outOpName in outOpNames:
                    outOpStatus = self.opStatus[outOpName]
                    if Status.isOnChip(outOpStatus) or Status.isInput(outOpStatus) \
                        or Status.isOutput(outOpStatus):
                        isOnChip = True
                        break
            if not isOnChip:
                self.opStatus[opName] = Status.POSTPROCESS

    def setPostStatus(self):
        for opName, status in self.opStatus.items():
            if Status.isOnChip(status) or Status.isPre(status):
                continue
            self.opStatus[opName] = Status.POSTPROCESS
        return

    def setStatus(self, inputOps, outputOps):
        self.setInputStatus(inputOps)
        self.setPreStatus()
        self.setOutputStatus(outputOps)
        self.setOnChipStatus()
        self.setPostStatus()

    def redirect(self):
        redirect = dict()
        for opName in self.opStatus.keys():
            if self.deleteLayers[opName]:
                re = list()
                for opInput in self.opInputs[opName]:
                    if opInput in redirect:
                        re += redirect[opInput]
                    else:
                        re.append(opInput)
                for opOutput in self.opOutputs[opName]:
                    redirect[opOutput] = re
        
        for opName, status in self.opStatus.items():
            if Status.isOutput(status):
                re = None
                for opOutput in self.opOutputs[opName]:
                    if opOutput in redirect:
                        re = redirect[opOutput]
                    else:
                        re = [opOutput]
                    self.outputDict[opOutput] = re

        
        # for outputName, realOutputs in self.outputDict.items():
        #     self.scaleFactors[outputName] = net.nodes[outputName].scale_factor.mean().item()
        
        for opName, status in self.opStatus.items():
            if Status.isInvalid(status) or \
                Status.isPre(status) or Status.isPost(status):
                continue
            if self.deleteLayers[opName]:
                if Status.isInput(status):
                    if Status.isOutput(status):
                        assert False, \
                            f"{op_name}: relu, view, concat layer cannot be both input layer and output layer\n"
                    for opOutput in self.opOutputs[opName]:
                        if opOutput not in self.input2Ops:
                            continue
                        inOpNames = self.input2Ops[opOutput]
                        for inOpName in inOpNames:
                            inOpStatus = self.opStatus[inOpName]
                            if Status.isOutput(inOpStatus):
                                self.opStatus[inOpName] = Status.addStatus(inOpStatus, Status.INPUT) 
                            else:
                                self.opStatus[inOpName] = Status.INPUT
                if Status.isInput(status):
                    self.opStatus[opName] = Status.PREPROCESS
                else:
                    self.opStatus[opName] = Status.INVALID
                continue
            newOpInputs = list()
            for opInput in self.opInputs[opName]:
                if opInput in redirect:
                    newOpInputs += redirect[opInput]
                else:
                    newOpInputs.append(opInput)
            self.opInputs[opName] = newOpInputs
        
        newInput2Ops = dict()
        for opName, status in self.opStatus.items():
            if Status.isInvalid(status) or Status.isPost(status) or Status.isPre(status):
                continue
            for opInput in self.opInputs[opName]:
                if opInput not in newInput2Ops:
                    newInput2Ops[opInput] = list()
                newInput2Ops[opInput].append(opName)
        self.input2Ops = newInput2Ops


        for outName, realOutputs in self.outputDict.items():
            if len(realOutputs) == 1 and realOutputs[0] == outName:
                continue
            for realOutput in realOutputs:
                outOpName = self.output2Ops[realOutput][0]
                outStatus = self.opStatus[outOpName]
                assert outOpName in self.opStatus
                if Status.isInput(outStatus):
                    self.opStatus[outOpName] = Status.addStatus(outStatus, Status.OUTPUT)
                else:
                    self.opStatus[outOpName] = Status.OUTPUT

        
        for opName, status in self.opStatus.items():
            if not Status.isInput(status):
                continue
            self.inputList += self.opInputs[opName]

        return

    def getAuxNet(self, net, baseDir):
        preNet = SpikeDAGModule()
        postNet = SpikeDAGModule() 
        for opName, status in self.opStatus.items():
            if Status.isPre(status):
                preNet.add_op(
                    opName, 
                    net.ops[opName]['op'], 
                    net.ops[opName]['in_nodes'], 
                    net.ops[opName]['out_nodes']
                )
            elif Status.isPost(status):
                postNet.add_op(
                    opName, 
                    net.ops[opName]['op'], 
                    net.ops[opName]['in_nodes'], 
                    net.ops[opName]['out_nodes']
                )
        preInput = list()
        preOutput = list()
        preTensors = set()

        postInput = list()
        postOutput = list()
        postTensors = set()               

        for opName, status in self.opStatus.items():
            if Status.isPre(status):
                for outputTensor in net.ops[opName]['out_nodes']:
                    preTensors.add(outputTensor)
            elif Status.isPost(status):
                for outputTensor in net.ops[opName]['out_nodes']:
                    postTensors.add(outputTensor)
        
        for opName, status in self.opStatus.items():
            if Status.isPre(status):
                for inputTensor in net.ops[opName]['in_nodes']:
                    if inputTensor not in preTensors:
                        if inputTensor not in preNet.inputs_nodes:
                            preNet.inputs_nodes.append(inputTensor)
            if Status.isPost(status):
                for inputTensor in net.ops[opName]['in_nodes']:
                    if inputTensor not in postTensors:
                        if inputTensor not in postNet.inputs_nodes:
                            postNet.inputs_nodes.append(inputTensor)
        
        for opName, status in self.opStatus.items():
            if Status.isPre(status):
                for outputTensor in net.ops[opName]['out_nodes']:
                    if outputTensor in self.inputList or \
                        outputTensor in postNet.inputs_nodes or \
                        outputTensor in net.outputs_nodes:
                        if outputTensor not in preNet.outputs_nodes:
                            preNet.outputs_nodes.append(outputTensor)
        
        for tensorName in net.inputs_nodes:
            if tensorName in self.inputList or \
                tensorName in postNet.inputs_nodes:
                    preNet.outputs_nodes.append(tensorName)
        
        for outputTensor in net.outputs_nodes:
            if outputTensor in postTensors:
                postNet.outputs_nodes.append(outputTensor)
        
        fileDir = os.path.join(baseDir,"auxNet")
        torch.save(preNet, os.path.join(fileDir, "preNet.pth"))
        torch.save(postNet,os.path.join(fileDir, "postNet.pth"))
        return
 
    def buildOnChipNet(self, net, weightBits, coreType, inputs):
        
        onChipNet = OnChipNetwork(coreType)
        onChipNet.inputList = deepcopy(self.inputList)
        onChipNet.outputDict = deepcopy(self.outputDict)
        onChipNet.input2Ops = deepcopy(self.input2Ops)
        for tensorName, opNames in self.output2Ops.items():
            names = list()
            for opName in opNames:
                if Status.isOnChip(self.opStatus[opName]):
                    names.append(opName)
            if len(names) > 0:
                onChipNet.output2Ops[tensorName] = names
        
        for opName, status in self.opStatus.items():
            if Status.isOnChip(status):
                onChipNet.opInputs[opName] = deepcopy(self.opInputs[opName])
                onChipNet.opOutputs[opName] = deepcopy(self.opOutputs[opName])
        
        originU = dict()
        for opName, status in self.opStatus.items():
            op = net.ops[opName]
            if op['op'].__class__.__name__.endswith('SpikeLinearSTDP'):
                originU[opName] = deepcopy(op['op'].learn)
                op['op'].switch_learn(False)

        net(inputs)

        for opName, status in self.opStatus.items():
            op = net.ops[opName]
            if op['op'].__class__.__name__.endswith('SpikeLinearSTDP'):
                op['op'].switch_learn(originU[opName])

        for opName, status in self.opStatus.items():
            if not Status.isOnChip(status):
                continue
            op = net.ops[opName]
            if op['op'].__class__.__name__.endswith("SpikeConv2d"):
                config = buildConv2d(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith("SpikeAvgPool2d"):
                config = buildAvgPool2d(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith("SpikeLinear"):
                config = buildFC(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith("SpikeConvTranspose2d"):
                config = buildTransConv2d(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith('SpikeAdd'):
                config = buildAdd(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith('SpikeMaxPool2d'):
                config = buildMaxPool2d(net, op, opName, weightBits, coreType)
            elif op['op'].__class__.__name__.endswith('SpikeLinearSTDP'):
                config = buildSTDP_FC(net, op, opName, weightBits, coreType)            
            # elif op['op'].__class__.__name__.endswith("SpikeReLU"):
            #     config = buildRelu(net, op, opName, weightBits)
            # elif op['op'].__class__.__name__.endswith("DAGViewOp"):
            #     config = buildView(net, op, opName, weightBits)
            # elif op['op'].__class__.__name__.endswith("ConcatOp"):
            #     config = buildConcat(net, op, opName, weightBits)
            else:
                assert False, f"the layer kind {op['op'].__class__.__name__} is not supported now"
            onChipNet.ops[opName] = config
        
        for tensorName in self.inputList:
            onChipNet.tensorSizes[tensorName] = net.nodes[tensorName].size()[1:]
        
        for opName, status in self.opStatus.items():
            if Status.isOnChip(status):
                for tensorName in self.opOutputs[opName]:
                    onChipNet.tensorSizes[tensorName] = net.nodes[tensorName].size()[1:]
        for name in self.outputDict.keys():
            onChipNet.outputShapes[name] = net.nodes[name].size()[1:]
        for name in self.outputDict.keys():
            if hasattr(net.nodes[name], "scale_factor"):
                onChipNet.outputScale[name] = net.nodes[name].scale_factor
            else:
                #TODO: if a layer doesn't have attribute scale_factor, 
                #      does it really mean that the scale_factor is 1 ? 
                onChipNet.outputScale[name] = 1
        return onChipNet

'''--------------------------------------------------------------------------'''
''' set the input layer and output layer by users                            '''
''' and the tool chain put the layers between (and include) them on the chip '''
'''--------------------------------------------------------------------------'''
def getDirect(hint, layer_num):
    ids = []
    while len(ids) == 0:
        ids = input(hint + "\n").strip()
        ids = ids.split(" ")
        valid = True
        for one_id in ids:
            if not one_id.isdigit() or int(one_id) >= layer_num:
                valid = False
                break
        if not valid:
            ids = []
    return ids

def parseDirect(net, needDirect=False):
    inputOps = list()
    outputOps = list()
    layers = dict()
    layerNum = len(net.ops)

    for layerId, (opName, op) in enumerate(net.ops.items()):
        layers[layerId] = opName
        print(layerId, opName)
        
        # print(f"{layerId} ({opName}) : {op['op']}")
    if needDirect:
        inputIds  = getDirect("Please input the on-chip input layers ,like 1 2 3:", layerNum)
        outputIds = getDirect("Plase input the on-chip output_layers, like 1 2 3:", layerNum)
        # inputIds = ['3']
        # outputIds = ['11']
        inputOps  = [layers[int(_)] for _ in inputIds]
        outputOps = [layers[int(_)] for _ in outputIds]
    return inputOps, outputOps


'''--------------------------------------------------------------------------'''
'''                    parse weight for network layers                       '''
'''                    now the weightbase can only be 1                      '''
'''--------------------------------------------------------------------------'''
def parseWeight(bitWidth, weightBase, weights):
    if weights is None:
        return None
    assert weightBase == 1, weightBase
    tmpWeight = deepcopy(weights)
    if weightBase != 1:
        new_weights = (tmpWeight / weightBase).detach().numpy()
    else:
        new_weights = tmpWeight.detach().numpy()
    new_weights += (1 << bitWidth)
    shape = new_weights.shape
    baseVector = (2 ** (np.arange(bitWidth)))
    baseWeight = np.zeros(shape)
    baseWeight = np.expand_dims(baseWeight,-1)
    baseWeight = baseWeight.repeat(bitWidth,-1).astype(int)
    for i in range(bitWidth):
        baseWeight[...,i] = baseVector[i]
    weightsParsed  = np.expand_dims(new_weights, -1).repeat(bitWidth,-1).astype(int)
    weightsParsed = (baseWeight & weightsParsed) != 0
    return weightsParsed

'''--------------------------------------------------------------------------'''
'''                              offline layers                              '''
'''--------------------------------------------------------------------------'''

def buildConv2d(net, op, name, weightBits, coreType):   
    layer = Conv2dInfo()
    module = op['op']
    inputs  = tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    layer.threshold = module.Vthr.detach().cpu()
    layer.name = name
    layer.memPotential = 0
    
    if module.mode == 'quantized':
        layer.mode = 'ann'
    elif module.mode == 'snn':
        layer.mode = 'snn'
    else:
        assert False, f"not support mode {module.mode}. The choices are ['quantized','snn']"

    if layer.mode == 'ann':
        layer.bitTrunc = module.shift_bit + 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False
    
    layer.resetMode = module.reset_mode
    # layer.inputNames = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])
    layer.inputSize  = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.outputSize = outputs[0].size()[1:]
    layer.kernelSize = module.weight.shape
    layer.scale = module.out_scales
    layer.bitWidth = weightBits

    onChip = True
    if module.quant_base is None:
        onChip = False

    if onChip:
        layer.weightBase= module.quant_base.item()
        layer.weight = parseWeight(layer.bitWidth, layer.weightBase, module.weight)
    
    bias = module.bias
    if (bias is not None) and (bias.abs().sum() != 0) and onChip:
        layer.bias = bias
    else:
        layer.bias = np.zeros(layer.outputSize[0])

    layer.stride = module.stride
    layer.padding = module.padding
    layer.dilation = module.dilation
    layer.groups = module.groups
    return layer

def buildFC(net, op, name, weightBits, coreType):
    inputs  = tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module = op['op']
    layer = FcInfo()
    layer.name = name
    layer.threshold = module.Vthr.detach().cpu()
    layer.memPotential = 0
    layer.resetMode = module.reset_mode

    if module.mode == 'quantized':
        layer.mode = 'ann'
    elif module.mode == 'snn':
        layer.mode = 'snn'
    else:
        assert False, f"not support mode {module.mode}. The choices are ['quantized','snn']"

    if layer.mode == 'ann':
        layer.bitTrunc = module.shift_bit + 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False
    # layer.inputNames = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])
    layer.inputSize  = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.outputSize = outputs[0].size()[1:]
    layer.kernelSize = module.weight.size()
    layer.bitWidth = weightBits
    layer.scale = module.out_scales

    onChip = True
    if module.quant_base is None:
        onChip = False
    if onChip:
        layer.weightBase= module.quant_base.item()
        layer.weight = parseWeight(layer.bitWidth, layer.weightBase, module.weight)
    
    bias = module.bias
    if (bias is not None) and (bias.abs().sum() != 0) and onChip:
        layer.bias = bias
    else:
        layer.bias = np.zeros(layer.outputSize[0])
    return layer

def buildTransConv2d(net, op, name, weightBits, coreType):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module= op['op']
    layer = TransConv2dInfo()
    layer.name = name
    layer.threshold = module.Vthr.detach().cpu()
    layer.memPotential = 0
    layer.resetMode = module.reset_mode

    if module.mode == 'quantized':
        layer.mode = 'ann'
    elif module.mode == 'snn':
        layer.mode = 'snn'
    else:
        assert False, f"not support mode {module.mode}. The choices are ['quantized','snn']"

    if layer.mode == 'ann':
        layer.bitTrunc = module.shift_bit + 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False

    layer.scale = module.out_scales
    # layer.inputNames = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])
    layer.outputSize = outputs[0].size()[1:]
    layer.inputSize = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.kernelSize = module.weight.shape
    layer.bitWidth = weightBits
    
    if layer.mode == 'online':
        layer.isOffline = False
        # layer.onlineLut = module.STDP_LUT
        # layer.lateral = module.lateral
        # layer.weightDecay = module.weightDecay
        # layer.upperWeight = module.upper_weight
        # layer.lowerWeight = module.lower_weight

    onChip = True
    if module.quant_base is None:
        onChip = False

    if onChip:
        layer.weightBase= module.quant_base.item()
        layer.weight = parseWeight(layer.bitWidth, layer.weightBase, module.weight)
    
    bias = module.bias
    if (bias is not None) and (bias.abs().sum() != 0) and onChip:
        layer.bias = bias
    else:
        layer.bias = np.zeros(layer.outputSize[0])

    layer.stride = module.stride
    layer.padding = module.padding
    # conv2d_layer['dilation'] = module.dilation
    layer.groups = module.groups
    layer.outputPadding = module.output_padding
    return layer  

def buildAvgPool2d(net, op, name, weightBits, coreType):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(et.nodes[_] for _ in op['out_nodes'])
    module = op['op'] 
    layer = Avgpool2dInfo()
    layer.name = name
    layer.resetMode = module.reset_mode

    if module.mode == 'quantized':
        layer.mode = 'ann'
    elif module.mode == 'snn':
        layer.mode = 'snn'
    else:
        assert False, f"not support mode {module.mode}. The choices are ['quantized','snn']"

    if layer.mode == 'ann':
        layer.bitTrunc = 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False

    layer.inputNames = deepcopy(op['in_nodes'])
    layer.outputNames = deepcopy(op['out_nodes'])
    layer.outputSize = outputs[0].size()[1:]
    layer.inputSize = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.kernelSize = module.kernel_size
    layer.scale = module.out_scales
    if isinstance(layer.kernelSize, int):
        layer.kernelSize = [module.kernel_size, module.kernel_size]
    layer.stride = module.stride
    if module.stride is None:
        layer.stride = module.kernel_size
    elif isinstance(module.stride, int):
        layer.stride = [module.stride, module.stride]
    layer.padding = module.padding
    if isinstance(module.padding, int):
        layer.padding = [module.padding, module.padding]
    return layer

def buildMaxPool2d(net, op, name, weightBits, coreType):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module = op['op'] 
    layer = Maxpool2dInfo()
    layer.name = name
    layer.resetMode = 'non-reset'

    if module.mode == 'quantized' or module.mode == 'ann':
        layer.mode = 'ann'
    else:
        layer.mode = 'snn'

    if layer.mode == 'ann':
        layer.bitTrunc = 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False

    layer.inputNames = deepcopy(op['in_nodes'])
    layer.outputNames = deepcopy(op['out_nodes'])
    layer.outputSize = outputs[0].size()[1:]
    layer.inputSize = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.kernelSize = module.kernel_size
    # layer.scale = module.out_scales
    layer.scale = 1
    if isinstance(layer.kernelSize, int):
        layer.kernelSize = [module.kernel_size, module.kernel_size]
    layer.pool = True
    layer.stride = module.stride
    if module.stride is None:
        layer.stride = module.kernel_size
    elif isinstance(layer.stride, int):
        layer.stride = [module.stride, module.stride]
    layer.padding = module.padding
    if isinstance(module.padding, int):
        layer.padding = [module.padding, module.padding]
    return layer

def buildAdd(net, op, name, weightBits, coreType):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    layer = AddInfo()
    layer.outputSize = outputs[0].size()[1:]
    layer.inputSize = [inputs[i].size()[1:] for i in range(len(inputs))]
    module = op['op']
    layer.name = name
    
    layer.bitWidth = weightBits

    if layer.mode == 'online':
        layer.isOffline = False
        # layer.onlineLut = module.STDP_LUT
        # layer.lateral = module.lateral
        # layer.weightDecay = module.weightDecay
        # layer.upperWeight = module.upper_weight
        # layer.lowerWeight = module.lower_weight

    onChip = True
    # if module.quant_base is None:
    #     onChip = False

    if onChip:
        # layer.weightBase = module.quant_base.item()
        layer.weight = parseWeight(layer.bitWidth, layer.weightBase, module.weight)

    # bias = module.bias
    # if onChip and (bias is not None) and (bias.abs().sum() != 0) :
    #     layer.bias = bias
    # else:
    #     layer.bias = np.zeros(layer.outputSize[0])
    layer.bias = np.zeros(layer.outputSize[0])
    layer.threshold = module.Vthr.detach().cpu()
    layer.memPotential = 0

    if module.mode == 'quantized':
        layer.mode = 'ann'
    elif module.mode == 'snn':
        layer.mode = 'snn'
    else:
        assert False, f"not support mode {module.mode}. The choices are ['quantized','snn']"

    if layer.mode == 'ann':
        layer.bitTrunc = module.shift_bit + 8
    else:
        layer.bitTrunc = 1

    if coreType == 'online':
        assert layer.mode == 'snn'
        layer.isOffline = False

    layer.resetMode = module.reset_mode
    # layer.inputNames = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])

    return layer


'''--------------------------------------------------------------------------'''
'''                              online layers                               '''
'''--------------------------------------------------------------------------'''

def buildSTDP_FC(net, op, name, weightBits, coreType):
    inputs  = tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module = op['op']
    layer = STDPFcInfo()

    layer.name = name
    layer.isOffline = False
    layer.resetMode = "zero"
    layer.mode = "snn"

    LUT = module.LUT.detach().cpu().int().numpy()
    layer.LUT = LUT
        
    if hasattr(module.Vthr,"detach"):
        layer.threshold = int(module.Vthr.detach().cpu())
        # print(layer.threshold)
    else:
        layer.threshold = module.Vthr

    layer.memPotential = 0
    layer.resetMem = int(module.reset_mem.detach().cpu())
    # print(layer.resetMem)
    layer.lowerMem = int(module.lower_mem.detach().cpu())
    # print(layer.lowerMem)
    layer.prohibation = int(module.prohibation.detach().cpu())
    # print(layer.prohibation)
    layer.lowerWeight = int(module.lower_weight.detach().cpu())
    # print(layer.lowerWeight)
    layer.upperWeight = int(module.upper_weight.detach().cpu())
    # print(layer.upperWeight)
    layer.weightDecay = int(module.weight_decay.detach().cpu())
    # print(layer.weightDecay)

    layer.learnMode = bool(module.learn)

    # if hasattr(module, "learn"):
    #     if module.learn_mode:
    #         layer.learnMode = True
    #     else:
    #         layer.learnMode = False
    # else:
    #     layer.learnMode = True

    assert coreType == 'online'
    layer.inputSize  = [inputs[i].size()[1:] for i in range(len(inputs))]
    layer.outputSize = outputs[0].size()[2:]
    layer.kernelSize = module.weight.size()
    layer.bitWidth = weightBits

    #TODO: whether online learning layers have out_scales or not 
    if hasattr(module, "out_scales"):
        layer.scale = module.out_scales
    else:
        layer.scale = 1

    onChip = True
    # if module.quant_base is None:
    #     onChip = False

    if onChip:
        # layer.weightBase= module.quant_base.item()
        layer.weightBase = 1
        layer.weight = parseWeight(layer.bitWidth, layer.weightBase, module.weight)
    
    bias = module.bias
    if (bias is not None) and (bias.abs().sum() != 0) and onChip:
        layer.bias = bias
    else:
        layer.bias = np.zeros(layer.outputSize[0])
    return layer


'''--------------------------------------------------------------------------'''
'''                     layers below are not used now                        '''
'''--------------------------------------------------------------------------'''

# not actually put view layer on chip, so not used now 
def buildView(net, op, name, weightBits):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module= op['op']
    layer = ViewInfo()
    layer.name = name
    # layer.inputNames = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])
    layer.outputSize = op_outputs[0].size()[1:]
    layer.inputSize  = [inputs[i].size()[1:] for i in range(len(outputs))]
    return layer

# not actually put concat layer on chip, so not used now 
def buildConcat(net, op, name, weightBits):
    inputs=tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module= op['op']
    layer = ConcatInfo()
    layer.name = op_name
    layer.in_nodes = deepcopy(op['in_nodes'])
    layer.out_nodes = deepcopy(op['out_nodes'])
    layer.outputSize = op_outputs[0].size()[1:]
    layer.inputSize = [inputs[i].size()[1:] for i in range(len(outputs))]
    return layer

# not actually put relu layer on chip, so not used now 
def buildRelu(net, op, name, weightBits):
    inputs  = tuple(net.nodes[_] for _ in op['in_nodes'])
    outputs = tuple(net.nodes[_] for _ in op['out_nodes'])
    module= op['op']
    layer = ReluInfo()
    if module.mode == 'quantized':
        layer.mode = 'ann'
    else:
        layer.mode = 'snn'
    layer.name = name
    # layer.inputNames  = deepcopy(op['in_nodes'])
    # layer.outputNames = deepcopy(op['out_nodes'])
    layer.outputSize  = outputs[0].size()[1:]
    layer.inputSize   = [inputs[i].size()[1:] for i in range(len(inputs))]
    return layer


'''--------------------------------------------------------------------------'''
'''                           parse the network                              '''
'''--------------------------------------------------------------------------'''

def parse(net, weightBits, timeStep, needDirect, baseDir, coreType, inputs):
    inputOps, outputOps = parseDirect(net, needDirect)
    networkInfo = NetworkInfo(net)

    # find out which layers are on the chip and which are not
    networkInfo.setStatus(inputOps, outputOps)

    # remove some layers that just pass on the results(relu, view, concat)
    # redirect the edges that connect to these layers to 
    # those that these layers connect to 
    networkInfo.redirect()

    # find out preNet and postNet that are not placed on chip 
    networkInfo.getAuxNet(net, baseDir)

    # take all the layers that are onChip and remove the others
    onChipNetwork = networkInfo.buildOnChipNet(net, weightBits, coreType, inputs)
    
    return onChipNetwork




