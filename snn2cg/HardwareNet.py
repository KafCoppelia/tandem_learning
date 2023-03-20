from .hwConfig import Hardware, CoreSet
import numpy as np
from copy import deepcopy
import math

class RelayCore:
    def __init__(self, inputWidth, LCN, targetLCN, isOffline):
        self.inputWidth = inputWidth
        self.LCN = LCN
        self.targetLCN = targetLCN
        self.isOffline = isOffline
        if not isOffline:
            assert self.inputWidth == 1, f"online cores can only support SNN now.\n"
        self.maxTime = 0
        self.minTime = 1 << 20
        self.maxDiff = Hardware.getAttr("SLOTNUM", self.isOffline) // self.LCN - 1
        self.axons = dict()
        self.neurons = list()
        self.destBase = list()
        self.destStar = list()

    def canRelay(self, axonId, neuronNum, relayTime, LCN, targetLCN, inputWidth, isOffline):
        '''
        decide whether this core can relay new neurons or not:
        cannot relay neurons in situations below  
            - no unused axons
            - no unused neurons
            - max begin time - min begin time >= maxDiff 
        '''
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", self.isOffline)
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        maxNeuronNum = hardwareNeuronNum // self.inputWidth
        if self.isOffline != isOffline:
            return False
        if axonId in self.axons or self.inputWidth * (1 + len(self.axons)) <= hardwareAxonNum:
            if len(self.neurons) + neuronNum <= maxNeuronNum * self.inputWidth:
                maxTime = max(self.maxTime, relayTime)
                minTime = min(self.minTime, relayTime)
                if self.maxTime == 0 or maxTime - minTime < self.maxDiff:
                    return True
        return False

    def relay(self, axonId, neuronNum, gPlusCoreId, beginTime):
        if axonId not in self.axons:
            self.axons.update({axonId: len(self.axons)})
        neuronBeg  = Hardware.getfullId2(gPlusCoreId, len(self.neurons))
        neuronEnd  = neuronBeg + neuronNum
        fullAxonId = Hardware.getfullId2(gPlusCoreId, self.axons[axonId])
        self.neurons+=[axonId for i in range(neuronNum)]
        self.maxTime = max(beginTime, self.maxTime)
        self.minTime = min(beginTime, self.minTime)
        assert self.maxTime - self.minTime < self.maxDiff
        return fullAxonId, list(range(neuronBeg, neuronEnd))

    def connect(self, neuronId, fullAxonId, starId):
        if len(self.destBase) < len(self.neurons):
            self.destBase += [-1 for i in range((len(self.neurons) - len(self.destBase)))]
            self.destStar += [0 for i in range((len(self.neurons) - len(self.destStar)))]
        self.destBase[neuronId] = fullAxonId
        self.destStar[neuronId] = starId
        return

    def store(self, timeStep):
        coreInfo = dict()
        SNNEN = (self.inputWidth != 8)
        neuronUnit =  self.LCN
        if self.isOffline:
            coreInfo['core'] = [
                0,                               #0  weight_width
                int(math.log2(self.LCN)),        #1  LCN
                int(not SNNEN),                  #2  input_width
                int(not SNNEN),                  #3  spike_width
                len(self.neurons) * neuronUnit,  #4  neuron_num
                0,                               #5  pool_max
                self.maxTime,                    #6  tick_wait_start
                timeStep,                        #7  tick_wait_end
                int(SNNEN),                      #8  SNN_EN
                int(math.log2(self.targetLCN)),  #9  target_LCN
                1 << 9,                          #10 test_chip_addr
            ]
        else:
            coreInfo['core'] = [
                0,                         #0  bit_select
                int(math.log2(self.LCN)),  #1  group_select
                0,                         #2  lateral_inhi_value
                0,                         #3  weight_decay_value
                1,                         #4  upper_weight
                0,                         #5  lower_weight
                0,                         #6  neuron_start
                len(self.neurons) - 1,     #7  neuron_end
                0,                         #8  inhi_core_x_star
                0,                         #9  inhi_core_y_star
                self.maxTime,              #10 core_start_time
                timeStep,                  #11 core_hold_time
                0,                         #12 LUT_random_en
                0,                         #13 decay_random_en
                0,                         #14 decay_random_en
                0,                         #15 online_mode_en
                1 << 9,                    #16 test_address
                1,                         #17 random_seed
            ]
        coreInfo['LUT'] = []
        if not self.isOffline:
            coreInfo['LUT'] = [0 for i in range(60)]
        coreInfo['neuron'] = dict()
        neuronPos = 0
        threshold_neg = (1 << 28)
        for i, neuronId in enumerate(self.neurons):
            if self.destBase[i] == -1:
                continue
            tickRelative = (Hardware.getSlotId(self.destBase[i], self.inputWidth, self.isOffline))
            destAxon = Hardware.getAxonId(self.destBase[i], self.inputWidth, self.isOffline) * self.inputWidth
            destCore = Hardware.getgPlusCoreId(self.destBase[i])
            destChip = Hardware.getGroupId(self.destBase[i])
            if self.isOffline:
                configInfo = [
                    tickRelative,      #0  tick_relative
                    destAxon,          #1  addr_axon
                    destCore,          #2  addr_core_x
                    self.destStar[i],  #3  addr_core_y
                    0,                 #4  addr_core_x_ex
                    0,                 #5  addr_core_y_ex
                    0,                 #6  addr_chip_x
                    0,                 #7  addr_chip_y
                    0,                 #8  reset_mode
                    0,                 #9  reset_v
                    0,                 #10 leak_post
                    0,                 #11 threshold_mask_ctrl
                    1,                 #12 threshold_neg_mode
                    threshold_neg,     #13 threshold_neg
                    1,                 #14 threshold_pos
                    0,                 #15 leak_reversal_flag
                    0,                 #16 leak_det_stoch
                    0,                 #17 leak_v
                    0,                 #18 weight_det_stoch
                    8,                 #19 bit_truncate
                    0,                 #20 vjt_pre
                ]
            else:
                configInfo = [
                    0,                 #0  leakage_reg           0
                    destAxon,          #13 addr_axon             1
                    destCore,          #9  addr_core_x           2
                    self.destStar[i],  #10 addr_core_y           3
                    0,                 #11 addr_core_x_star      4
                    0,                 #12 addr_core_y_star      5
                    0,                 #7  addr_chip_x           6
                    0,                 #8  addr_chip_y           7
                    1,                 #1  threshold_reg         8
                    0,                 #2  floor_threshold_reg   9
                    0,                 #3  reset_potential_reg   10
                    0,                 #4  initial_potential_reg 11
                    0,                 #5  potential_reg         12
                    tickRelative,      #6  time_slot             13
                    0,                 #14 plasticity_start      14
                    0,                 #15 plasticity_end        15
                ]
            weightBit = 1 << (self.axons[neuronId])
            if not SNNEN:
                weight = "{:0144b}".format(weightBit)[::-1]
            else:
                if self.isOffline:
                    weight = "{:01152b}".format(weightBit)[::-1]
                else:
                    weight = "{:01024b}".format(weightBit)[::-1]
            
            for j in range(self.LCN):
                neuronInfo = {
                    'parameter': configInfo,
                    'weight':weight
                }
                coreInfo['neuron'][neuronPos] = neuronInfo
                neuronPos += 1
        return coreInfo

class RelayGroup:
    def __init__(self):
        self.gPlusCoreId = -1
        self.relayCores = dict()
        self.newCoreFlag = True
    
    def setBegId(self, gPlusCoreId):
        self.gPlusCoreId = gPlusCoreId

    def newCore(self):
        self.newCoreFlag = True

    def relayNeuron(self, axonId, neuronNum, beginTime, LCN, targetLCN, inputWidth, isOffline):
        assert self.gPlusCoreId > 0
        lastId = self.gPlusCoreId - 1
        if self.newCoreFlag or lastId not in self.relayCores or \
                not self.relayCores[lastId].canRelay(axonId, neuronNum, beginTime, LCN, targetLCN, inputWidth, isOffline):
            self.relayCores[self.gPlusCoreId] = RelayCore(inputWidth, LCN, targetLCN, isOffline)
            
            CoreSet.register(deepcopy(self.gPlusCoreId), isOffline)

            self.newCoreFlag = False
            lastId = self.gPlusCoreId
            self.gPlusCoreId += 1

            

        assert lastId in self.relayCores, f"{lastId}\n {sorted(self.relayCores.keys())}"
        return self.relayCores[lastId].relay(axonId, neuronNum, lastId, beginTime)

    def connect(self, fullNeuronId, fullAxonId, starId):
        gPlusCoreId = Hardware.getgPlusCoreId(fullNeuronId)
        neuronId = Hardware.getNeuronId(fullNeuronId)

        self.relayCores[gPlusCoreId].connect(neuronId, fullAxonId, starId)

    def store(self, timeStep):
        relayGroupInfo = dict()
        print(f"relayCore num : {len(self.relayCores.keys())} {list(self.relayCores.keys())}")
        for coreId, core in self.relayCores.items():
            relayGroupInfo[coreId] = core.store(timeStep)
        return relayGroupInfo

class ComputeNeuron:
    def __init__(
        self, axonPlace, weights, bias, LCN, bitWidth, 
        resetMode, threshold, inputWidth, SNNEN, bitTrunc,
        #online parameters
        isOffline, lowerMem, resetMem, initMem, learnMode
        ):
        # self.LCN = LCN
        # self.targetLCN = 1
        self.isOffline = isOffline
        self.lowerMem = lowerMem
        self.resetMem = resetMem
        self.initMem = initMem
        self.learnMode = learnMode
        
        hardwareAxonNum = Hardware.getAttr("AXONNUM", isOffline)
        maxAxonNum = hardwareAxonNum // inputWidth
        self.weight = np.zeros(shape=[maxAxonNum, LCN * bitWidth],dtype=bool)
        for axonId, weight in weights.items():
            axonPos = axonPlace[axonId][0]
            timeSlot = axonPos // maxAxonNum
            slotId = axonPos % maxAxonNum
            self.weight[
                slotId, (timeSlot * bitWidth) : ((timeSlot * bitWidth) + bitWidth)
            ] = weight
        self.leak = int(bias)
        self.resetMode = resetMode
        self.threshold = int(threshold)
        self.bitTrunc = int(bitTrunc)
        self.dest = [-1,0]
    
    def copy(self):
        raise NotImplementedError()
        bitWidth = self.weight.shape[1]
        maxAxonNum = self.weight.shape[0]
        computeNeuron = ComputeNeuron(
            maxAxonNum, None, 0, 1, bitWidth, self.resetMode, self.threshold
        )
        computeNeuron.weight = deepcopy(self.weight)
        computeNeuron.leak = deepcopy(self.leak)
        return computeNeuron

    def connect(self, fullAxonId, starId):
        self.dest[:] = [fullAxonId, starId]
        return
    
    def store(
        # self, neuronPos, inputWidth, bitWidth, learnAxonNum, learnMode
        self, neuronPos, outputWidth, bitWidth, learnAxonNum, learnMode
    ):
        # assert self.isOffline
        neuronInfo = dict()
        reset = 0
        if self.resetMode == "subtraction":
            reset = 1
        elif self.resetMode == "non-reset":
            reset = 2
        neuronNum = self.weight.shape[1]
        if self.dest[0] == -1:
            return dict()
        isOutput = False
        if self.dest[0] <= -Hardware.OUTPUTBEG:
            isOutput = True
            dest = -self.dest[0]
        else:
            dest = self.dest[0]
        # destSlot = Hardware.getSlotId(dest, inputWidth, self.isOffline)
        # destAxon = Hardware.getAxonId(dest, inputWidth, self.isOffline)
        destSlot = Hardware.getSlotId(dest, outputWidth, self.isOffline)
        destAxon = Hardware.getAxonId(dest, outputWidth, self.isOffline) * outputWidth
        tickRelative = destSlot

        
        if not isOutput:
            destCore = Hardware.getgPlusCoreId(dest)
        else:
            destCore = -Hardware.getgPlusCoreId(dest)
        
        if self.learnMode:
            plasticityBeg = 0
            plasticityEnd1 = self.weight.shape[0] - 1
            plasticityEnd2 = (learnAxonNum % self.weight.shape[0]) - 1
        else:
            plasticityBeg = 0
            plasticityEnd1 = self.weight.shape[0] - 1
            plasticityEnd2 = self.weight.shape[0] - 1

        if self.isOffline:
            configInfo = [
                tickRelative,      #0  tick_relative
                destAxon,          #1  addr_axon
                destCore,          #2  addr_core_x
                self.dest[1],      #3  addr_core_y
                0,                 #4  addr_core_x_ex
                0,                 #5  addr_core_y_ex
                0,                 #6  addr_chip_x
                0,                 #7  addr_chip_y
                reset,             #8  reset_mode
                0,                 #9  reset_v
                0,                 #10 leak_post
                0,                 #11 threshold_mask_ctrl
                1,                 #12 threshold_neg_mode
                (1 << 28),         #13 threshold_neg
                self.threshold,    #14 threshold_pos
                0,                 #15 leak_reversal_flag
                0,                 #16 leak_det_stoch
                self.leak,         #17 leak_v
                0,                 #18 weight_det_stoch
                self.bitTrunc,     #19 bit_truncate
                0,                 #20 vjt_pre
            ]
        else:
            configInfo = [
                self.leak,         #0  leakage_reg            0 
                destAxon,          #13 addr_axon              1 
                destCore,          #9  addr_core_x            2 
                self.dest[1],      #10 addr_core_y            3 
                0,                 #11 addr_core_x_star       4 
                0,                 #12 addr_core_y_star       5 
                0,                 #7  addr_chip_x            6 
                0,                 #8  addr_chip_y            7 
                self.threshold,    #1  threshold_reg          8 
                self.lowerMem,     #2  floor_threshold_reg    9 
                self.resetMem,     #3  reset_potential_reg    10 
                self.initMem,      #4  initial_potential_reg  11 
                self.initMem,      #5  potential_reg          12 
                tickRelative,      #6  time_slot              13 
                plasticityBeg,     #14 plasticity_start       14 
                plasticityEnd1,    #15 plasticity_end         15 
            ]

        for i in range(neuronNum):

            if  not self.isOffline:
                # weight = str(weightInts[i])
                LCN_id = i // bitWidth
                innerLCN = i % bitWidth
                axonNum = self.weight.shape[0]
                segLen = axonNum // bitWidth
                axonBeg = innerLCN*segLen
                LCNBeg =  LCN_id * bitWidth
                weight = self.weight[(axonBeg):(axonBeg + segLen),(LCNBeg):(LCNBeg + bitWidth)]
                weight = weight[:,::-1]
                weight = weight.reshape(-1).astype(int).tolist()
                weightList = [str(i) for i in weight]
                weight = "".join(weightList)
            else:
                weightList = self.weight[:, i].astype(int).tolist()
                weightList = [str(i) for i in weightList]
                weight = "".join(weightList)

            if self.isOffline:
                neuronInfo[neuronPos + i] = {
                    'parameter': configInfo,
                    'weight':weight
                }
            else:
                if i != neuronNum - 1:
                    neuronInfo[neuronPos + i] = {
                        'parameter':configInfo,
                        'weight':weight
                    }
                else:
                    newConfigInfo = deepcopy(configInfo)
                    newConfigInfo[-1] = plasticityEnd2
                    neuronInfo[neuronPos + i] = {
                        'parameter':newConfigInfo,
                        'weight':weight
                    }
        return neuronInfo
    
    def setOutput(self, offset, outputWidth):
        # axonPos = Hardware.OUTPUTBEG + offset
        hardwareAxonNum = Hardware.getAttr("AXONNUM", self.isOffline)
        hardwareCOREBASE = Hardware.getAttr("COREBASE",self.isOffline)
        axonNum = hardwareAxonNum // outputWidth
        coreOffset = offset // axonNum
        axonOffset = offset % axonNum
        axonPos = -(Hardware.OUTPUTBEG + (coreOffset << hardwareCOREBASE) + axonOffset)
        self.dest[0] = axonPos
    
    def getOutput(self):
        # offset = self.dest[0] - Hardware.OUTPUTBEG
        # coreOffset = offset // Hardware.AXONNUM
        # axonOffset = offset % Hardware.AXONNUM
        # axonPos = Hardware.OUTPUTPOS + (coreOffset << Hardware.COREBASE) + axonOffset
        return -(self.dest[0])
    
    def getRealOutput(self, outputWidth):
        if outputWidth == 1:
            return self.getOutput()
        else:
            hardwareCOREBASE = Hardware.getAttr("COREBASE",self.isOffline)
            hardwareCOREMASK = (1 << hardwareCOREBASE) - 1

            oldAxonOffset = ((-self.dest[0]) - Hardware.OUTPUTBEG) & (hardwareCOREMASK)
            newAxonOffset = oldAxonOffset * outputWidth
            return (-self.dest[0]) - oldAxonOffset + newAxonOffset
        
    
    
    def reBuild(self, bitWidth, LCN):
        weightBase = 2 ** np.arange(bitWidth)
        weightBase = weightBase.reshape(1,bitWidth)
        slotNum = self.weight.shape[0]
        if bitWidth > 1:
            weightBase[0,-1] *= -1
        connections = dict()
        base = 0
        axonBase = 0
        for i in range(LCN):
            weight = weightBase * self.weight[
                :, (i * bitWidth): (i * bitWidth + bitWidth)
            ]
            originWeight = weight.sum(1)
            #TODO: use np.nonzeros
            for j, weight in enumerate(originWeight):
                if weight == 0:
                    continue
                connections[j + i * slotNum] = weight
        info = [connections, self.leak, self.threshold, self.bitTrunc]
        return info

class ComputeCore:
    
    def __init__(
        self, LCN, bitWidth, SNNEN, inputWidth, outputWidth, 
        maxPool, axonPlace, isOffline, 
        *onlineParameters
    ):
        hardwareNeuronNum = Hardware.getAttr("NEURONNUM", isOffline)

        #whether this is an online learning core or not 
        self.isOffline = isOffline

        #for both online and offline cores
        self.maxNeuronNum = hardwareNeuronNum * inputWidth
        self.LCN = LCN
        self.bitWidth = bitWidth
        self.axonPlace = axonPlace
        self.neurons = dict()
        self.begTime = 0

        #for only offline inferring cores
        self.SNNEN = SNNEN
        self.inputWidth = inputWidth
        self.outputWidth = outputWidth
        self.maxPool = maxPool
        self.targetLCN = 1

        #for only online learning cores
        if not isOffline:
            self.LUT, self.resetMem, self.prohibition, self.lowerWeight, \
                self.upperWeight, self.weightDecay, self.lowerMem, self.learnMode, \
                self.inhiCoreStar = onlineParameters
            axonNum = len(self.axonPlace)
            self.learnAxonNum = axonNum
            
            #check
            if self.learnMode:
                axons = set(range(axonNum))
                for axonId, axonPos in self.axonPlace.items():
                    assert len(axonPos) == 1
                    assert axonPos[0] in axons
                    axons.remove(axonPos[0])
                assert len(axons) == 0
            
        else:
            self.LUT = None
            self.lowerMem = 0
            self.resetMem = 0
            self.prohibition = 0
            self.lowerWeight = 0
            self.upperWeight = 0
            self.weightDecay = 0
            self.learnMode = False
            self.learnAxonNum = 0
    
    def addNeuron(self, neuronId, weights, bias, resetMode, threshold, bitTrunc):
        neuron = ComputeNeuron(
            self.axonPlace, weights, bias, self.LCN, self.bitWidth, resetMode, 
            threshold, self.inputWidth, self.SNNEN, bitTrunc, 
            #online parameters
            self.isOffline, self.lowerMem, self.resetMem, self.resetMem, self.learnMode
        )
        self.neurons[neuronId] = neuron
        return
    
    def connect(self, neuronId, fullAxonId, starId):
        self.neurons[neuronId].connect(fullAxonId, starId)
    
    def setTargetLCN(self, targetLCN):
        self.targetLCN = targetLCN

    def setOutput(self, neuronId, offset):
        self.neurons[neuronId].setOutput(offset, self.outputWidth)

    def getOutput(self, neuronId):
        return self.neurons[neuronId].getOutput()
    
    def getRealOutput(self, neuronId):
        return self.neurons[neuronId].getRealOutput(self.outputWidth)

    def setBegTime(self, begTime):
        self.begTime = begTime

    def store(self, timeStep):
        inputWidth = int(self.inputWidth == 8)
        outputWidth = int(self.outputWidth == 8)
        SNNEN = int(self.SNNEN)
        coreInfo = dict()
        neuronNum = len(self.neurons) * self.LCN * self.bitWidth
        bitSelect = {1:0,2:1,4:2,8:3}


        if self.isOffline:
            coreInfo['core'] = [
                bitSelect[self.bitWidth],       #0  weight_width
                int(math.log2(self.LCN)),       #1  LCN
                inputWidth,                     #2  input_width
                outputWidth,                    #3  spike_width
                neuronNum,                      #4  neuron_num
                self.maxPool,                   #5  pool_max
                self.begTime,                   #6  tick_wait_start
                timeStep,                       #7  tick_wait_end
                SNNEN,                          #8  SNN_EN
                int(math.log2(self.targetLCN)), #9  target_LCN
                1 << 9                          #10 test_chip_addr
            ]
        else:
            coreInfo['core'] = [
                bitSelect[self.bitWidth],      #0  bit_select
                int(math.log2(self.LCN)),      #1  group_select
                self.prohibition,              #2  lateral_inhi_value
                self.weightDecay,              #3  weight_decay_value
                self.upperWeight,              #4  upper_weight
                self.lowerWeight,              #5  lower_weight
                0,                             #6  neuron_start
                neuronNum - 1,                 #7  neuron_end
                self.inhiCoreStar,             #8  inhi_core_x_star
                0,                             #9  inhi_core_y_star
                self.begTime,                  #10 core_start_time
                timeStep,                      #11 core_hold_time
                0,                             #12 LUT_random_en
                0,                             #13 decay_random_en
                0,                             #14 leakage_order
                self.learnMode,                #15 online_mode_en
                1 << 9,                        #16 test_address
                1,                             #17 random_seed
            ]
        coreInfo['LUT'] = []
        if not self.isOffline:
            coreInfo['LUT'] = deepcopy(self.LUT.tolist())
        neuronPos = 0
        coreInfo['neuron'] = dict()
        for neuronId, neuron in self.neurons.items():
            neuronInfo = neuron.store(
                # neuronPos, self.inputWidth, self.bitWidth, self.learnAxonNum,
                neuronPos,  self.outputWidth, self.bitWidth, self.learnAxonNum,
                self.learnMode
            )
            for tmpId, info in neuronInfo.items():
                coreInfo['neuron'][tmpId] = info
            neuronPos += len(neuronInfo)
        if neuronPos == 0:
            # if there's no valid neurons, then the core never starts
            if self.isOffline:
                coreInfo['core'][6] = 0
            else:
                coreInfo['core'][10] = 0
        return coreInfo

    def reBuild(self):
        neuronInfo = dict()
        # neuronUnit = self.LCN * self.bitWidth
        neuronUnit = 1
        for neuronId, neuron in self.neurons.items():
            neuronInfo[neuronUnit * neuronId] = neuron.reBuild(
                self.bitWidth, self.LCN
            )
        coreInfo = [neuronInfo, self.SNNEN, self.outputWidth, self.maxPool]
        return coreInfo

class LocalPlace:
    def __init__(self, isOffline):
        self.cores = dict()
        self.isOffline = isOffline
        return

    def addCore(self, coreId, core):
        self.cores[coreId] = core

    def connect(self, coreId, neuronId, fullAxonId, starId):
        self.cores[coreId].connect(neuronId, fullAxonId, starId)
        return

    def setBegTime(self, coreId, begTime):
        assert coreId in self.cores, f"{coreId} not in coreset {self.cores.keys()}"
        self.cores[coreId].setBegTime(begTime)

    def getCoreNum(self):
        return len(self.cores)

    def canExtend(self, coreId):
        return self.cores[coreId].canExtend()

    def extend(self, coreId, neuronId, num):
        ids = self.cores[coreId].extend(neuronId, num)
        for i in range(len(ids)):
            ids[i] = Hardware.getfullId(0, coreId, ids[i])
        return ids

    def store(self, groupId, timeStep):
        localInfo = dict()
        coreIds = list()
        for coreId, core in self.cores.items():
            gPlusCoreId = Hardware.getgPlusCoreId2(groupId, coreId)
            localInfo[gPlusCoreId] = core.store(timeStep)
            coreIds.append(gPlusCoreId)
        return localInfo, coreIds

    def setOutput(self, coreId, neuronId, offset):
        self.cores[coreId].setOutput(neuronId, offset)

    def getOutput(self, coreId, neuronId):
        return self.cores[coreId].getOutput(neuronId)
    
    def getRealOutput(self, coreId, neuronId):
        return self.cores[coreId].getRealOutput(neuronId)

    def reBuild(self):
        coreInfo = dict()
        for coreId, core in self.cores.items():
            coreInfo[coreId] = core.reBuild()
        return coreInfo
    
class ComputeGroup:
    def __init__(self):
        self.localPlace = dict()

    def addLocalPlace(self, localId, localPlace):
        self.localPlace[localId] = localPlace
        isOffline = localPlace.isOffline
        for coreId in localPlace.cores.keys():
            gPlusCoreId = Hardware.getgPlusCoreId2(localId, coreId)
            CoreSet.register(gPlusCoreId, isOffline)
        return

    def getCoreNum(self):
        coreNum = 0
        for localPlace in self.localPlace.values():
            coreNum += localPlace.getCoreNum()
        return coreNum

    def setBegTime(self, neuronId, begTime):
        groupId = Hardware.getGroupId(neuronId)
        coreId = Hardware.getCoreId(neuronId)
        self.localPlace[groupId].setBegTime(coreId, begTime)

    def connect(self, fullNeuronId, fullAxonId, starId):
        groupId = Hardware.getGroupId(fullNeuronId)
        coreId = Hardware.getCoreId(fullNeuronId)
        neuronId = Hardware.getNeuronId(fullNeuronId)
        self.localPlace[groupId].connect(coreId, neuronId, fullAxonId, starId)
        return

    def canExtend(self, fullId):
        groupId = Hardware.getGroupId(fullId)
        coreId = Hardware.getCoreId(fullId)
        return self.localPlace[groupId].canExtend(coreId)

    def extend(self, fullId, num):
        groupId = Hardware.getGroupId(fullId)
        coreId = Hardware.getCoreId(fullId)
        neuronId = Hardware.getNeuronId(fullId)
        ids = self.localPlace[groupId].extend(coreId, neuronId, num)
        for i in range(len(ids)):
            ids[i] = Hardware.addGroupId(groupId, ids[i])
        return ids

    def store(self, timeStep):
        localPlaceInfo = dict()
        cores = list()
        for groupId, localPlace in self.localPlace.items():
            localPlaceInfo[groupId],tmpC= localPlace.store(groupId, timeStep)
            cores+=tmpC
        cores.sort()
        print(f"localplace core number: {len(cores)} {cores}")
        return localPlaceInfo

    def setOutput(self, fullId, offset):
        groupId = Hardware.getGroupId(fullId)
        coreId = Hardware.getCoreId(fullId)
        neuronId = Hardware.getNeuronId(fullId)
        self.localPlace[groupId].setOutput(coreId, neuronId, offset)

    def getOutput(self, fullId):
        groupId = Hardware.getGroupId(fullId)
        coreId = Hardware.getCoreId(fullId)
        neuronId = Hardware.getNeuronId(fullId)
        return self.localPlace[groupId].getOutput(coreId, neuronId)

    def getRealOutput(self, fullId):
        groupId = Hardware.getGroupId(fullId)
        coreId = Hardware.getCoreId(fullId)
        neuronId = Hardware.getNeuronId(fullId)
        return self.localPlace[groupId].getRealOutput(coreId, neuronId)

    def getOutNeuron(self, fullId):
        gPlusCoreId = Hardware.getgPlusCoreId(fullId)
        neuronId = Hardware.getNeuronId(fullId)
        return [gPlusCoreId, neuronId]

    def reBuild(self, groupId):
        infos = self.localPlace[groupId].reBuild()
        newInfos = dict()
        for coreId, coreInfo in infos.items():
            gPlusCoreId = Hardware.getgPlusCoreId2(groupId, coreId)
            newInfos[gPlusCoreId] = coreInfo
        return newInfos

class HardwareNetwork:
    def __init__(self):
        self.computeGroup = ComputeGroup()
        self.relayGroup = RelayGroup()
        return
    def beginRelay(self):
        groupId = max(self.computeGroup.localPlace.keys()) + 1
        self.relayGroup.setBegId(
            Hardware.getgPlusCoreId2(groupId, 0)
        )
    def store(self, timeStep):
        info = {
            'localPlace': self.computeGroup.store(timeStep),
            'relay': self.relayGroup.store(timeStep)
        }
        return info



