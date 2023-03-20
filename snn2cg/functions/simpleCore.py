import numpy as np
import json
class Hardware:
    AXON_ON = 1024
    NEURON_ON = 1024
    AXON_OFF = 1152
    NEURON_OFF = 512
    CHIP_MASK = (1 << 10) - 1
    CORE_MASK = (1 << 10) - 1

def read_config(configPath):
    read_config = json.load(configPath)
    config = dict()
    
    offline_neuron_config = read_config['offline_core']['neuron']
    config['offline_neuron'] = [
        int(offline_neuron_config['tick_relative']), #0  tick_relative
        0,                                           #1  addr_axon
        0,                                           #2  addr_core_x
        0,                                           #3  addr_core_y
        0,                                           #4  addr_core_x_ex
        0,                                           #5  addr_core_y_ex
        0,                                           #6  addr_chip_x
        0,                                           #7  addr_chip_y
        int(offline_neuron_config['reset_mode']),    #8  reset_mode
        int(offline_neuron_config['reset_v']),       #9  reset_v
        0,                                           #10 leak_post
        0,                                           #11 threshold_mask_ctrl
        1,                                           #12 threshold_neg_mode
        (1 << 28),                                   #13 threshold_neg
        int(offline_neuron_config['threshold_pos']),  #14 threshold_pos
        0,                                           #15 leak_reversal_flag
        0,                                           #16 leak_det_stoch
        int(offline_neuron_config['leak_v']),        #17 leak_v
        0,                                           #18 weight_det_stoch
        int(self.bitTrunc),                          #19 bit_truncate
        0,                                           #20 vjt_pre
    ]

    offline_core_config = read_config['offline_core']['core']
    offline_bitWidth =  int(offline_core_config['weight_width'])
    offline_LCN =       int(offline_core_config['LCN'])
    offline_targetLCN = int(offline_core_config['target_LCN'])
    offline_inputWidth = int(int(offline_core_config['inputWidth']) == 8)
    offline_outputWidth = int(int(offline_core_config['outputWidth']) == 8)
    config['offline_core'] = [
        int(math.log2(offline_bitWidth)),             #0  weight_width
        int(math.log2(offline_LCN)),                  #1  LCN
        offline_inputWidth,                           #2  input_width
        offline_outputWidth,                          #3  spike_width
        int(offline_core_config['neuron_num']),       #4  neuron_num
        int(offline_core_config['pool_max']),         #5  pool_max
        int(offline_core_config['tick_wait_start']),  #6  tick_wait_start
        int(offline_core_config['tick_wait_end']),    #7  tick_wait_end
        int(offline_core_config['SNN_EN']),           #8  SNN_EN
        int(math.log2(offline_targetLCN)),            #9  target_LCN
        1 << 9                                        #10 test_chip_addr
    ]

    online_neuron_config = read_config['online_core']['neuron']
    config['online_neuron'] = [
        int(online_neuron_config['leakage_reg']),         #0  leakage_reg
        int(online_neuron_config['threshold_reg']),       #1  threshold_reg
        int(online_neuron_config['floor_threshold_reg']), #2  floor_threshold_reg
         int(online_neuron_config['reset_potential_reg']),     #3  reset_potential_reg
        self.initMem,      #4  initial_potential_reg
        self.initMem,      #5  potential_reg
        tickRelative,      #6  time_slot
        0,                 #7  addr_chip_x
        0,                 #8  addr_chip_y
        destCore,          #9  addr_core_x
        self.dest[1],      #10 addr_core_y
        0,                 #11 addr_core_x_star
        0,                 #12 addr_core_y_star
        0,                 #13 addr_axon
        plasticityBeg,     #14 plasticity_start
        plasticityEnd1,    #15 plasticity_end
    ]

    online_core_config = read_config['online_core']['core']
    config['online_core'] = [
        
    ]





def weight_method_1(size, rate):
    if rate == 0.0:
        return np.zeros(size)
    elif rate >= 1.0:
        return np.ones(size)
    else:
        return (np.random.rand(size) <= rate).int()
    return

def connect_method_1(in_chip, in_core, in_axon, is_offline, core_config):
    if is_offline:
        pass
    else:
        pass
    return

def connect_method_2(in_chip, in_core, in_axon, is_offline, core_config):
    return 




class OffLineCore:
    def __init__(self, ):
        self.neuronConfig = [
            tickRelative,      #0  tick_relative
            0,                 #1  addr_axon
            0,                 #2  addr_core_x
            0,                 #3  addr_core_y
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
        self.coreConfig = [
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
            1 << 9
        ]
        return
    def setWeightMapping(self):
        return
    def setRelative(self):
        return 
    def setConnectionMapping(self):
        return
    def store(self):
        return 
    

class OnlineCore:
    def __init__(self):
        self.neuronConfig = [

        ]
        self.coreConfig = [

        ]
        return

