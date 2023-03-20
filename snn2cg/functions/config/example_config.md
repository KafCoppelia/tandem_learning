```json
{
    "type": "ASIC",
    "offline_core":{
        "neuron":{
            "tick_relative": 0,
            "addr_axon": 0,            # no_use
            "addr_core_x": 0,          # no_use
            "addr_core_y":0,           # no_use
            "addr_core_x_ex": 0,       # no_use
            "addr_core_y_ex": 0,       # no_use
            "addr_chip_x": 0,          # no_use
            "addr_chip_y": 0,          # no_use
            "reset_mode": 0,          
            "reset_v": 0,
            "leak_post": 0,            # no_use
            "threshold_mask_ctrl": 0,  # no_use
            "threshold_neg_mode": 1,   # no_use
            "threshold_neg": -1,       # no_use
            "threshold_pos": 1,
            "leak_reversal_flag": 0,   # no_use
            "leak_det_stoch": 0,       # no_use
            "leak_v": 0,               
            "weight_det_stoch": 0,     # no_use
            "bit_truncate": 0, 
            "vjt_pre": 0               # no_use
        },
        "core":{
            "weight_width": 8,    # [1,2,4,8]
            "LCN": 1, 
            "input_width": 8,     # 1 or 8
            "spike_width": 8,     # 1 or 8
            "neuron_num":  1888,
            "pool_max": 0,
            "tick_wait_start": 1,
            "tick_wait_end": 1,
            "SNN_EN": 1,
            "target_LCN": 1,
            "test_chip_addr":512  # no use
        },
        "connect_mode": 1,
        "weight_mode": 1,
        "weight_rate": 1,
        "neuron_num": 1888,
        "position":["*****_*****"]
    },
    "online_core":{
        "neuron":{
            "leakage_reg": 0,
            "threshold_reg":0, 
            "addr_axon"  : 0,
            "addr_core_x": 0,
            "addr_core_y":0,
            "addr_core_x_star":0,
            "addr_core_y_star":0,
            "addr_chip_x":0,
            "addr_chip_y":0,
            
            "floor_threshold_reg":0,
            "reset_potential_reg":0,
            "initial_potential_reg":0,  
            "potential_reg":0, 
            "time_slot":0,
            "plasticity_start":0,
            "plasticity_end":0
        },
        "core":{
            "bit_select": 1,
            "group_select": 1,
            "lateral_inhi_value": 0,
            "weight_decay_value": 0,
            "upper_weight": 127,
            "lower_weight": -128,
            "neuron_start": 0,
            "neuron_end": 1023,
            "inhi_core_x_star": 0, 
            "inhi_core_y_star": 0,
            "core_start_time": 1,
            "core_hold_time": 1,
            "LUT_random_en": 0,
            "decay_random_en": 0,
            "leakage_order": 0,
            "online_mode_en": 0,
            "test_address": 512,
            "random_seed":1
        },
        "connect_mode":1,
        "weight_mode":1,
        "weight_rate": 1,
        "neuron_num": 1024,
        "position":["111**_111**"]
    },
    "dest_chip": "00000_00000"
}
```