from models.cifar10 import CifarNetSmallSpiking_Inference, gen_inference_net_dag
from snn2cg.hwConfig import Hardware
from snn2cg.transformer import transform
import torch
from snn2cg.parser import parse

if __name__ == "__main__":
    Tencode = 1
    neuronParam = {
		'neuronType': 'IF',
		'vthr': 1,
		'leaky_rate_mem': 0.8
	}
    device = torch.device("cuda")
    
    spnet = CifarNetSmallSpiking_Inference(Tencode, 1, f"./exp/cifar10/if_small_tencode{Tencode}_ckp.pt", device)
    
    net = gen_inference_net_dag(spnet)
    bit = 8
    time = 16
    direct = True
    _dir = "./output"
    core = "offline"
    hardware = "v2"
    
    _input = torch.rand([10, 2, 3, 32, 32]).to(device)
    
    onChipNet = parse(net, bit, time, direct, _dir, core, _input)
    
    Hardware.setNoCLevel([1], [1], True)
    # Hardware.setNoCLevel([2,2], [2,2], True)
    # Hardware.setNoCLevel([1], [1], False)
    hardwareNetwork, softwareNetwork, weightMappings = transform(
        onChipNet, bit, 1, time, hardware
    )
    # softwareNetwork.print("conv4",3* 36 + 2 * 6 + 5)
    # inputSizes = onChipNet.getShapes()
    # checkOneLayer(
    #     net, onChipNet.opOutputs, 
    #     onChipNet.tensorSizes,inputSizes, softwareNetwork, 
    #     hardwareNetwork, args.time
    # )
    # storeNetInfo(onChipNet, infoDir)
    # storeMappingInfo(hardwareNetwork, softwareNetwork, weightMappings, infoDir, args.time)