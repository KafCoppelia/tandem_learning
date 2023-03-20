import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.snn import LinearBN1d_if, ConvBN2d_if, ConvBN2dInference_if, LinearBN1dInference_if
from lib.functional import InputDuplicate

from snn2cg.spike_layers import SpikeLayer, SpikeConv2d, SpikeLinear
from snn2cg.spike_dag import SpikeDAGModule


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(128, eps=1e-4, momentum=0.9))

        self.conv2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(256, eps=1e-4, momentum=0.9))

        self.conv3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.conv4 = nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(1024, eps=1e-4, momentum=0.9))

        self.conv5 = nn.Sequential(nn.Conv2d(1024, 512, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(512, eps=1e-4, momentum=0.9))

        self.fc6 = nn.Sequential(nn.Linear(8*8*512, 1024, bias=False),
                                 nn.BatchNorm1d(1024, eps=1e-4, momentum=0.9))

        self.fc7 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        # Conv Layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # FC Layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = self.fc7(F.dropout(x, p=0.2))

        return F.log_softmax(x, dim=1)

class CifarNetSmall(nn.Module):
    def __init__(self):
        super(CifarNetSmall, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(8, eps=1e-4, momentum=0.9))

        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(16, eps=1e-4, momentum=0.9))
        
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(32, eps=1e-4, momentum=0.9))
    
        self.conv4 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64, eps=1e-4, momentum=0.9))
        
        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(32, eps=1e-4, momentum=0.9))

        self.fc6 = nn.Linear(8*8*32, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)

        # Conv Layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # FC Layers
        x = x.view(x.size(0), -1)
        # x = F.relu(self.fc4(x))
        x = self.fc6(F.dropout(x, p=0.2))

        return F.log_softmax(x, dim=1)

class CifarNetSmallIF(nn.Module):

    def __init__(self, neuronParam, Tsim):
        super(CifarNetSmallIF, self).__init__()
        self.T = Tsim
        self.neuronParam = neuronParam
        self.conv1 = ConvBN2d_if(
            3, 8, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv2 = ConvBN2d_if(
            8, 16, 3, stride=2, padding=1, neuronParam=self.neuronParam)
        self.conv3 = ConvBN2d_if(
            16, 32, 3, stride=2, padding=1, neuronParam=self.neuronParam)
        self.conv4 = ConvBN2d_if(
            32, 64, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        self.conv5 = ConvBN2d_if(
            64, 32, 3, stride=1, padding=1, neuronParam=self.neuronParam)
        # self.fc6 = LinearBN1d_if(8*8*32, 512, neuronParam=self.neuronParam)
        self.fc6 = nn.Linear(8*8*32, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x_spike, x = InputDuplicate.apply(x, self.T)
        x_spike = x_spike.view(-1, self.T, 3, 32, 32)
        x = x.view(-1, 3, 32, 32)

        # Conv Layer
        x_spike, x = self.conv1(x_spike, x)
        x_spike, x = self.conv2(x_spike, x)
        x_spike, x = self.conv3(x_spike, x)
        x_spike, x = self.conv4(x_spike, x)
        x_spike, x = self.conv5(x_spike, x)

        # FC Layers
        x = x.view(x.size(0), -1)
        # x_spike = x_spike.view(x_spike.size(0), self.T, -1)

        # x_spike, x = self.fc6(x_spike, x)
        x = self.fc6(F.dropout(x, p=0.2))

        return F.log_softmax(x, dim=1)

class SpikingNet(SpikeLayer):

    def __init__(self, Tsim, leaky_rate):
        super(SpikingNet, self).__init__()
        self.T = Tsim

        self.conv1 = SpikeConv2d(
            3, 64, 3, stride=1, padding=1, leaky_rate=leaky_rate)
        self.conv2 = SpikeConv2d(64, 128, 3, stride=2,
                                 padding=1, leaky_rate=leaky_rate)
        self.conv3 = SpikeConv2d(
            128, 256, 3, stride=2, padding=1, leaky_rate=leaky_rate)
        self.conv4 = SpikeConv2d(
            256, 512, 3, stride=1, padding=1, leaky_rate=leaky_rate)
        self.conv5 = SpikeConv2d(
            512, 256, 3, stride=1, padding=1, leaky_rate=leaky_rate)
        self.fc6 = SpikeLinear(8*8*256, 512, leaky_rate=leaky_rate)
        self.fc7 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x_spike, x = InputDuplicate.apply(x, self.T)
        x_spike = x_spike.view(-1, self.T, 3, 32, 32)
        x = x.view(-1, 3, 32, 32)

        # Conv Layer
        x_spike, x = self.conv1(x_spike, x)
        x_spike, x = self.conv2(x_spike, x)
        x_spike, x = self.conv3(x_spike, x)
        x_spike, x = self.conv4(x_spike, x)
        x_spike, x = self.conv5(x_spike, x)

        # FC Layers
        x = x.view(x.size(0), -1)  # TODO 看一下尺寸
        x_spike = x_spike.view(x_spike.size(0), self.T, -1)

        x_spike, x = self.fc6(x_spike, x)
        x = self.fc7(F.dropout(x, p=0.2))

        return F.log_softmax(x, dim=1)

class InferenceNet(nn.Module):
    
    def __init__(self, pt_dir, device=torch.device("cuda")):
        super(InferenceNet, self).__init__()
        state = torch.load(pt_dir, map_location=device)
        self.weights = state["model_state_dict"]
        
        print(list(self.weights.keys()))

    @staticmethod
    def _get_inf_conv2d_params(conv2d_w, conv2d_b, bn2d_w, bn2d_b, bn2d_running_m, bn2d_running_v):
        device = torch.device("cuda")
        conv2d_weight = conv2d_w.detach().to(device)
        conv2d_bias = conv2d_b.detach().to(device)

        bnGamma = bn2d_w
        bnBeta = bn2d_b
        bnMean = bn2d_running_m
        bnVar = bn2d_running_v

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Conv' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(conv2d_weight.permute(
            1, 2, 3, 0), ratio).permute(3, 0, 1, 2)
        biasNorm = torch.mul(conv2d_bias - bnMean, ratio) + bnBeta

        return weightNorm, biasNorm

    def get_inf_conv2d_params(self, conv_name: str):
        conv2d_w_n = self.weights["module." + conv_name + ".conv2d.weight"]
        conv2d_b_n = self.weights["module." + conv_name + ".conv2d.bias"]
        bn2d_w_n = self.weights["module." + conv_name + ".bn2d.weight"]
        bn2d_b_n = self.weights["module." + conv_name + ".bn2d.bias"]
        bn2d_rm_n = self.weights["module." + conv_name + ".bn2d.running_mean"]
        bn2d_rv_n = self.weights["module." + conv_name + ".bn2d.running_var"]
        
        return InferenceNet._get_inf_conv2d_params(conv2d_w_n, conv2d_b_n, bn2d_w_n, bn2d_b_n, bn2d_rm_n, bn2d_rv_n)
    
    @staticmethod
    def _get_inf_linear1d_params(linear_w, linear_b, bn1d_w, bn1d_b, bn1d_running_m, bn1d_running_v):
        device = torch.device("cuda")
        linearif_weight = linear_w.detach().to(device)
        linearif_bias = linear_b.detach().to(device)

        bnGamma = bn1d_w
        bnBeta = bn1d_b
        bnMean = bn1d_running_m
        bnVar = bn1d_running_v

        # re-parameterization by integrating the beta and gamma factors
        # into the 'Linear' layer weights
        ratio = torch.div(bnGamma, torch.sqrt(bnVar))
        weightNorm = torch.mul(linearif_weight.permute(1, 0), ratio).permute(1, 0) 
        biasNorm = torch.mul(linearif_bias - bnMean, ratio) + bnBeta
        
        return weightNorm, biasNorm

    def get_inf_linear1d_params(self, linear_name: str):
        linear_w_n = self.weights["module." + linear_name + ".linear.weight"]
        linear_b_n = self.weights["module." + linear_name + ".linear.bias"]
        bn1d_w_n = self.weights["module." + linear_name + ".bn1d.weight"]
        bn1d_b_n = self.weights["module." + linear_name + ".bn1d.bias"]
        bn1d_rm_n = self.weights["module." + linear_name + ".bn1d.running_mean"]
        bn1d_rv_n = self.weights["module." + linear_name + ".bn1d.running_var"]
        
        return InferenceNet._get_inf_linear1d_params(linear_w_n, linear_b_n, bn1d_w_n, bn1d_b_n, bn1d_rm_n, bn1d_rv_n)

class CifarNetSmallIF_Inference(InferenceNet):

    def __init__(self, neuronParam, Tsim: int, pt_dir: str, device=torch.device("cuda")):
        super(CifarNetSmallIF_Inference, self).__init__(pt_dir, device)
        self.T = Tsim
        self.neuronParam = neuronParam
        
        w1, b1 = self.get_inf_conv2d_params("conv1")
        self.conv1 = ConvBN2dInference_if(
            w1, b1, stride=1, padding=1, neuronParam=self.neuronParam)
        
        w2, b2 = self.get_inf_conv2d_params("conv2")
        self.conv2 = ConvBN2dInference_if(
            w2, b2, stride=2, padding=1, neuronParam=self.neuronParam)
        
        w3, b3 = self.get_inf_conv2d_params("conv3")
        self.conv3 = ConvBN2dInference_if(
            w3, b3, stride=2, padding=1, neuronParam=self.neuronParam)
        
        w4, b4 = self.get_inf_conv2d_params("conv4")
        self.conv4 = ConvBN2dInference_if(
            w4, b4, stride=1, padding=1, neuronParam=self.neuronParam)
        
        w5, b5 = self.get_inf_conv2d_params("conv5")
        self.conv5 = ConvBN2dInference_if(
            w5, b5, stride=1, padding=1, neuronParam=self.neuronParam)
        
        self.fc6 = nn.Linear(8*8*32, 10)
        self.fc6.weight.data, self.fc6.bias.data = self.weights["module.fc6.weight"], self.weights["module.fc6.bias"]

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x_spike, x = InputDuplicate.apply(x, self.T)
        x_spike = x_spike.view(-1, self.T, 3, 32, 32)

        # Conv Layer
        x_spike = self.conv1(x_spike)
        x_spike = self.conv2(x_spike)
        x_spike = self.conv3(x_spike)
        x_spike = self.conv4(x_spike)
        x_spike = self.conv5(x_spike)

        # FC Layers
        x_spike = x_spike.view(x_spike.size(0), self.T, -1)
        xc = torch.mean(x_spike, dim=1)

        out = self.fc6(F.dropout(xc, p=0.2))

        return F.log_softmax(out, dim=1)

class CifarNetSmallSpiking_Inference(SpikeLayer, InferenceNet):

    def __init__(self, Tsim: int, leaky_rate: float, pt_dir: str, device=torch.device("cuda")):
        super(CifarNetSmallSpiking_Inference, self).__init__(pt_dir, device)
        self.T = Tsim
        # self.reference_net = CifarNetSmallIF_Inference(neuronParam, Tsim, f"./exp/cifar10/lif_small_tencode{Tsim}_ckp.pt", device)
        
        w1, b1 = self.get_inf_conv2d_params("conv1")
        self.conv1 = SpikeConv2d(
            3, 8, 3, w1, b1, stride=1, padding=1, leaky_rate=leaky_rate)
        
        w2, b2 = self.get_inf_conv2d_params("conv2")
        self.conv2 = SpikeConv2d(
            8, 16, 3, w2, b2, stride=2, padding=1, leaky_rate=leaky_rate)
        
        w3, b3 = self.get_inf_conv2d_params("conv3")
        self.conv3 = SpikeConv2d(
            16, 32, 3, w3, b3, stride=2, padding=1, leaky_rate=leaky_rate)
        
        w4, b4 = self.get_inf_conv2d_params("conv4")
        self.conv4 = SpikeConv2d(
            32, 64, 3, w4, b4, stride=1, padding=1, leaky_rate=leaky_rate)
        
        w5, b5 = self.get_inf_conv2d_params("conv5")
        self.conv5 = SpikeConv2d(
            64, 32, 3, w5, b5, stride=1, padding=1, leaky_rate=leaky_rate)

        # self.fc6 = nn.Linear(8*8*32, 10)
        # self.fc6.weight.data, self.fc6.bias.data = self.weights["module.fc6.weight"], self.weights["module.fc6.bias"]

    def forward(self, x):
        # x = x.view(-1, 3*32*32)
        # x_spike, x = InputDuplicate.apply(x, self.T)
        # x_spike = x_spike.view(-1, self.T, 3, 32, 32)
        
        # Conv Layer
        x_spike = self.conv1(x_spike)
        x_spike = self.conv2(x_spike)
        x_spike = self.conv3(x_spike)
        x_spike = self.conv4(x_spike)
        x_spike = self.conv5(x_spike)

        # FC Layers
        # x_spike = x_spike.view(x_spike.size(0), self.T, -1)
        # xc = torch.mean(x_spike, dim=1)

        # out = self.fc6(F.dropout(xc, p=0.2))

        return x_spike

def gen_inference_net_dag(spnet: InferenceNet):
    dag = SpikeDAGModule()

    dag.add_node("input")
    dag.add_node('conv1_out')
    dag.add_node('conv2_out')
    dag.add_node('conv3_out')
    dag.add_node('conv4_out')
    # dag.add_node('conv5_out')
    dag.add_node('output')
    
    dag.add_op('conv1', spnet.conv1, ['input'], ['conv1_out'])
    dag.add_op('conv2', spnet.conv2, ['conv1_out'], ['conv2_out'])
    dag.add_op('conv3', spnet.conv3, ['conv2_out'], ['conv3_out'])
    dag.add_op('conv4', spnet.conv4, ['conv3_out'], ['conv4_out'])
    dag.add_op('conv5', spnet.conv5, ['conv4_out'], ['output'])
    # dag.add_op('fc6', spnet.fc6, ['conv5_out'], ['output'])

    dag.inputs_nodes = ["input"]
    dag.outputs_nodes = ["output"]

    return dag