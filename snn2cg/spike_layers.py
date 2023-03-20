import torch
import torch.nn as nn
import torch.nn.functional as F
from .spike_tensor import SpikeTensor
from torch import Tensor
from typing import Any

"""
ANN量化计算模式
卷积 & 全连接
1. 每一层有输入x、权重w、偏置b、输出o，每个变量都可表示为a=a_q*a_scale，a_q为int8数值，a_scale为量化因子
2. 由于输出是截取的，简单起见，我们规定w_scale=2**k，那么o_scale=x_scale*2**k'即可满足截取条件
3. 偏置和输出分别有b_scale和o_scale
模拟计算
o=clamp(conv/linear(x,w,b)/o_scale,-128,127)*o_scale
上板计算
o_q=clamp(conv/linear(x_q,w_q,b_q)>>k'',-128,127)
上板参数
k''，quant_base=1
模拟量化方法
1. 输入x已经量化好，得到原始的x_q，与x_scale
2. 对每一层
    2.1 先量化权重，暂时使用最大值进行量化，找到合适的k，获得w_q，w_scale
    2.2 设置b_scale=w_scale*x_scale，bias的位宽为30bit，目前使用int24表示
    2.2 然后搜索合适的k'，让输出与原始输出尽量接近
    2.3 给输出带上o_scale，进入下一层
模拟量化参数转上板参数
k''=k'
"""

def warp_one_in_one_out_func(func):
    def new_func(input,*args,**kwargs):
        if isinstance(input,SpikeTensor):
            out=SpikeTensor(func(input.data,*args,**kwargs),input.timesteps,input.scale_factor)
        else:
            out=func(input,*args,**kwargs)
        return out
    return new_func
F.dropout=warp_one_in_one_out_func(F.dropout)

# mem : 256*64*32*32
def generate_spike_mem_potential(out_s,mem_potential,Vthr,reset_mode):
    """
    out_s: is a Tensor of the output of different timesteps [timesteps, *sizes]
    mem_potential: is a placeholder Tensor with [*sizes]
    """
    assert reset_mode == 'subtraction'
    spikes = []

    for t in range(out_s.size(0)):
        mem_potential += out_s[t]
        spike=(mem_potential>=Vthr).float()
        mem_potential-=spike*Vthr
        spikes.append(spike)
    return spikes

def conv2d_generate_spike_out(spike, spike_in, spike_out, mem, T, weight, bias, stride, padding, Vthr, leaky_rate):

    for t in range(T):
        x = F.conv2d(spike_in[:,t,:,:,:], weight, None, stride, padding) + bias

        mem = mem * leaky_rate + x - Vthr * spike
        spike = torch.ge(mem, Vthr).float()

        spike_out[:,t,:,:] = spike

    spike_count_out = torch.sum(spike_out, dim=1)
    
    return spike_out, spike_count_out

def linear_generate_spike_out(spike, pot_in, spike_out, mem, T, bias, Vthr, leaky_rate):

    for t in range(T):
        x = pot_in[:,t,:].squeeze() + bias

        mem = mem * leaky_rate + x - Vthr * spike
        spike = torch.ge(mem, Vthr).float()

        spike_out[:,t,:] = spike

    spike_count_out = torch.sum(spike_out, dim=1).squeeze(dim=1)
    
    return spike_out, spike_count_out

def generate_spike_quick(out,Vthr,reset_mode,timesteps):
    """
    DEBUG
    """
    assert reset_mode == 'subtraction'
    spikes = []
    for t in range(timesteps):
        spike=(out>=Vthr).float()
        out-=spike*Vthr
        spikes.append(spike)
    return spikes

class SpikeLayer(nn.Module):
    
    def forward(self, x):
        return self.surrogate_forward(x)
        
        # elif self.mode=='snn':
        #     if not isinstance(x, SpikeTensor):
        #         replica_data = torch.cat([x for _ in range(self.timesteps)], 0)
        #         x = SpikeTensor(replica_data, self.timesteps, scale_factor=1)

        #     out=self.snn_forward(x)
        #     return out
        
        # elif self.mode=='ann':
        #     if isinstance(x,SpikeTensor):
        #         x=x.to_float()
        #     return self.ann_forward(x)
        # elif self.mode=='Qann':
        #     return self.Qann_forward(x)
        # elif self.mode=='quantized':
        #     return self.Qann_forward_chip(x)
        # else:
        #     raise NotImplementedError

class SpikeReLU(nn.Module):
    def __init__(self,threshold,quantize=False):
        super().__init__()
        self.max_val=1
        self.quantize=quantize
        self.threshold=threshold
        self.activation_bitwidth=None
    
    def forward(self,x):
        if isinstance(x, SpikeTensor):
            return x
        else:
            return F.relu(x,inplace=True)

class SpikeConv2d(SpikeLayer, nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, weightNorm, biasNorm, stride=1, \
                    padding=0, dilation=1, groups=1, \
                    bias=True, padding_mode='zeros', leaky_rate=1, bn=None):
        # TODO : add batchnorm here
        super(SpikeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, padding_mode)
        self.mem_potential=None
        self.register_buffer('out_scales',torch.ones(out_channels))
        self.register_buffer('Vthr',torch.ones(1))
        self.register_buffer('leakage',torch.zeros(out_channels))
        self.register_buffer('quant_base',torch.ones(1))
        self.register_buffer('shift_bit',torch.zeros(1))

        self.reset_mode='subtraction'
        # self.quant_base=None
        self.bn=bn
        self.spike_generator=generate_spike_mem_potential
        self.mode="snn" # mode can be {ann, snn, snn_cpu}
        self.spike_gen_mode='normal' # spike_gen_mode can be {normal, quick}
        self.timesteps=None
        
        '''
            Add new code below
        '''
        self.SpikeConv2dLIF = SpikeConv2dLIF.apply
        self.weightNorm = weightNorm
        self.biasNorm = biasNorm
        self.device = torch.device("cuda")
        self.stride = stride
        self.padding = padding
        self.leaky_rate = leaky_rate
        '''
			Add new code above
        '''
    
    def surrogate_forward(self, input_feature_st):
        output_features_st, _ = self.SpikeConv2dLIF(
            input_feature_st, 0, self.weightNorm, self.biasNorm, self.stride, self.padding, \
                1, self.leaky_rate, self.device)

        return output_features_st

    def snn_forward(self, x):
        Vthr = self.Vthr.view(1,-1,1,1)
        if self.spike_gen_mode=='normal':
            S = F.conv2d(x.data, self.weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

            if self.bn is not None:
                S=self.bn(S)
            chw = S.size()[1:]
            out_s = S.view(x.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = self.spike_generator(out_s,self.mem_potential,Vthr,self.reset_mode)
            
        elif self.spike_gen_mode=='quick':
            if self.bias is None:
                bias=None
            else:
                bias=self.bias*x.timesteps
            S = F.conv2d(x.firing_number(), self.weight, bias, self.stride, self.padding, self.dilation,
                        self.groups)
            spikes=generate_spike_quick(S,Vthr,self.reset_mode,self.timesteps)
        out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
        
        return out

    def ann_forward(self,x):
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def Qann_forward(self,x):
        if not hasattr(x,"scale"):
            if hasattr(self,'x_scale'):
                x_scale=self.x_scale
            else:
                x_scale=x.abs().max()/127
            x=(x/x_scale).round_().clamp_(-128,127)*x_scale
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # out = (out / self.o_scale).clamp(-128,127) * self.o_scale
        # out.scale=self.o_scale
        o_scale=self.out_scales[0]
        out=(out/o_scale).clamp(0,255)*o_scale
        out.scale=o_scale
        # out=(out/self.o_scale).clamp(-128,127)*self.o_scale
        # out.scale=self.o_scale
        return out

    def Qann_forward_chip(self,x):
        if not hasattr(x,"scale"):
            if hasattr(self,'x_scale'):
                x_scale=self.x_scale
            else:
                x_scale=x.abs().max()/127
            x=(x/x_scale).round_().clamp_(-128,127)
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = F.relu(out)
        out = (out >> self.shift_bit).floor_()
        # out.clamp_(-128,127)
        out.clamp_(0,255)
        out.scale=self.out_scales[0]
        # out.scale=self.o_scale
        return out
    
     
class SpikeConv2dLIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_in, features_in, weight, bias, stride, padding, Vthr, leaky_rate, device=torch.device('cuda')):
        N, T, in_channels, iH, iW = spike_in.shape
        out_channels, in_channels, kH, kW = weight.shape
        mem = torch.zeros_like(F.conv2d(spike_in[:,0,:,:,:], weight, bias, stride, padding), device=device)
        bias_distribute = F.conv2d(torch.zeros_like(spike_in[:,0,:,:,:]), weight, bias, stride, padding)/T # init the membrane potential with the bias
        _, _, outH, outW = mem.shape
        spike_out = torch.zeros(N, T, out_channels, outH, outW, device=device)
        spike = torch.zeros(N, out_channels, outH, outW, device=device)
        
        return conv2d_generate_spike_out(spike, spike_in, spike_out, mem, T, weight, bias_distribute, stride, padding, Vthr, leaky_rate)

    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        
        grad_spike_count_out = grad_spike_count_out.clone()
        # grad_spike_in, grad_weight, grad_device, grad_bias, grad_stride, grad_padding = None, \
		# 		None, None, None, None, None
    
        return None, grad_spike_count_out, None, None, None, None, None, None, None

class SpikeConvTranspose2d(SpikeLayer,nn.ConvTranspose2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1,
                 bias=True,dilation=1, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups,bias,dilation, padding_mode)
        self.mem_potential=None
        self.register_buffer('out_scales',torch.ones(out_channels))
        self.register_buffer('Vthr',torch.ones(1))
        self.register_buffer('leakage',torch.zeros(out_channels))
        self.reset_mode='subtraction'
        self.quant_base=None
        self.spike_generator=generate_spike_mem_potential
        self.mode=None # mode can be {ann, snn}
        self.timesteps=None
    
    def ann_forward(self, x):
        out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return out
    
    def snn_forward(self, x):
        Vthr = self.Vthr.view(1,-1,1,1)
        out = F.conv_transpose2d(x.data, self.weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        chw = out.size()[1:]
        out_s = out.view(x.timesteps, -1, *chw)
        self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
        spikes=self.spike_generator(out_s,self.mem_potential,Vthr,self.reset_mode)
        out = SpikeTensor(torch.cat(spikes, 0), x.timesteps,self.out_scales)
        return out


class SpikeLinear(SpikeLayer, nn.Linear):
    
    def __init__(self, in_features, out_features, weightNorm, biasNorm, bias=True, leaky_rate=1):
        super(SpikeLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage',torch.zeros(out_features))
        self.register_buffer('quant_base',torch.ones(1))
        self.register_buffer('shift_bit',torch.zeros(1))
        self.reset_mode='subtraction'
        # self.quant_base=None
        self.spike_generator = generate_spike_mem_potential
        self.mode = "snn" # mode can be {ann, snn}
        self.timesteps = None
        
        '''
            Add new code below
        '''
        self.SpikeLinearLIF = SpikeLinearLIF.apply
        self.device = torch.device("cuda")
        self.weightNorm = weightNorm
        self.biasNorm = biasNorm
        self.leaky_rate = leaky_rate  
        '''
            Add new code above
        '''

    def ann_forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out

    def surrogate_forward(self, input_feature_st):
        output_st, _ = self.SpikeLinearLIF(
            input_feature_st, 0, self.weightNorm, self.biasNorm, 1, self.leaky_rate, self.device)
        
        return output_st
    
    def snn_forward(self, x):
        Vthr=self.Vthr.view(1,-1)
        out = F.linear(x.data, self.weight, self.bias)
        chw = out.size()[1:]
        out_s = out.view(x.timesteps, -1, *chw)
        self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
        spikes=self.spike_generator(out_s,self.mem_potential,Vthr,self.reset_mode)
        
        out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
        # if self.last_layer:
        #     return out.data
        
        return out

    def Qann_forward(self,x):
        if not hasattr(x,"scale"):
            x_scale=x.abs().max()/127
            x=(x/x_scale).round_().clamp_(-128,127)*x_scale
        out = F.linear(x, self.weight, self.bias)
        # out = (out / self.o_scale).clamp(-128,127) * self.o_scale
        # out.scale=self.o_scale
        o_scale=self.out_scales[0]
        out=(out/o_scale).clamp(0,255)*o_scale
        out.scale=o_scale
        # out=(out/self.o_scale).clamp(-128,127)*self.o_scale
        # out.scale=self.o_scale
        return out

    def Qann_forward_chip(self,x):
        if not hasattr(x,"scale"):
            if hasattr(self,'x_scale'):
                x_scale=self.x_scale
            else:
                x_scale=x.abs().max()/127
            x=(x/x_scale).round_().clamp_(-128,127)
        out = out = F.linear(x, self.weight, self.bias)
        out = F.relu(out)
########################### Warring: must last layer #############################################
        out = (out >> self.shift_bit).floor_().clamp_(0,255) # * self.out_scales[0]
##################################################################################################
        out.scale=self.out_scales[0]
        return out
    

class SpikeLinearLIF(torch.autograd.Function):
	
    @staticmethod
    def forward(ctx, spike_in, ann_output, weight, bias, Vthr, leaky_rate, device=torch.device('cuda')):
        
        N, T, _ = spike_in.shape
        out_features = bias.shape[0]
        pot_in = spike_in.matmul(weight.t())
        spike_out = torch.zeros_like(pot_in, device=device)
        bias_distribute = bias.repeat(N, 1)/T # distribute bias through simulation time steps
        mem = torch.zeros(N, out_features, device=device)
        spike = torch.zeros(N, out_features, device=device) # init input spike train
        
        spike_out, spike_count_out = linear_generate_spike_out(spike, pot_in, spike_out, mem, T, bias_distribute, Vthr, leaky_rate)
        
        return spike_out, spike_count_out
    
    @staticmethod
    def backward(ctx, grad_spike_out, grad_spike_count_out):
        """Auxiliary function only, no gradient required"""
        grad_ann_out = grad_spike_count_out.clone()
        
        return None, grad_ann_out, None, None, None, None, None

class SpikeAdd(nn.Module):
    def __init__(self,out_features):
        super().__init__()
        self.register_buffer('out_scales', torch.ones(out_features))
        self.register_buffer('Vthr', torch.ones(1))
        self.register_buffer('leakage',torch.zeros(out_features))
        # self.register_buffer('bias',torch.zeros(out_features))
        self.register_buffer('ma',torch.zeros(1))
        self.register_buffer('mb',torch.zeros(1))
        self.register_buffer('shift_bit',torch.zeros(1))
        self.weight=nn.Parameter(torch.ones([out_features,2]))
        self.weight.alpha=1
        self.reset_mode='subtraction'
        self.quant_base=None
        self.spike_generator=generate_spike_mem_potential
        self.mode=None # mode can be {ann, snn}
        self.timesteps=None

    def forward(self,a,b):
        if self.mode=='snn':
            if not isinstance(a,SpikeTensor):
                replica_data = torch.cat([a for _ in range(self.timesteps)], 0)
                a = SpikeTensor(replica_data, self.timesteps, scale_factor=1)
            if not isinstance(b,SpikeTensor):
                replica_data = torch.cat([b for _ in range(self.timesteps)], 0)
                b = SpikeTensor(replica_data, self.timesteps, scale_factor=1)
            Vthr=self.Vthr.view(1,-1,1,1)
            out = a.data*self.weight[:,0].view(1,-1,1,1)+b.data*self.weight[:,1].view(1,-1,1,1)
            chw = out.size()[1:]
            out_s = out.view(a.timesteps, -1, *chw)
            self.mem_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes=self.spike_generator(out_s,self.mem_potential,Vthr,self.reset_mode)
            out = SpikeTensor(torch.cat(spikes, 0), a.timesteps, self.out_scales)
        elif self.mode=='ann':
            if isinstance(a,SpikeTensor):
                a=a.to_float()
            if isinstance(b,SpikeTensor):
                b=b.to_float()
            out = a*self.weight[:,0].view(1,-1,1,1)+b*self.weight[:,1].view(1,-1,1,1)+self.bias.view(1,-1,1,1)
        elif self.mode=='Qann':
            out=a+b
            o_scale=self.out_scales[0]
            out=(out/o_scale).clamp(-128,127)*o_scale
            out.scale=o_scale
            # out=(out/self.o_scale).clamp(-128,127)*self.o_scale
            # out.scale=self.o_scale
        elif self.mode=='quantized':
            out=self.weight[:,0].view(1,-1,1,1)*a+self.weight[:,1].view(1,-1,1,1)*b
            # out=self.ma*a+self.mb*b
            out=(out>>self.shift_bit).floor_().clamp_(-128,127)
            # out.scale=self.o_scale
            out.scale=self.out_scales[0]
        else:
            raise NotImplementedError
        return out

class SpikeMaxPool2d(SpikeLayer,nn.MaxPool2d):
    def __init__(self, kernel_size, stride = None, padding= 0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        # self.last_layer=last_layer
        # self.register_buffer('out_scales', torch.ones(out_features))
        # self.register_buffer('Vthr', torch.ones(1))
        # self.register_buffer('leakage',torch.zeros(out_features))
        # self.register_buffer('quant_base',torch.ones(1))
        # self.register_buffer('shift_bit',torch.zeros(1))
        # self.reset_mode='subtraction'
        # self.quant_base=None
        # self.spike_generator=generate_spike_mem_potential
        self.mode=None # mode can be {ann, snn}
        self.timesteps=None
    
    def ann_forward(self,x):
        out=nn.MaxPool2d.forward(self,x)
        if hasattr(x,'scale'):
            out.scale=x.scale
        return out

'''
    Layers below is added by KafCoppelia
'''
class SpikeIdentity(nn.Identity):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(nn.Identity, self).__init__()
