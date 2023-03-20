import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('/home/wangqiankun/PAIFlow/ANN2SNN/')
from scipy.io import loadmat
# device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
device=torch.device('cpu')
import tqdm
import spike_dag
from spike_tensor import SpikeTensor
class SpikeLinearSTDP(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None,\
            timesteps=1000, Vthr=12500, reset_mem=0, lower_mem=0, prohibation=11250,\
            lower_weight=0, upper_weight=127, LUT=[], LUT_size=60, shift=29, weight_decay=15, learn=True):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mem_potentail=None

        # self.LUT=LUT
        # self.LUT=torch.tensor([0,-1,2,1,0]).to(device)
        # self.timesteps=timesteps
        # self.Vthr=Vthr
        # self.reset_mem=reset_mem
        # self.lower_mem=lower_mem
        # self.prohibation=prohibation
        # self.lower_weight=lower_weight
        # self.upper_weight=upper_weight
        # self.weight_decay=weight_decay
        # self.learn=learn
        raw_LUT=[0]*LUT_size
        raw_LUT[27:32]=[0,-1,2,1,0]
        self.register_buffer('LUT', torch.tensor(raw_LUT).to(device))
        self.register_buffer('LUT_size', torch.tensor(LUT_size))
        # self.register_buffer('LUT', torch.tensor(LUT).to(device))
        self.register_buffer('shift', torch.tensor(shift))
        self.register_buffer('timesteps', torch.tensor(timesteps))
        self.register_buffer('Vthr', torch.tensor(Vthr))
        self.register_buffer('reset_mem', torch.tensor(reset_mem,dtype=torch.float32))
        self.register_buffer('lower_mem', torch.tensor(lower_mem))
        self.register_buffer('prohibation',torch.tensor(prohibation))
        self.register_buffer('lower_weight',torch.tensor(lower_weight))
        self.register_buffer('upper_weight',torch.tensor(upper_weight))
        self.register_buffer('weight_decay',torch.tensor(weight_decay))
        self.register_buffer('learn',torch.tensor(learn))

        # self.LUT_size=len(LUT)//2

    def init(self,batch_size):
        assert batch_size==1
        self.mem_potentail=torch.zeros(batch_size,self.out_features,dtype=torch.float32).to(device)
        self.spike_out=torch.zeros(self.timesteps,batch_size,self.out_features).to(device)
        self.counter_in=0*torch.ones(batch_size,self.in_features).long().to(device)
        self.counter_in_trace=torch.zeros(batch_size,self.in_features).long().to(device)
        self.counter_out=0*torch.ones(batch_size,self.out_features).long().to(device)
        self.counter_out_trace=torch.zeros(batch_size,self.out_features).long().to(device)
        self.spike_out_reg=torch.zeros(batch_size,self.out_features).to(device) 
    
    def switch_learn(self, learn):
        self.learn.data[...]=learn

    def forward(self, x):
        if isinstance(x,SpikeTensor):
            x=x.data.view(self.timesteps,-1,self.in_features)
        batch_size=x.shape[1]
        self.init(batch_size)
        # for t in tqdm.tqdm(range(self.timesteps)):
        # printId = 148
        for t in range(self.timesteps):
            indata=x[t]
            
            prohibit=0
            if self.spike_out_reg.sum()>0 and self.learn:
                prohibit=self.prohibation
            self.mem_potentail+=(F.linear(indata,self.weight,None)-prohibit)
            self.mem_potentail.clamp_(self.lower_mem,None)
            spikes=self.mem_potentail>=self.Vthr
            self.mem_potentail[spikes]=self.reset_mem
            self.spike_out[t]=spikes.float()

            if not self.learn:
                continue
            
            self.counter_in[indata==1]=0
            self.counter_in_trace[indata==1]=1
            self.counter_in[indata==0]+=1
            self.counter_in.clamp_(None,31)

            self.counter_out[spikes]=0
            self.counter_out_trace[spikes]=1

            self.counter_out[~spikes]+=1
            self.counter_out.clamp_(None,31)
            # with open('tmp2.txt', 'a') as f:
            #     print(f"[{t}] {int(self.mem_potentail[0,printId])} {int(F.linear(indata,self.weight,None)[0,printId])}", file=f,end=" ")
            #     print(f"{int(self.counter_out[0,printId])} {int(self.counter_in[0,0])} {prohibit} {int(self.weight.data[printId,0])}",file=f, end = " ")
            for b in range(batch_size):
                change1=spikes.view(self.out_features,1)*\
                    (self.counter_in[b]<=self.shift+1).view(1,self.in_features)*\
                    self.LUT[(self.counter_in[b]+self.shift).clamp_(0,self.LUT_size-1)].view(1,self.in_features)
                self.weight.data+=change1
                self.weight.data.clamp_(None,self.upper_weight)

                change2=(~spikes.view(self.out_features,1))*\
                    (self.counter_out[b]<=self.shift).view(self.out_features,1)*\
                    (indata[b]==1).view(1,self.in_features)*\
                    self.LUT[(-self.counter_out[b]+self.shift).clamp_(0,self.LUT_size-1)].view(self.out_features,1)
                self.weight.data+=change2
                self.weight.data.clamp_(self.lower_weight,None)
            #     with open('tmp2.txt','a') as f:
            #         print(f"{int(self.weight.data[printId,0])} {spikes[0,printId]} {(change1 + change2)[printId,0]} ", file=f, end ="")
            # with open('tmp2.txt','a') as f:
            #     print(f"{(spikes[0,128:256]).sum()} {(spikes[0,256:384]).sum()} {(spikes[0,384:]).sum()}",file=f)

            self.spike_out_reg=spikes
        for b in range(batch_size):
            if not self.learn:
                continue
            # print(self.weight.data[0,0])
            self.weight.data[(self.counter_out_trace[b]==1).view(self.out_features,1)*(self.counter_in_trace[b]==0).view(1,self.in_features)]-=self.weight_decay
            # print(self.weight.data[0,0])
            self.weight.data.clamp_(self.lower_weight,None)
        out=SpikeTensor(self.spike_out,self.timesteps,scale_factor=1)
        return out
        
class OldSpikeLinearSTDP(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None,\
            timesteps=1000, Vthr=12500, reset_mem=0, lower_mem=0, prohibation=11250,\
            lower_weight=0, upper_weight=127, LUT=[], weight_decay=15, learn=True):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mem_potentail=None

        # self.LUT=LUT
        # self.LUT=torch.tensor([0,-1,2,1,0]).to(device)
        # self.timesteps=timesteps
        # self.Vthr=Vthr
        # self.reset_mem=reset_mem
        # self.lower_mem=lower_mem
        # self.prohibation=prohibation
        # self.lower_weight=lower_weight
        # self.upper_weight=upper_weight
        # self.weight_decay=weight_decay
        # self.learn=learn
        self.register_buffer('LUT', torch.tensor([0,-1,2,1,0]).to(device))
        self.register_buffer('timesteps', torch.tensor(timesteps))
        self.register_buffer('Vthr', torch.tensor(Vthr))
        self.register_buffer('reset_mem', torch.tensor(reset_mem,dtype=torch.float32))
        self.register_buffer('lower_mem', torch.tensor(lower_mem))
        self.register_buffer('prohibation',torch.tensor(prohibation))
        self.register_buffer('lower_weight',torch.tensor(lower_weight))
        self.register_buffer('upper_weight',torch.tensor(upper_weight))
        self.register_buffer('weight_decay',torch.tensor(weight_decay))
        self.register_buffer('learn',torch.tensor(learn))

        self.LUT_size=len(LUT)//2

    def init(self,batch_size):
        assert batch_size==1
        self.mem_potentail=torch.zeros(batch_size,self.out_features,dtype=torch.float32).to(device)
        self.spike_out=torch.zeros(self.timesteps,batch_size,self.out_features).to(device)
        self.counter_in=31*torch.ones(batch_size,self.in_features).long().to(device)
        self.counter_in_trace=torch.zeros(batch_size,self.in_features).long().to(device)
        self.counter_out=31*torch.ones(batch_size,self.out_features).long().to(device)
        self.counter_out_trace=torch.zeros(batch_size,self.out_features).long().to(device)
        self.spike_out_reg=torch.zeros(batch_size,self.out_features).to(device) 
    
    def switch_learn(self, learn):
        self.learn.data[...]=learn

    def forward(self, x):
        if isinstance(x,SpikeTensor):
            x=x.data.view(self.timesteps,-1,self.in_features)
        batch_size=x.shape[1]
        self.init(batch_size)
        # for t in tqdm.tqdm(range(self.timesteps)):
        for t in range(self.timesteps):
            indata=x[t]
            
            prohibit=0
            if self.spike_out_reg.sum()>0 and self.learn:
                prohibit=self.prohibation
            self.mem_potentail+=(F.linear(indata,self.weight,None)-prohibit)
            self.mem_potentail.clamp_(self.lower_mem,None)

            spikes=self.mem_potentail>=self.Vthr
            self.mem_potentail[spikes]=self.reset_mem
            self.spike_out[t]=spikes.float()

            if not self.learn:
                continue
            
            self.counter_in[indata==1]=0
            self.counter_in_trace[indata==1]=1
            self.counter_in[indata==0]+=1
            self.counter_in.clamp_(None,31)

            self.counter_out[spikes]=0
            self.counter_out_trace[spikes]=1

            self.counter_out[~spikes]+=1
            self.counter_out.clamp_(None,31)
            for b in range(batch_size):
                change=spikes.view(self.out_features,1)*\
                    (self.counter_in[b]<=2).view(1,self.in_features)*\
                    self.LUT[(self.counter_in[b]+2).clamp_(0,4)].view(1,self.in_features)
                self.weight.data+=change
                self.weight.data.clamp_(None,self.upper_weight)

                change=(~spikes.view(self.out_features,1))*\
                    (self.counter_out[b]<=2).view(self.out_features,1)*\
                    (indata[b]==1).view(1,self.in_features)*\
                    self.LUT[(-self.counter_out[b]+2).clamp_(0,4)].view(self.out_features,1)
                self.weight.data+=change
                self.weight.data.clamp_(self.lower_weight,None)

            self.spike_out_reg=spikes
        for b in range(batch_size):
            if not self.learn:
                continue
            self.weight.data[(self.counter_out_trace[b]==1).view(self.out_features,1)*(self.counter_in_trace[b]==0).view(1,self.in_features)]-=self.weight_decay
            self.weight.data.clamp_(self.lower_weight,None)
        out=SpikeTensor(self.spike_out,self.timesteps,scale_factor=1)
        return out

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1=SpikeLinearSTDP(784,512,False)
    
    def forward(self,x):
        out=self.fc1(x)
        return out
class OldNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1=OldSpikeLinearSTDP(784,512,False)
    
    def forward(self,x):
        out=self.fc1(x)
        return out

if __name__ =='__main__':
    file_name='/home/wangqiankun/PAIFlow/ANN2SNN/STDP/spike_train.mat'
    spike_train=loadmat(file_name,mat_dtype=True)
    
    oldmodel=OldNet().to(device)
    # oldmodel.fc1.weight.data=torch.randint(0,7,[512,784]).float().to(device)
    oldmodel.fc1.weight.data[...]=2
    indata=torch.from_numpy(spike_train['spike_train']).view(784,100,1000).permute(2,1,0).float().to(device)
    for e in range(1):
        for index in tqdm.tqdm(range(10)):
        # for index in range(1):
            data=indata[:,index,:].view(1000,1,784)
            oldmodel(data)

    model=Net().to(device)
    # model.fc1.weight.data=torch.randint(0,7,[512,784]).float().to(device)
    model.fc1.weight.data[...]=2
    for e in range(1):
        for index in tqdm.tqdm(range(10)):
        # for index in range(1):
            data=indata[:,index,:].view(1000,1,784)
            model(data)

    print((oldmodel.fc1.weight!=model.fc1.weight).sum())

    for e in range(1):
        for index in tqdm.tqdm(range(10)):
        # for index in range(1):
            data=indata[:,index,:].view(1000,1,784)
            model(data)

    print((oldmodel.fc1.weight!=model.fc1.weight).sum())
    # torch.save(model,'/home/wangqiankun/PAIFlow/ANN2SNN/STDP/STDPer.pth')
    save=False
    if save:
        model.fc1.switch_learn(False)
        DAGNet=spike_dag.SpikeDAGModule().to(device)
        DAGNet.add_node('dag_input0')
        DAGNet.inputs_nodes.append('dag_input0')
        DAGNet.add_op('fc1',model.fc1,in_nodes=['dag_input0'],out_nodes=['fc1_out'])
        DAGNet.add_node('fc1_out')
        DAGNet.outputs_nodes=DAGNet.find_end_nodes()
        DAGNet.to(device)
        for index in tqdm.tqdm(range(100)):
            data=indata[:,index,:].view(1000,1,784)
            out=DAGNet(data)



