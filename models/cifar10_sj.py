from spikingjelly.clock_driven import neuron, layer, surrogate, functional
import torch.nn as nn
import torch.nn.functional as F

'''
    Ver 0.0.0.12
'''
class CifarNetSmall_SJ(nn.Module):
    
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8, eps=1e-4, momentum=0.9),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=1e-4, momentum=0.9),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-4, momentum=0.9),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
    
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.9),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-4, momentum=0.9),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.fc6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*32, 10),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
    
    def forward(self, x):
        xx = self.conv1(x)
        
        xxx = self.conv2(xx)
        xxx = self.conv3(xxx)
        xxx = self.conv4(xxx)
        xxx = self.conv5(xxx)
        out = self.fc6(F.dropout(xxx, p=0.2))
    
        for t in range(1, self.T):
            xxx = self.conv2(xx)
            xxx = self.conv3(xxx)
            xxx = self.conv4(xxx)
            xxx = self.conv5(xxx)

            out += self.fc6(F.dropout(xxx, p=0.2))

        return out / self.T