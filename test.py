import torch

a = torch.rand([10, 3, 32, 32])
aa = a.view(-1, 3*32*32)
print(aa.shape)
b = aa.unsqueeze(dim=1).repeat(1, 20, 1)
print(b.shape)
c = b.sum(dim=1)
print(c.shape)
bb = b.view(-1, 20, 3, 32, 32) # spike_x
print(bb.shape)
cc = c.view(-1, 3, 32, 32) # x
print(cc.shape)

# ----------

k = torch.rand([256, 8, 8])

kk = k.view(k.size(0), 8, -1)
print(kk.shape)