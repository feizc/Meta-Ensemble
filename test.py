import torch 

a = torch.gt(torch.arange(128), 64).long().unsqueeze(0)
print(a.size())

