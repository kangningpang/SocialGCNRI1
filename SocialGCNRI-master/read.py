import torch

f = open('filename','rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu
print(data)
