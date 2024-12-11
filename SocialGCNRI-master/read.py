import torch

f = open('D:\研究生\研二\\2024.4-7\实验\SocialLGN\SocialLGN-master\data.pkl','rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu
print(data)