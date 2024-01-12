import torch

n_data = torch.ones(100,2)
print(n_data)

x0 = torch.normal(2*n_data,1)
print(x0)
y0 = torch.zeros(100)
print(y0)

x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)
