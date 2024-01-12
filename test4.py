from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)

import torch
print(torch.cuda.is_available())

x = torch.empty(5, 3)
print(x)

zero_x = torch.zeros(5, 3, dtype=torch.long)
print(zero_x)

# tensor 数值是 [5.5, 3]
tensor1 = torch.tensor([5.5, 3])
print(tensor1)

# 显示定义新的尺寸是 5*3，数值类型是 torch.double
tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)  # new_* 方法需要输入 tensor 大小
print(tensor2)

# 修改数值类型
tensor3 = torch.randn_like(tensor2, dtype=torch.float)
print('tensor3: ', tensor3)

print(tensor3.size())