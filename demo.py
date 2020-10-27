import torch
import torch.nn as nn
import torch.nn.functional as F
a = torch.tensor([16, 2, 3, 4])
b = torch.tensor([3., 5, 6, 7])
loss_fn = nn.L1Loss()
loss = loss_fn(a, b)
print(loss)
