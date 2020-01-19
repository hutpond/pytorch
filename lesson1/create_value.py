import sys
import torch


def create_value():
  x1 = torch.empty(5, 3)
  x2 = torch.rand(5, 3)
  x3 = torch.zeros(5, 3, dtype=torch.long)
  x4 = torch.tensor([5.5, 3])
  print(x1)
  print(x2)
  print(x3)
  print(x4)


if __name__ == '__main__':
  create_value()