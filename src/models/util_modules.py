"""
    Helpful PyTorch models that are not provided by default.
    
    Author: Luis Denninger <l_denninger@uni-bonn.de>

"""

import torch
import torch.nn as nn
import math

from typing import Optional

class PointWiseLinear(nn.Module):
    """
        Point-wise linear layer.
        This module computes a a separate linear projection for each point in the input.
    """
    def __init__(self,
                  in_features: int,
                   out_features: int,
                     num_points: int,
                       bias: Optional[bool] = True,
                          ) -> None:
        super(PointWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_points = num_points
        self.weight = nn.Parameter(torch.Tensor(num_points, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_points, out_features))
        else:
            self.bias = None
        self.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward function for the point-wise linear layer.

            Arguments:
                x (torch.Tensor): batched input tensor. Additional dimensions are squeezed and reshaped. [batch_size,..., num_points, in_features]
        """
        shape = x.shape
        x = x.view(-1, self.num_points, self.in_features)
        x = torch.einsum('njk,bnk->bnj', self.weight, x)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        x = x.reshape(*shape[:-1], self.out_features)
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, num_points={self.num_points} bias={self.bias is not None}'