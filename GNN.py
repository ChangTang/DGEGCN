import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #以下两句可以合为一句话 Xavier 初始化方法如下：mindspore.common.initializer.XavierUniform(gain = 1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        #稀疏相乘时有 mindspore.nn.SparseTensorDenseMatmul
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

