import math
#import torch

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform

class GNNLayer(nn.Cell):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #以下两句可以合为一句话 Xavier 初始化方法如下：mindspore.common.initializer.XavierUniform(gain = 1)
        #self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #torch.nn.init.xavier_uniform_(self.weight)
        self.weight = ms.Parameter(ms.Tensor(shape = (in_features,out_features),dtype = ms.float32,init = XavierUniform(1)))
    
    def construct(self,features, adj1,adj2,adj3, active=True):
        #定义相应操作
        matmul = ops.MatMul()
        spmul = ops.SparseTensorDenseMatmul()
        relu = ops.ReLU()
        
        #support = torch.mm(features, self.weight)
        support = matmul(features, self.weight)
        #稀疏相乘时有 mindspore.nn.SparseTensorDenseMatmul
        #output = torch.spmm(adj, support)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++附加修改
        output = spmul(adj1,adj2,adj3, support)
        #output = matmul(adj,support)
        
        if active:
            #output = F.relu(output)
            output = relu(output)
        return output