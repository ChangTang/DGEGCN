from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from myloss_mind import KLDivLoss

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
"""

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.nn import Adam
from mindspore import ParameterTuple
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net


#from utils import load_data, load_graph
from utils_mind import load_data, load_graph, XavierNormal
#from GNN import GNNLayer
from GNN_mind import GNNLayer
from evaluation import eva
from collections import Counter



import warnings
warnings.filterwarnings("ignore")



#设置环境上下文
context.set_context(mode  = context.GRAPH_MODE,device_target = "GPU")
#context.set_context(mode  = context.PYNATIVE_MODE,device_target = "CPU")

#AE已改mindspore
class AE(nn.Cell):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        #mindspore.nn.Dense
        self.enc_1 = nn.Dense(n_input, n_enc_1)
        self.enc_2 = nn.Dense(n_enc_1, n_enc_2)
        self.enc_3 = nn.Dense(n_enc_2, n_enc_3)
        self.z_layer = nn.Dense(n_enc_3, n_z)

        self.dec_1 = nn.Dense(n_z, n_dec_1)
        self.dec_2 = nn.Dense(n_dec_1, n_dec_2)
        self.dec_3 = nn.Dense(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Dense(n_dec_3, n_input)
        
        self.relu = nn.ReLU()
        
        
    def construct(self, x):
    
        #mindspore.nn.ReLU
        enc_h1 = self.relu(self.enc_1(x))
        enc_h2 = self.relu(self.enc_2(enc_h1))
        enc_h3 = self.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = self.relu(self.dec_1(z))
        dec_h2 = self.relu(self.dec_2(dec_h1))
        dec_h3 = self.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z
#AE结束


class SDCN(nn.Cell):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        
        #使用load_checkpoint 与 load_param_into_net
        #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        print("loading......")
        self.ae_params = load_checkpoint(args.pretrain_path)
        load_param_into_net(self.ae, self.ae_params, strict_load=True)
        print("ae net is loaded")
        
        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        #与之前相同的初始化方法
        #self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        #torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.cluster_layer = ms.Parameter(ms.Tensor(shape = (n_clusters, n_z),dtype = ms.float32,init = XavierNormal(1)))

        # degree
        self.v = v
        
    def construct(self, x, adj1,adj2,adj3):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj1,adj2,adj3)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj1,adj2,adj3)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj1,adj2,adj3)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj1,adj2,adj3)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj1,adj2,adj3, active=False)
        #mindspore.nn.Softmax
        #predict = F.softmax(h, dim=1)
        softmax = nn.Softmax(axis = 1)
        predict = softmax(h)

        # Dual Self-supervised Module
        #q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        #q = q.pow((self.v + 1.0) / 2.0)
        #q = (q.t() / torch.sum(q, 1)).t()
        Sum = ops.ReduceSum()
        Pow = ops.Pow()
        ExpandDims = ops.ExpandDims()
        z_afterEx = ExpandDims(z,1)
        q = 1.0 / (1.0 + Sum(Pow(z_afterEx - self.cluster_layer, 2), 2) / self.v)
        q = Pow(q,(self.v + 1.0) / 2.0)
        q = (q.T / Sum(q, 1)).T

        return x_bar, q, predict, z
        
        
def target_distribution(q):
    Sum = ops.ReduceSum()
    Pow = ops.Pow()
    q_afterSum = Sum(q,0)
    q_afterPow = Pow(q,2)
    weight = q_afterPow / q_afterSum
    weight_afterSum = Sum(weight,1)
    return (weight.T / weight_afterSum).T

class WithLossCell(nn.Cell):
    def __init__(self,net,auto_prefix=False):
        super(WithLossCell,self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.kldivloss = ops.KLDivLoss()
        #self.kldivloss = KLDivLoss()
        self.mseloss = nn.MSELoss()
        self.Log = ops.Log()
        
        
    def construct(self,data,adj1,adj2,adj3,p):
        x_bar,q,pred,_ = self.net(data,adj1,adj2,adj3)
        
        kl_loss = self.kldivloss(self.Log(q),p)
        ce_loss = self.kldivloss(self.Log(pred),p)
        re_loss = self.mseloss(x_bar,data)
        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        
        return loss
        
class TrainOneStepCell(nn.Cell):
    def __init__(self,netT,optim,sens=1.0,auto_prefix = False):
        super(TrainOneStepCell,self).__init__(auto_prefix=auto_prefix)
        self.netloss = netT
        
        self.netloss.set_grad()
        #测试--------------------------------------------------------------------------------------------------------------------
        self.netloss.net.ae.set_grad(False)
        
        self.weights = ParameterTuple(netT.trainable_params())
        
        self.optimizer = optim
        self.grad = ops.GradOperation(get_by_list = True,sens_param = True)
        self.sens = sens
        
    def set_sens(self,value):
        self.sens = value
        
    def construct(self,data,adj1,adj2,adj3,p):
        weights = self.weights
        loss = self.netloss(data,adj1,adj2,adj3,p)
        sens = ops.Fill()(ops.DType()(loss),ops.Shape()(loss),self.sens)
        grads = self.grad(self.netloss,weights)(data,adj1,adj2,adj3,p,sens)
        return ops.depend(loss,self.optimizer(grads))

#其上已经全部转换为mindspore 其下需要详细自定义逐步实现训练循环

def train_sdcn(dataset):
    net = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0)
    print(net)

    #optimizer = Adam(net.parameters(), lr=args.lr)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    
    #output = spmul(adj.indices,adj.values,adj.dense_shape, support)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++附加修改
    #可以直接不变到稠密矩阵，传参时传分解后的三个矩阵
    #sparse_to_dense = ops.SparseToDense()
    #adj = sparse_to_dense(adj.indices,adj.values,adj.dense_shape)
    adj1 = adj.indices
    adj2 = adj.values
    adj3 = adj.dense_shape
    
    
    
    #不需要这一句话
    #adj = adj.cuda()

    # cluster parameter initiate
    #data = torch.Tensor(dataset.x)
    data = ms.Tensor(dataset.x,dtype = ms.float32)
    y = dataset.y
    
    #意思为此部分不计算梯度  mindspore中可以直接使用语句 Cell.set_grad(requires_grad=False) 说明此部分不使用梯度
    """
    with torch.no_grad():
        _, _, _, _, z = net.ae(data)
    """
    #不计算梯度
    net.ae.set_grad(False)
    _, _, _, _, z = net.ae(data)
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.asnumpy())
    y_pred_last = y_pred
    net.cluster_layer.set_data(ms.Tensor(kmeans.cluster_centers_))
    eva(y, y_pred, 'pae')
    
    
    network = WithLossCell(net)
    network = TrainOneStepCell(network,optimizer)
    net.set_train()
    
    
    #此部分需要在withloss实现
    #修改两个函数 WithLossCell TrainOneStepCell
    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
        
            #出错，不支持传入稀疏矩阵!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            
            
            
            _, tmp_q, pred, _ = net(data, adj1,adj2,adj3)
            #tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            #numpy   .asnumpy()
            res1 = tmp_q.asnumpy().argmax(1)       #Q
            res2 = pred.asnumpy().argmax(1)   #Z
            res3 = p.asnumpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')
        
        output = network(data,adj1,adj2,adj3,p)
        
        print("epoch: {0}/200 , losses: {1}".format(epoch,output.asnumpy(),flush=True))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='usps')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='ckpt')
    args = parser.parse_args()

    args.pretrain_path = 'data/{}.ckpt'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703


    print(args)
    train_sdcn(dataset)













