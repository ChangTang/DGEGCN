import numpy as np
import scipy.sparse as sp
import h5py
import mindspore as ms
from mindspore.common.initializer import Initializer
import math

#无torch
def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_mindspore_sparse_tensor(adj)

    return adj

#无torch
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

#稠密矩阵转换 有  mindspore.SparseTensor
def sparse_mx_to_mindspore_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    indices = ms.Tensor.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #values = torch.from_numpy(sparse_mx.data)
    values = ms.Tensor.from_numpy(sparse_mx.data)
    #shape = torch.Size(sparse_mx.shape)
    shape = sparse_mx.shape
    #print("indices shape : ",indices.shape)
    return ms.SparseTensor(indices.T, values, shape)


#设置自定义数据集    DatasetGenerator
"""
class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
"""

class load_data:
    def __init__(self,dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,index):
        return ms.Tensor.from_numpy(np.array(self.x[idx])),\
               ms.Tensor.from_numpy(np.array(self.y[idx])),\
               ms.Tensor.from_numpy(np.array(idx))
        

def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out
    
def _assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr
#自定义初始化方式
class XavierNormal(Initializer):
    r"""
    Initialize the array with xavier uniform algorithm, and from a uniform distribution collect samples within
    U[-boundary, boundary] The boundary is defined as:

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    - where :math:`n_{in}` is the number of input units in the weight tensor.
    - where :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.

    Returns:
        Array, assigned array.
    """
    def __init__(self, gain=1):
        super(XavierNormal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        boundary = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.uniform(-boundary, boundary, arr.shape)

        _assignment(arr, data)

        
        
        
        
        
        
