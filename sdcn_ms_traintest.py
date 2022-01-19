def train_sdcn(dataset):
    net = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0)
    print(net)

    #optimizer = Adam(net.parameters(), lr=args.lr)
    optimizer = nn.Adam(net.trainable_params(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    #不需要这一句话
    #adj = adj.cuda()

    # cluster parameter initiate
    #data = torch.Tensor(dataset.x)
    data = ms.Tensor(dataset.x)
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
    y_pred = kmeans.fit_predict(z.data.asnumpy())
    y_pred_last = y_pred
    net.cluster_layer.data = ms.Tensor(kmeans.cluster_centers_)
    eva(y, y_pred, 'pae')




    #此部分需要在withloss实现
    
    #修改两个函数 WithLossCell TrainOneStepCell
    for epoch in range(200):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = net(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            #numpy   .asnumpy()
            res1 = tmp_q.asnumpy().argmax(1)       #Q
            res2 = pred.data.asnumpy().argmax(1)   #Z
            res3 = p.data.asnumpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _ = net(data, adj)
        
        kl_loss = F.kl_div(q.log(), p)
        ce_loss = F.kl_div(pred.log(), p)
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class WithLossCell(nn.Cell):
    def __init__(self,net,auto_prefix=False):
        super(WithLossCell,self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.kldivloss = ops.KLDivLoss()
        self.mseloss = nn.MSELoss()
        self.Log = ops.Log()
        
        
    def construct(self,data,adj,p):
        x_bar,q,pred,_ = self.net(data,adj)
        
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
        self.netloss.net.ae.set_grad(False)
        
        self.weights = ParameterTuple(netT.trainable_params())
        
        self.optimizer = optim
        self.grad = ops.GradOperation(get_by_list = True,sens_param = True)
        self.sens = sens
        
    def set_sens(self,value):
        self.sens = value
        
    def construct(self,data,adj,p):
        weights = self.weights
        loss = self.netloss(data,adj,p)
        sens = ops.Fill()(ops.DType()(loss),ops.Shape()(loss),self.sens)
        grads = self.grad(self.netloss,weights)(data,adj,p,sens)
        return ops.depend(loss,self.optimizer(grads))
        












        
        
