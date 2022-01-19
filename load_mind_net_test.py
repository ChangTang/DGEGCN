import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import Model, load_checkpoint, save_checkpoint, load_param_into_net

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
    def construct(self, x):
    
        relu = nn.ReLU()
        #mindspore.nn.ReLU
        enc_h1 = relu(self.enc_1(x))
        enc_h2 = relu(self.enc_2(enc_h1))
        enc_h3 = relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = relu(self.dec_1(z))
        dec_h2 = relu(self.dec_2(dec_h1))
        dec_h3 = relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


net = AE(500, 500, 2000, 2000, 500, 500,256,10)
params = load_checkpoint("data/usps.ckpt")
load_param_into_net(net, params, strict_load=True)
print("==========strict load mode===========")







