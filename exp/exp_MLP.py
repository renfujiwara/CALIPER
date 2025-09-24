from typing import Optional

import torch
from torch import nn, Tensor
from exp_forecasting.exp_basic import Exp_Basic

    
class Model(nn.Module):
    def __init__(self, configs, bias=True, feature_encode_dim=2):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  #L 
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  #H 
        self.hidden_dim=configs.d_model
        self.res_hidden=configs.d_model 
        self.encoder_num=configs.e_layers
        self.decoder_num=configs.d_layers
        self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = configs.c_out
        self.enc_in    = configs.enc_in 
        dropout=configs.dropout
        
        self.fc1 = nn.Linear(self.seq_len*self.enc_in, self.hidden_dim, bias=bias)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.hidden_dim, self.decode_dim*self.pred_len, bias=bias)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):        
        B, L, C = x_enc.shape
        x = x_enc.reshape(B, L * C)             
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h)
        y = self.fc2(h)                     
        y = y.view(B, self.pred_len, self.decode_dim)
        return y

class net(nn.Module):
    def __init__(self, configs, device='cuda', max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        configs.device=device
        self.model = Model(configs)
        self.to(device)
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return x
    
    def store_grad(self):
        pass

class Exp_STSLib(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.input_channels_dim = args.enc_in
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor', 'encoder', 'inv']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.net_class = net
        self.model_init(args)
        self.augmenter = None
        self.aug = args.aug