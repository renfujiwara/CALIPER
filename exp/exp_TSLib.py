import os
from typing import Optional
from pathlib import Path
import importlib

import torch
from torch import nn, Tensor
from exp_forecasting.exp_basic import Exp_Basic

class net(nn.Module):
    def __init__(self, configs, device='cuda', max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        configs.device=device
        Model = getattr(importlib.import_module('TSLib.models.{}'.format(configs.method)), 'Model')
        self.model = Model(configs)
        self.to(device)
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):           # x: [Batch, Input length, Channel]
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
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)
    
    

