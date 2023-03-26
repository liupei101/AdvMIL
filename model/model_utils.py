import torch
import torch.nn as nn
import math

from utils.func import generate_noise
from .backbone_utils import make_embedding_layer, GAPool


################################################
###   Functions for Network Initialization   ###
################################################
@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

@torch.no_grad()
def general_init_weight(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

def init_pytorch_defaults(m, version='041'):
    '''
    copied from AMDIM repo: https://github.com/Philip-Bachman/amdim-public/
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1)) * 0.5
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False

################################################
###      Functions/Layers for Generator      ###
################################################
def get_hop_dims(d, hops):
    res, cur_d = [], d
    for i in range(hops):
        cur_d = cur_d // 2
        if cur_d > 1:
            res.append(cur_d)
        else:
            break
    return res

def make_noise_mlp_layer(in_dim:int, out_dim:int, noise, hops:int=1, norm:bool=False, dropout:float=0.25):
    mlp_hid_dims = get_hop_dims(in_dim, hops)
    mlp_num_layers = len(mlp_hid_dims) + 1

    mlp_in_dims  = [in_dim] + mlp_hid_dims
    mlp_out_dims = mlp_hid_dims + [out_dim]

    MLPs = nn.ModuleList()
    for i in range(mlp_num_layers):
        add_dim  = 0 if noise[i] != 1 else mlp_in_dims[i]
        cur_din  = mlp_in_dims[i] + add_dim
        cur_dout = mlp_out_dims[i]
        if i == mlp_num_layers - 1: # The last layer
            cur_layer = nn.Sequential(nn.Linear(cur_din, cur_dout))
        else:
            cur_layer = make_mlp_layer(cur_din, cur_dout, norm, dropout)
        MLPs.append(cur_layer)
    return MLPs

class NoisePerturbationLayer(nn.Module):
    """Add noise to input by concating the data of same size to input.
    [B, N, C] -> [B, N, 2C] --Linear-LayerNorm-Dropout--> [B, N, C]
    """
    def __init__(self, in_channels, noise_dist='uniform', norm=True, dropout=0.25):
        super(NoisePerturbationLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2*in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.Dropout(dropout),
        )
        self.noise_dist = noise_dist

    def forward(self, x):
        noise = generate_noise(*x.size(), to_device=x.device, distribution=self.noise_dist)
        data = torch.cat([x, noise], dim=-1)
        out = self.layer(data)
        return out

################################################
###    Functions/Layers for Discriminator    ###
################################################
def make_efficient_mlp_layer(dim, layer_norm=True, dropout=0.25):
    layers = [
        nn.Linear(dim, dim//2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dim//2, dim),
    ]
    if layer_norm:
        layers.insert(1, nn.LayerNorm(dim_out))
    return nn.Sequential(*layers)

def make_mlp_layer(dim_in, dim_out, layer_norm=True, dropout=0.25):
    layers = [
        nn.Linear(dim_in, dim_out),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    ]
    if layer_norm:
        layers.insert(1, nn.LayerNorm(dim_out))
    return nn.Sequential(*layers)

def make_embedding_y_layer(args):
    """[B, 1] -> [B, C']"""
    in_dim, hid_dims = args.in_dim, args.hid_dims
    layers = []
    for i in range(len(hid_dims)):
        layers.append(make_mlp_layer(in_dim, hid_dims[i], args.norm, args.dropout))
        in_dim = hid_dims[i]
    net_y = nn.Sequential(*layers)
    return net_y

class EmbedXLayer(nn.Module):
    """[B, N, C] -embedding-> [B, N', C'] -gap-> [B, C']"""
    def __init__(self, args):
        super(EmbedXLayer, self).__init__()
        out_dim = args.out_dim
        # add default params for this embedding layer
        args.scale = 4
        args.dw_conv = False
        self.embedding = make_embedding_layer(args.backbone, args)
        # extra layers
        self.fc1 = make_efficient_mlp_layer(out_dim, False, args.dropout)
        self.pool = GAPool(out_dim, out_dim, args.dropout)
        self.fc2 = make_efficient_mlp_layer(out_dim, False, args.dropout)

    def forward(self, x, return_instance=False):
        emb_ins = self.embedding(x) # [B, N, C] -> [B, N/16, C']
        fc_ins = self.fc1(emb_ins) # [B, N/16, C']
        emb_bag = self.pool(fc_ins) # [B, C']
        fc_bag = self.fc2(emb_bag) # [B, C']
        if return_instance:
            return fc_bag, fc_ins
            # return fc_bag, emb_ins
        return fc_bag
