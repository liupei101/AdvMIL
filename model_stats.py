"""
This script is used to evaluate model's computation efficiency.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace
import argparse
from thop import profile, clever_format
from ptflops import get_model_complexity_info

from model import GANSurv
from model.backbone import load_backbone
from utils.func import sparse_key, sparse_str
from dataset.utils import prepare_dataset


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('# parameters: %d' % num_params)
    print('# trainable parameters: %d' % num_params_train)

def model_setup(args):
    # load model configuration: args.m, args.b
    print("[info] Got model setting:", args)
    print('[info] {}: initializing model...'.format(args.m))

    if args.m == 'D':
        dict_netD = {
            'disc_netx_in_dim': 1024,
            'disc_netx_out_dim': 128,
            'disc_netx_ksize': 1,
            'disc_netx_backbone': 'avgpool',
            'disc_netx_dropout': 0.25,
            'disc_nety_in_dim': 1,
            'disc_nety_hid_dims': '64-128',
            'disc_nety_norm': False,
            'disc_nety_dropout': 0.0
        }
        disc_x_args = SimpleNamespace(**sparse_key(dict_netD, prefixes='disc_netx'))
        disc_y_args = SimpleNamespace(**sparse_key(dict_netD, prefixes='disc_nety'))
        disc_y_args.hid_dims = sparse_str(disc_y_args.hid_dims)
        model = GANSurv.PrjDiscriminator(disc_x_args, disc_y_args, prj_path='x', inner_product='bag')
    elif args.m == 'G':
        if args.b == 'cluster' or args.b == 'patch':
            cfg = {
                'bcb_dims': '1024-384-384', # input dim -> hidden dim -> embedding dim
                'gen_dims': '384-1', # embedding dim -> out dim
                'gen_noi_noise': '1-1', # mlp: 384 ->- 384/2 ->- 1
                'gen_noi_noise_dist': 'uniform',  # gaussian / uniform
                'gen_noi_hops': 1,
                'gen_norm': False,
                'gen_dropout': 0.6,
                'gen_out_scale': 'sigmoid' # sigmoid / exp
            }
        elif args.b == 'graph':
            cfg = {
                'bcb_dims': '1024-128-128', # input dim -> hidden dim -> embedding dim
                'gen_dims': '128-1', # embedding dim -> out dim
                'gen_noi_noise': '1-1', # mlp: 384 ->- 384/2 ->- 1
                'gen_noi_noise_dist': 'uniform',  # gaussian / uniform
                'gen_noi_hops': 1,
                'gen_norm': False,
                'gen_dropout': 0.6,
                'gen_out_scale': 'sigmoid' # sigmoid / exp
            }
        else:
            pass

        backbone_dims = sparse_str(cfg['bcb_dims'])
        backbone = load_backbone(args.b, backbone_dims)
        dim_in, dim_out = sparse_str(cfg['gen_dims'])
        args_noise = SimpleNamespace(**sparse_key(cfg, prefixes='gen_noi'))
        args_noise.noise = sparse_str(args_noise.noise)
        model = GANSurv.Generator(dim_in, dim_out, backbone, args_noise, 
            cfg['gen_norm'], cfg['gen_dropout'], cfg['gen_out_scale'])
    else:
        raise ValueError('check the arguments your passed.')
    
    print('[info] {}: finished model loading'.format(args.m))

    return model

def D_input_constructor(args=((1, 200, 1024), 'tuple')):
    bs, N, d = args[0]
    ret_format = args[1]
    assert ret_format in ['tuple', 'mapping']
    X = torch.randn(bs, 16*N, d).cuda() # with N regions, each of which has 16 small patches
    t = torch.randn(bs, 1).cuda() # scalar time
    if ret_format == 'tuple':
        return X, t
    else:
        return {'x': X, 't': t}

def G_input_constructor(args=('patch', 'tuple')):
    # we load real data for generator since cluster/graph backbones must have their respective structures.
    # this patient who will be loaded is '128599' from NLST with 1 slides, 210 patches at 5x, and 3360 patches at 20x
    # It will keep the same with the data fed into D
    bcb_type = args[0]
    ret_format = args[1]
    assert ret_format in ['tuple', 'mapping']
    # input cfg
    cfg = {
        'bcb_mode': bcb_type,
        'path_patch': '/hdd/liup/data/WSI/NLST/processed/feat-x20-RN50-B-color_norm/pt_files',
        'path_graph': '/hdd/liup/data/WSI/NLST/processed/wsigraph-x20-features',
        'path_cluster': '/hdd/liup/data/WSI/NLST/processed/patch-x20-cluster8-ids',
        'path_coordx5': '/hdd/liup/data/WSI/NLST/processed/hier-x5-tiles-s256/patches',
        'path_label': '/hdd/liup/data/WSI/NLST/table/nlst_path_full.csv',
        'feat_format': 'pt',
        'time_format': 'ratio',
        'time_bins': 4,
    }
    dataset = prepare_dataset(['128599'], cfg)
    data_idx, data_x, data_y = dataset[0]
    data_x, data_x_ext = data_x[0], data_x[1]
    if bcb_type == 'graph':
        param1 = data_x_ext.cuda()
        param2 = None
        # pred = self.netG(data_x_ext, None) # data_x_ext -> GraphData if backbone=graph
    elif bcb_type == 'patch':
        data_x = data_x.unsqueeze(0)
        param1 = data_x.cuda()
        param2 = None
        # pred = self.netG(data_x, None) # skip coords if backbone=patch
    else:
        data_x = data_x.unsqueeze(0)
        param1 = data_x.cuda() # [1, N, c]
        param2 = data_x_ext.cuda() # [N, ]
        # pred = self.netG(data_x, data_x_ext) # generate pred given data_x
    if ret_format == 'tuple':
        return param1, param2
    else:
        return {'x': param1, 'x_ext': param2}


# IT IS RECOMMENDED TO USE ptflops 
# thop would ignore some parameters and MACs in nn.Transformer
# but ptflops can record them.
parser = argparse.ArgumentParser(description='Configurations for Models.')
parser.add_argument('-a', type=str, choices=['thop', 'ptflops'], default='ptflops')
parser.add_argument('-m', type=str, choices=['G', 'D'], default='D')
parser.add_argument('-b', type=str, choices=['cluster', 'graph', 'patch'], default='patch')

# python3 model_stats.py -a ptflops -m D -b patch
if __name__ == '__main__':
    args = parser.parse_args()
    model = model_setup(args)
    print_network(model)
    model = model.cuda()

    N = 210 # Patient '128599' from NLST with 1 slides, 210 patches at 5x, and 3360 patches at 20x

    if args.a == 'thop':
        if args.m == 'D':
            all_input = D_input_constructor(args=((1, N, 1024), 'tuple'))
            macs, params = profile(model, inputs=all_input)
        else:
            all_input = G_input_constructor(args=(args.b, 'tuple'))
            macs, params = profile(model, inputs=all_input)
    else:
        if args.m == 'D':
            macs, params = get_model_complexity_info(
                model, ((1, N, 1024), 'mapping'), as_strings=False,
                input_constructor=D_input_constructor, print_per_layer_stat=True, verbose=True
            )
        else:
            macs, params = get_model_complexity_info(
                model, (args.b, 'mapping'), as_strings=False,
                input_constructor=G_input_constructor, print_per_layer_stat=True, verbose=True
            )

    print("#Params: {}, #MACs: {}".format(params, macs))
    macs, params = clever_format([macs, params], "%.2f")
    print("#Params: {}, #MACs: {}".format(params, macs))
