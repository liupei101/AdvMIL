import torch
import torch.nn as nn

from utils.func import generate_noise
from .model_utils import EmbedXLayer
from .model_utils import make_embedding_y_layer
from .model_utils import make_noise_mlp_layer

#######################################################################
# AdvMIL Implementation: details of generator and discriminator.
#######################################################################

class Generator(nn.Module):
    """General Generator Class. It's composed of two main parts: Backbone and NoiseLayer.
    (1) Backbone: it takes WSIs (bags) as input, and outputs WSI-level vectors (usually after GAP), denoted by H.
    (2) NoiseLayer: it takes H and N (noise vectors) as input, and outputs time predictions. It could be MLPs.
    Note that Backbone could be any existing networks for WSIs, e.g., typically, cluster-, graph-, and patch-based networks.
    For NoiseLayer, we add N into H by tensor concatenation.
    """
    def __init__(self, dim_in, dim_out, backbone:nn.Module, args_noise, norm=False, dropout=0.25, out_scale:str='sigmoid'):
        super(Generator, self).__init__()
        self.noise = args_noise.noise
        self.hops  = args_noise.hops
        self.noise_dist = 'uniform' if args_noise.noise_dist is None else args_noise.noise_dist
        assert len(self.noise) == self.hops + 1
        self.MLPs = make_noise_mlp_layer(dim_in, dim_out, self.noise, hops=self.hops, norm=norm, dropout=dropout)
        self.backbone = backbone
        self.out_scale = out_scale

    def forward(self, x, x_ext):
        H = self.backbone(x, x_ext)
        for i, layer_i in enumerate(self.MLPs):
            if self.noise[i] == 1:
                N = generate_noise(*H.size(), to_device=H.device, distribution=self.noise_dist)
                data = torch.cat([H, N], dim=1)
            else:
                data = H
            hidx = layer_i(data)
            H = hidx
        if self.out_scale == 'sigmoid':
            out = torch.sigmoid(H)
        elif self.out_scale == 'exp':
            out = torch.exp(H)
        else:
            out = H
        return out


class Discriminator(nn.Module):
    """Discriminator for pair (X, t), where X denotes WSI and t denotes time"""
    def __init__(self, args_netx, args_nety, **kws):
        super(Discriminator, self).__init__()
        self.net_pair_one = EmbedXLayer(args_netx)
        self.net_pair_two = make_embedding_y_layer(args_nety)
        dim_x, dim_y = args_netx.out_dim, args_nety.hid_dims[-1]
        self.fc = nn.Linear(dim_x + dim_y, 1)
        print('[info] Typical discriminator without projection')

    def forward(self, x, t):
        """pair (x_real, t_real)/(x_fake, t_fake)"""
        hid_t = self.net_pair_two(t) # [B, 1] -> [B, C']
        hid_x = self.net_pair_one(x) # [B, N, C] -> [B, C']
        hid_feat = torch.cat([hid_x, hid_t], dim=1)
        out = self.fc(hid_feat) # [B, 2C'] -> [B, 1]
        return out


class PrjDiscriminator(nn.Module):
    """Discriminator for pair (X, t), where X denotes WSI and t denotes time"""
    def __init__(self, args_netx, args_nety, prj_path='x', inner_product='bag'):
        super(PrjDiscriminator, self).__init__()
        assert inner_product in ['bag', 'instance']
        self.inner_product = inner_product
        self.net_pair_one = EmbedXLayer(args_netx)
        self.net_pair_two = make_embedding_y_layer(args_nety)
        dim_x, dim_y = args_netx.out_dim, args_nety.hid_dims[-1]
        self.prj_path = prj_path
        if prj_path == 'x':
            self.prj_layer = nn.Linear(dim_x, 1)
        elif prj_path == 'y':
            self.prj_layer = nn.Linear(dim_y, 1)
        else:
            self.prj_layer = None
        print('[info] Discriminator is with projection: {}'.format(self.prj_path))

    def forward(self, x, t):
        """pair (x_real, t_real)/(x_fake, t_fake)"""
        hid_t = self.net_pair_two(t) # [B, 1] -> [B, C']
        if self.inner_product == 'bag':
            hid_x = self.net_pair_one(x) # [B, N, C] -> [B, C']
            out = (hid_t * hid_x).sum(dim=-1, keepdim=True) # [B, 1]
        elif self.inner_product == 'instance':
            hid_x, emb_ins = self.net_pair_one(x, return_instance=True) # [B, C'], [B, N/16, C']
            out_ins = (emb_ins * hid_t).sum(dim=-1, keepdim=False) # [B, N/16]
            out = out_ins.mean(dim=-1, keepdim=True) # [B, 1]
        else:
            pass

        if self.prj_layer is not None:
            p = self.prj_layer(hid_x if self.prj_path == 'x' else hid_t)
            out = out + p
        return out
