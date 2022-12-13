"""
It contains the main-stream networks used for WSI prognosis modeling. 
(1) Cluster-based network baseline: DeepAttnMISL
(2) Graph-based   network baseline: PatchGCN
(3) Patch-base (Sequence-based)   : An improved ESAT (DualTrans_High_Stream)

All above is truncated version, i.e., the prediction head in network is removed.
"""
from typing import List
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch_geometric.nn import GENConv, DeepGCNLayer

from .backbone_utils import *


def Model_Zoo(mode):
    if mode == 'patch':
        return DualTrans_HS # An improved ESAT
    elif mode == 'cluster':
        return DeepAttMISL # original DeepAttMISL
    elif mode == 'graph':
        return PatchGCN # original PatchGCN
    else:
        return None

def load_backbone_param(mode, dims):
    # Default Params
    if mode == 'patch':
        param_emb = SimpleNamespace(in_dim=dims[0], out_dim=dims[1], scale=4, dw_conv=False, ksize=1) # FC Embedding
        param_tra = SimpleNamespace(d_model=dims[1], nhead=8, dropout=0.25, num_layers=1) # Transformer Encoder
        args = [dims[:3], 'avgpool', param_emb, 'Transformer', param_tra]
        kws = {'dropout': 0.25}
    elif mode == 'cluster':
        args = [dims[:3]]
        kws = {'num_clusters': 8, 'dropout': 0.25}
    elif mode == 'graph':
        args = [dims[:3]]
        kws = {'num_layers': 1, 'dropout': 0.25}
    else:
        args, kws = [], {}
    return args, kws

def load_backbone(mode, dims):
    net = Model_Zoo(mode)
    args, kws = load_backbone_param(mode, dims)
    model = net(*args, **kws)
    return model

class DeepAttMISL(nn.Module):
    """Same as the official implementation: DeepAttnMISL/blob/master/DeepAttnMISL_model.py"""
    def __init__(self, dims:List, num_clusters=8, dropout=0.25):
        super(DeepAttMISL, self).__init__()
        assert len(dims) == 3 # [1024, 384, 384]
        dim_in, dim_hid, dim_out = dims
        assert dim_hid == dim_out
        self.dim_hid = dim_hid
        self.num_clusters = num_clusters
        self.phis = nn.Sequential(*[nn.Conv2d(dim_in, dim_hid, 1), nn.ReLU()]) # It's equivalent to FC + ReLU
        self.pool1d = nn.AdaptiveAvgPool1d(1)    
        ### Cluster Attention MIL Construction (i.e., GAP layer)
        self.attention_net = nn.Sequential(*[
            nn.Linear(dim_hid, dim_hid), nn.ReLU(), nn.Dropout(dropout),
            Attn_Net_Gated(L=dim_hid, D=dim_hid, dropout=dropout, n_classes=1)])

    def forward(self, x_path, cluster_id, *args):
        if cluster_id is not None:
            cluster_id = cluster_id.detach().cpu().numpy()
        x_path = x_path.squeeze(0) # batch_size=1
        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            x_cluster_i = x_path[cluster_id==i].T.unsqueeze(0).unsqueeze(2) # [N, d] -> [1, d, 1, N]
            h_cluster_i = self.phis(x_cluster_i) # [1, d, 1, N] -> [1, d', 1, N]
            if h_cluster_i.shape[-1] == 0: # no any instance in this cluster
                h_cluster_i = torch.zeros((1, self.dim_hid, 1, 1)).cuda()
            h_cluster.append(self.pool1d(h_cluster_i.squeeze(2)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)
        # Global Attention Pooling
        A, h_path = self.attention_net(h_cluster) 
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        H = torch.mm(A, h_path)
        return H

class PatchGCN(nn.Module):
    def __init__(self, dims:List, num_layers:int=3, edge_agg:str='spatial', dropout:float=0.25):
        super(PatchGCN, self).__init__()
        assert len(dims) == 3 # [1024, 384, 384]
        dim_in, dim_hid, dim_out = dims
        self.edge_agg = edge_agg
        self.num_layers = num_layers
        self.fc = nn.Sequential(*[nn.Linear(dim_in, dim_hid), nn.ReLU(), nn.Dropout(dropout)])
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = GENConv(dim_hid, dim_hid, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(dim_hid, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=(i+1)%3)
            self.layers.append(layer)
        dim_sum = dim_hid * (1 + self.num_layers)
        self.path_phi = nn.Sequential(*[nn.Linear(dim_sum, dim_out), nn.ReLU(), nn.Dropout(dropout)])
        self.path_attention_head = Attn_Net_Gated(L=dim_out, D=dim_out, dropout=dropout, n_classes=1)

    def forward(self, x_path, *args):
        data = x_path
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_attr = None
        x = self.fc(data.x)
        x_ = x 
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        h_path = x_ # [N, dim_sum], dim_sum = dim_hid * (1 + num_layers)
        h_path = self.path_phi(h_path) 
        # Global Attention Pooling
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        H = torch.mm(F.softmax(A_path, dim=1), h_path)
        return H

class DualTrans_HS(nn.Module):
    def __init__(self, dims:List, emb_backbone:str, args_emb_backbone, 
        tra_backbone:str, args_tra_backbone, dropout:float=0.25):
        super(DualTrans_HS, self).__init__()
        assert len(dims) == 3 # dim_in, dim_hid, dim_out = [1024, 384, 384]
        dim_in, dim_hid, dim_out = dims
        assert dim_hid == dim_out
        assert emb_backbone in ['avgpool', 'gapool']
        assert tra_backbone in ['Transformer', 'Identity']
        self.patch_embedding_layer = make_embedding_layer(emb_backbone, args_emb_backbone)
        self.dim_hid = dim_hid
        self.patch_encoder_layer = make_transformer_layer(tra_backbone, args_tra_backbone)
        self.pool = GAPool(dim_out, dim_out)

    def forward(self, x, coord, *args):
        """x: [B, N, d], coord: the coordinates after discretization if not None"""
        patch_emb = self.patch_embedding_layer(x)
        if coord is not None:
            PE = compute_pe(coord, ndim=self.dim_hid, device=x.device, dtype=x.dtype)
            patch_emb += PE
        patch_feat = self.patch_encoder_layer(patch_emb)
        H = self.pool(patch_feat)
        return H
