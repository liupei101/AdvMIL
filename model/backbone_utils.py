import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.func import to_relative_coord


################################################
###              All Basic Layers            ###
################################################
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        """Attention Network with Sigmoid Gating (3 fc layers)"""
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class GAPool(nn.Module):
    """GAPool: Global Attention Pooling"""
    def __init__(self, in_dim, hid_dim, dropout=0.25):
        super(GAPool, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        """x -> out : [B, N, d] -> [B, d]"""
        emb = self.fc1(x) # [B, N, d']
        scr = self.score(x) # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        rep = self.fc2(new_emb) # [B, N, 1]
        rep = torch.transpose(rep, 2, 1)
        attn = F.softmax(rep, dim=2) # [B, 1, N]
        out = torch.matmul(attn, x).squeeze(1) # [B, 1, d]
        return out


################################################
###     Functions & Layers for DualTrans     ###
################################################
def sequence2square(x, s):
    """[B, N, C] -> [B*(N/s^2), C, s, s]"""
    size = x.size()
    assert size[1] % (s * s) == 0
    L = size[1] // (s * s)
    x = x.view(-1, s, s, size[2])
    x = x.permute(0, 3, 1, 2)
    return x, L

def square2sequence(x, L):
    """[B*L, C, s, s] -> [B, L*s*s, c]"""
    size = x.size()
    assert size[0] % L == 0
    x = x.view(size[0], size[1], -1)
    x = x.transpose(2, 1).view(size[0]//L, -1, size[1])
    return x

def posemb_sincos_2d(y, x, dim, device, dtype, temperature=10000):
    """Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py#L12"""
    # y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def compute_pe(coord: torch.Tensor, ndim=384, step=1, device='cpu', dtype=torch.float):
    assert coord.shape[0] == 1 # coord: [B, N, 2]
    coord = coord.squeeze(0)
    ncoord, ref_xy, rect = to_relative_coord(coord)
    assert rect[0] % step == 0 and rect[1] % step == 0
    y = torch.div(ncoord[:, 1], step, rounding_mode='floor')
    x = torch.div(ncoord[:, 0], step, rounding_mode='floor')
    PE = posemb_sincos_2d(y, x, ndim, device, dtype) # [N, ndim]
    PE = PE.unsqueeze(0) # [1, N, ndim]
    return PE

def make_embedding_layer(backbone:str, args):
    """
    backbone: ['gapool', 'avgpool']
    """
    if backbone == 'gapool':
        layer = GAPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    elif backbone == 'avgpool':
        layer = AVGPoolPatchEmbedding(args.in_dim, args.out_dim, args.scale, args.dw_conv, args.ksize)
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return layer

def make_transformer_layer(backbone:str, args):
    """[B, N, C] --Transformer--> [B, N, C]
    Transformer/Nystromformer: for long range dependency building.
    """
    if backbone == 'Transformer':
        patch_encoder_layer = nn.TransformerEncoderLayer(
            args.d_model, args.nhead, dim_feedforward=args.d_model, 
            dropout=args.dropout, activation='relu', batch_first=True
        )
        patch_transformer = nn.TransformerEncoder(patch_encoder_layer, num_layers=args.num_layers)
    elif backbone == 'Identity':
        patch_transformer = nn.Identity()
    else:
        raise NotImplementedError(f'{backbone} has not implemented.')
    return patch_transformer

class AVGPoolPatchEmbedding(nn.Module):
    """head layer (FC/Conv2D) + pooling Layer (avg pooling) for patch embedding.
    (1) ksize = 1 -> head layer = FC; (2) ksize = 3 -> head layer = Conv2D
    Patch data with shape of [B, N, C]
    if scale = 1, then apply Conv2d with stride=1: [B, N, C] -> [B, C, N] --conv1d--> [B, C', N]
    elif scale = 2/4, then apply Conv2d with stride=2: [B, N, C] -> [B*(N/s^2), C, s, s] --conv2d--> [B*(N/s^2), C, 1, 1] -> [B, N/s^2, C]
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv=False, ksize=3, stride=1):
        super(AVGPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        self.stride = stride
        if scale == 4:
            # Conv2D on the grid of 4 x 4: stride=2 + ksize=3 or stride=1 + ksize=1/3
            assert (stride == 2 and ksize == 3) or (stride == 1 and (ksize == 1 or ksize == 3)), \
                'Invalid stride or kernel_size when scale=4'
            if not dw_conv:
                self.conv = nn.Conv2d(in_dim, out_dim, ksize, stride, padding=(ksize-1)//2)
            else:
                self.conv = None
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise NotImplementedError()

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        print(f"Patch Embedding Layer with emb = {'FC' if ksize == 1 else 'Conv'}, pooling = AvgPool.")

    def forward(self, x):
        """x: [B, N ,C]"""
        x, L = sequence2square(x, self.scale) # [B*N/16, C, 4, 4]
        x = self.conv(x) # [B*N/16, C, 4/s, 4/s]
        x = square2sequence(x, L) # [B, N/(s*s), C]
        x = self.norm(x)
        x = self.act(x)
        x, L = sequence2square(x, self.scale//self.stride) # [B*N/16, C, 4/s, 4/s]
        x = self.pool(x) # [B*N/16, C, 1, 1]
        x = square2sequence(x, L) # [B, N/16, C]
        return x


class GAPoolPatchEmbedding(nn.Module):
    """head layer (FC/Conv2D) + pooling Layer (global-attention pooling) for patch embedding.
    (1) ksize = 1 -> head layer = FC; (2) ksize = 3 -> head layer = Conv2D
    Global Attention Pooling for patch data with shape of [B, N, C]. [B, N, C] -> [B, N/(scale^2), C']
    """
    def __init__(self, in_dim, out_dim, scale:int=4, dw_conv:bool=False, ksize=3):
        super(GAPoolPatchEmbedding, self).__init__()
        assert scale == 4, 'It only supports for scale = 4'
        assert ksize == 1 or ksize == 3, 'It only supports for ksize = 1 or 3'
        self.scale = scale
        if not dw_conv:
            self.conv = nn.Conv2d(in_dim, out_dim, ksize, 1, padding=(ksize-1)//2)
        else:
            self.conv = None
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.pool = GAPool(out_dim, out_dim, 0.0)
        print(f"Patch Embedding Layer with emb = {'FC' if ksize == 1 else 'Conv'}, pooling = GAPool.")

    def forward(self, x):
        # conv2d (strid=1) embedding (spatial continuity)
        x, L = sequence2square(x, self.scale) # [B*N/(s^2), C, s, s]
        x = self.conv(x) # [B*N/(s^2), C, s, s]
        x = square2sequence(x, L) # [B, N, C]
        x = self.norm(x)
        x = self.act(x)
        # gapool
        sz = x.size() # [B, N, C]
        x = x.view(-1, self.scale*self.scale, sz[2]) # [B*N/(scale^2), scale*scale, C]
        x = self.pool(x) # [B*N/(scale^2), C]
        x = x.view(sz[0], -1, sz[2]) # [B, N/(scale^2), C]
        return x
