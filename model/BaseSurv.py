"""
Survival Analysis for baseline traditional (non-adversarial) models.
"""
import torch
import torch.nn as nn

from .model_utils import make_noise_mlp_layer


class SurvNet(nn.Module):
    """Geneneral Survival Analysis Model
    (1) out_scale = 'sigmoid', NLL-based networks, optimized by a NLL loss.
    NLLNet = Backbone + NLL prediction head.
    Prediction is the probs ([0, 1]) at time points (called hazard function).
    (2) out_scale = 'none', Cox-based networks, optimized by a PLE loss. 
    CoxNet = Backbone + Cox prediction head.
    Prediction is proportional hazard.
    (3) out_scale = 'none', Regression networks, optimized by a L1 loss. 
    RegNet = Backbone + Regression prediction head.
    Prediction is continous time.
    """
    def __init__(self, dim_in, dim_out, backbone:nn.Module, hops=1, norm=False, dropout=0.25, out_scale='none'):
        super(SurvNet, self).__init__()
        self.backbone = backbone
        # won't involve any noise in the forward, only used for keeping the same form as Generator.
        noise = [0] * (1 + hops)
        mlps = make_noise_mlp_layer(dim_in, dim_out, noise, hops=hops, norm=norm, dropout=dropout)
        if out_scale == 'sigmoid':
            self.out_layer = nn.Sequential(*mlps, nn.Sigmoid())
            print("A NLL or Regression-based survival model is initialized.")
        elif out_scale == 'none':
            self.out_layer = nn.Sequential(*mlps)
            print("A Cox-based survival model is initialized.")
        else:
            pass

    def forward(self, x, x_ext):
        H = self.backbone(x, x_ext)
        out = self.out_layer(H)
        return out