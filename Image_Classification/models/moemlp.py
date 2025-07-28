
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

from tutel import moe as tutel_moe


class MoEMlp(nn.Module):
    def __init__(self, in_features, hidden_features, num_local_experts, top_value, capacity_factor=1.25,
                 cosine_router=False, normalize_gate=False, use_bpr=True, is_gshard_loss=True, is_group_sparsity_loss=False,
                 gs_loss_weight=None, gs_map_size=None, gs_window_size=None, gs_stride=None, gs_padding_mode=None,
                 gate_noise=1.0, cosine_router_dim=256, cosine_router_init_t=0.5, moe_drop=0.0, init_std=0.02,
                 gs_loss_mode='conv', mlp_fc2_bias=True, aux_loss_weight=0.0):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.use_bpr = use_bpr
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias
        self.aux_loss_weight = aux_loss_weight

        self._dropout = nn.Dropout(p=moe_drop)

        _gate_type = {'type': 'cosine_top' if cosine_router else 'top',
                      'k': top_value, 'capacity_factor': capacity_factor,
                      'gate_noise': gate_noise, 'fp32_gate': True}
        if cosine_router:
            _gate_type['proj_dim'] = cosine_router_dim
            _gate_type['init_t'] = cosine_router_init_t
        self._moe_layer = tutel_moe.moe_layer(
            gate_type=_gate_type,
            model_dim=in_features,
            experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_features,
                     'activation_fn': lambda x: self._dropout(F.gelu(x))},
            scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True),
            seeds=(1, 1, 1),
            batch_prioritized_routing=use_bpr,
            normalize_gate=normalize_gate,
            is_gshard_loss=is_gshard_loss,
            is_group_sparsity_loss=is_group_sparsity_loss,
            gs_loss_weight=gs_loss_weight, gs_map_size=gs_map_size, gs_window_size=gs_window_size, gs_stride=gs_stride, gs_padding_mode=gs_padding_mode, gs_loss_mode=gs_loss_mode
        )
        if not self.mlp_fc2_bias:
            self._moe_layer.experts.batched_fc2_bias.requires_grad = False

    def forward(self, x, sigma=0):
        x = self._moe_layer(x, sigma=sigma)

        assert self.aux_loss_weight > 0, 'aux_loss_weight should be greater than 0'

        return x, x.l_aux * self.aux_loss_weight

    def extra_repr(self) -> str:
        return f'Param count for MoE, ' \
               f'in_features = {self.in_features}, hidden_features = {self.hidden_features}, ' \
               f'num_local_experts = {self.num_local_experts}, top_value = {self.top_value}, ' \
               f'cosine_router={self.cosine_router} normalize_gate={self.normalize_gate}, use_bpr = {self.use_bpr}'

    def _init_weights(self):
        if hasattr(self._moe_layer, "experts"):
            trunc_normal_(self._moe_layer.experts.batched_fc1_w, std=self.init_std)
            trunc_normal_(self._moe_layer.experts.batched_fc2_w, std=self.init_std)
            nn.init.constant_(self._moe_layer.experts.batched_fc1_bias, 0)
            nn.init.constant_(self._moe_layer.experts.batched_fc2_bias, 0)
