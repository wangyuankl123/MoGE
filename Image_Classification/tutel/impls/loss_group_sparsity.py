# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from .losses import load_importance_loss


def cal_group_sparsity_gaussian_weight(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:

    assert sigma > 0, "sigma must larger than 0"

    inv_sigma_sq = -0.5 / (sigma ** 2)
    center = window_size // 2

    coords = torch.arange(window_size, device=device) - center
    dist_sq = coords[:, None] ** 2 + coords ** 2  # [W, W]

    gaussian_weights = torch.exp(dist_sq * inv_sigma_sq)
    gaussian_weights /= gaussian_weights.sum()  # [window_size, window_size]

    return gaussian_weights


def load_importance_sparsity_loss(
        scores_wo_noise: torch.Tensor,
        topk_logits: torch.Tensor,
        num_global_experts: int,
        gate_noise: float,
        map_size: tuple,
        window_size: int,
        stride: int,
        gs_loss_weight: float,
        gs_sigma: float
) -> torch.Tensor:

    gaussian_weights_flat = cal_group_sparsity_gaussian_weight(window_size, gs_sigma, scores_wo_noise.device)
    gaussian_weights_flat = gaussian_weights_flat.unsqueeze(0).unsqueeze(0)
    gaussian_weights_flat = nn.Parameter(data=gaussian_weights_flat, requires_grad=False)

    def group_sparsity_loss(scores_wo_noise: torch.Tensor) -> torch.Tensor:
        feature_map = scores_wo_noise.view(scores_wo_noise.size(0), map_size[0], map_size[1])

        squared_feature_map = feature_map ** 2  # [batch_size, map_size[0], map_size[1]]
        squared_feature_map = squared_feature_map.unsqueeze(1)  # [batch_size, 1, map_size[0], map_size[1]]

        sum_weighted = F.conv2d(squared_feature_map, gaussian_weights_flat, bias=None, stride=stride, padding=0,)
        sum_weighted = sum_weighted.view(sum_weighted.size(0), -1)

        sqrt_sum = torch.sqrt(sum_weighted + 1e-8)
        l_group_sparsity = sqrt_sum.sum()

        l_group_sparsity = l_group_sparsity / window_size / window_size / sum_weighted.size(1)

        return l_group_sparsity * gs_loss_weight

    l_load_importance = load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    l_group_sparsity = group_sparsity_loss(scores_wo_noise)

    return l_load_importance + l_group_sparsity