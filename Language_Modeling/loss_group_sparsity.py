# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn


def transform_to_2d_map(global_experts):
    height, width = 1, global_experts
    for i in range(1, int(math.sqrt(global_experts)) + 1):
        if int(global_experts / i) == global_experts / i:
            if global_experts / i - i < width - height:
                height, width = i, global_experts / i
    return height, int(width)


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
        gs_loss_weight = 0,
        gs_sigma = 0,
        window_size = 3,
        stride = 1,
) -> torch.Tensor:

    assert gs_loss_weight > 0, "group sparse loss weight should larger than 0"
    assert gs_sigma > 0, "group sparse sigma should larger than 0"
    assert scores_wo_noise.size(1) <= 64, "Expert quantity error"
    map_size = transform_to_2d_map(scores_wo_noise.size(1))

    gaussian_weights_flat = cal_group_sparsity_gaussian_weight(window_size, gs_sigma, scores_wo_noise.device)
    gaussian_weights_flat = gaussian_weights_flat.unsqueeze(0).unsqueeze(0)
    gaussian_weights_flat = nn.Parameter(data=gaussian_weights_flat, requires_grad=False)

    def group_sparsity_loss(scores_wo_noise: torch.Tensor) -> torch.Tensor:
        feature_map = scores_wo_noise.view(scores_wo_noise.size(0), map_size[0], map_size[1])

        squared_feature_map = feature_map ** 2  # [batch_size, map_size[0], map_size[1]]
        squared_feature_map = squared_feature_map.unsqueeze(1)  # [batch_size, 1, map_size[0], map_size[1]]

        sum_weighted = F.conv2d(squared_feature_map, gaussian_weights_flat, bias=None, stride=stride, padding=0,)
        # print(sum_weighted.shape)
        sum_weighted = sum_weighted.view(sum_weighted.size(0), -1)

        sqrt_sum = torch.sqrt(sum_weighted + 1e-8)
        l_group_sparsity = sqrt_sum.sum()

        l_group_sparsity = l_group_sparsity / sum_weighted.size(1)

        return l_group_sparsity * gs_loss_weight

    l_group_sparsity = group_sparsity_loss(scores_wo_noise)

    return l_group_sparsity