# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from torch import nn


def cal_group_sparsity_gaussian_weight(filter_size: int, sigma: float, device: torch.device) -> torch.Tensor:

    assert sigma > 0, "sigma must larger than 0"

    inv_sigma_sq = -0.5 / (sigma ** 2)
    center = filter_size // 2

    coords = torch.arange(filter_size, device=device) - center
    dist_sq = coords[:, None] ** 2 + coords ** 2  # [W, W]

    gaussian_weights = torch.exp(dist_sq * inv_sigma_sq)
    gaussian_weights /= gaussian_weights.sum()  # [filter_size, filter_size]

    return gaussian_weights


def group_sparse_regularization(
    scores_wo_noise: torch.Tensor,
    map_size: tuple,
    filter_size: int,
    gs_loss_weight: float,
    gs_sigma: float,
    stride = 1
) -> torch.Tensor:

    gaussian_weights_flat = cal_group_sparsity_gaussian_weight(filter_size, gs_sigma, scores_wo_noise.device)
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

        l_group_sparsity = l_group_sparsity / filter_size / filter_size / sum_weighted.size(1)

        return l_group_sparsity * gs_loss_weight

    l_group_sparsity = group_sparsity_loss(scores_wo_noise)

    return l_group_sparsity