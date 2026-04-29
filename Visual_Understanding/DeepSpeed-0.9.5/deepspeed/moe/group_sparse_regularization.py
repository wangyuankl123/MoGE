# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import Any, Dict


class SigmaScheduler():
    def __init__(self, sigma_min, sigma_max, n_iteration, gamma=0.2):
        assert sigma_max >= sigma_min, "sigma_max must larger or equal to sigma_min"

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_iteration = n_iteration
        self.gamma = gamma
        self.iter = 0
        self.curr_sigma = sigma_max

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_sigma(self):
        return max(self.curr_sigma, self.sigma_min)

    def step(self):
        t = min(self.iter / self.n_iteration, 1.0)

        self.curr_sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * (t ** self.gamma)
        self.iter += 1


def transform_to_2d_map(global_experts):
    height, width = 1, global_experts
    for i in range(1, int(math.sqrt(global_experts)) + 1):
        if int(global_experts / i) == global_experts / i:
            if global_experts / i - i < width - height:
                height, width = i, global_experts / i
    return height, int(width)


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
    num_experts: int,
    # filter_size: int,           # define manually
    gs_loss_weight: float = 0.01,      # define manually, this value needs to be multiplied by 100
    gs_sigma: float = 0.2,            # define manually
    stride = 1
) -> torch.Tensor:
    map_size = transform_to_2d_map(num_experts)

    if num_experts == 8:
        filter_size = 2
    elif num_experts == 16:
        filter_size = 3
    else:
        raise ValueError('Number of experts is not implemented yet')

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