import torch
import torch.distributed as dist


def replace_swin_ffn_with_moe(config, model):
    from models.moemlp import MoEMlp

    layer_indices = config.MODEL.SWIN_UPCYCLING.MOE_BLOCKS
    for layer_index, block_index in layer_indices:
        target_module = model.layers[layer_index].blocks[block_index]

        dense_fc1_weight = target_module.mlp.fc1.weight.data
        dense_fc2_weight = target_module.mlp.fc2.weight.data
        dense_fc1_bias = target_module.mlp.fc1.bias.data
        dense_fc2_bias = target_module.mlp.fc2.bias.data

        target_module.mlp = MoEMlp(in_features=target_module.mlp.fc1.in_features,
                                   hidden_features=int(target_module.mlp.fc1.in_features * config.MODEL.SWIN_UPCYCLING.MLP_RATIO),
                                   num_local_experts=config.MODEL.SWIN_UPCYCLING.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.SWIN_UPCYCLING.TOP_VALUE,
                                   capacity_factor=config.MODEL.SWIN_UPCYCLING.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.SWIN_UPCYCLING.COSINE_ROUTER,
                                   cosine_router_dim=config.MODEL.SWIN_UPCYCLING.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.SWIN_UPCYCLING.COSINE_ROUTER_INIT_T,
                                   normalize_gate=config.MODEL.SWIN_UPCYCLING.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.SWIN_UPCYCLING.USE_BPR,
                                   is_gshard_loss=config.MODEL.SWIN_UPCYCLING.IS_GSHARD_LOSS,
                                   aux_loss_weight=config.MODEL.SWIN_UPCYCLING.AUX_LOSS_WEIGHT,
                                   gate_noise=config.MODEL.SWIN_UPCYCLING.GATE_NOISE,
                                   moe_drop=config.MODEL.SWIN_UPCYCLING.MOE_DROP,
                                   mlp_fc2_bias=config.MODEL.SWIN_UPCYCLING.MLP_FC2_BIAS,
                                   init_std=config.MODEL.SWIN_UPCYCLING.INIT_STD,

                                   is_group_sparsity_loss=config.MODEL.SWIN_UPCYCLING.IS_GROUP_SPARSITY_LOSS,
                                   gs_loss_weight=config.MODEL.SWIN_UPCYCLING.GS_LOSS_WEIGHT,
                                   gs_map_size=config.MODEL.SWIN_UPCYCLING.GS_MAP_SIZE,
                                   gs_window_size=config.MODEL.SWIN_UPCYCLING.GS_WINDOW_SIZE,
                                   gs_stride=config.MODEL.SWIN_UPCYCLING.GS_STRIDE,
                                   gs_padding_mode=config.MODEL.SWIN_UPCYCLING.GS_PADDING_MODE,
                                   gs_loss_mode=config.MODEL.SWIN_UPCYCLING.GS_LOSS_MODE,
        )

        fused_experts_network = target_module.mlp._moe_layer.experts

        moe_fc1_weight = fused_experts_network.batched_fc1_w.data
        moe_fc2_weight = fused_experts_network.batched_fc2_w.data
        moe_fc1_bias = fused_experts_network.batched_fc1_bias.data
        moe_fc2_bias = fused_experts_network.batched_fc2_bias.data

        for i in range(moe_fc1_weight.size(0)):
            moe_fc1_weight[i, :, :] = dense_fc1_weight
            moe_fc1_bias[i, :] = dense_fc1_bias
            moe_fc2_weight[i, :, :] = dense_fc2_weight.T
            moe_fc2_bias[i, :] = dense_fc2_bias

        fused_experts_network.batched_fc1_w = torch.nn.Parameter(moe_fc1_weight)
        fused_experts_network.batched_fc2_w = torch.nn.Parameter(moe_fc2_weight)
        fused_experts_network.batched_fc1_bias = torch.nn.Parameter(moe_fc1_bias)
        fused_experts_network.batched_fc2_bias = torch.nn.Parameter(moe_fc2_bias)

    return model


def replace_vit_ffn_with_moe(config, model):
    from models.moemlp import MoEMlp

    block_indices = config.MODEL.VIT_UPCYCLING.MOE_BLOCKS
    for block_index in block_indices:
        target_module = model.blocks[block_index]

        dense_fc1_weight = target_module.mlp.fc1.weight.data
        dense_fc2_weight = target_module.mlp.fc2.weight.data
        dense_fc1_bias = target_module.mlp.fc1.bias.data
        dense_fc2_bias = target_module.mlp.fc2.bias.data

        target_module.mlp = MoEMlp(in_features=target_module.mlp.fc1.in_features,
                                   hidden_features=int(target_module.mlp.fc1.in_features * config.MODEL.VIT_UPCYCLING.MLP_RATIO),
                                   num_local_experts=config.MODEL.VIT_UPCYCLING.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.VIT_UPCYCLING.TOP_VALUE,
                                   capacity_factor=config.MODEL.VIT_UPCYCLING.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.VIT_UPCYCLING.COSINE_ROUTER,
                                   cosine_router_dim=config.MODEL.VIT_UPCYCLING.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.VIT_UPCYCLING.COSINE_ROUTER_INIT_T,
                                   normalize_gate=config.MODEL.VIT_UPCYCLING.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.VIT_UPCYCLING.USE_BPR,
                                   is_gshard_loss=config.MODEL.VIT_UPCYCLING.IS_GSHARD_LOSS,
                                   aux_loss_weight=config.MODEL.VIT_UPCYCLING.AUX_LOSS_WEIGHT,
                                   gate_noise=config.MODEL.VIT_UPCYCLING.GATE_NOISE,
                                   moe_drop=config.MODEL.VIT_UPCYCLING.MOE_DROP,
                                   mlp_fc2_bias=config.MODEL.VIT_UPCYCLING.MLP_FC2_BIAS,
                                   init_std=config.MODEL.VIT_UPCYCLING.INIT_STD,

                                   is_group_sparsity_loss=config.MODEL.VIT_UPCYCLING.IS_GROUP_SPARSITY_LOSS,
                                   gs_loss_weight=config.MODEL.VIT_UPCYCLING.GS_LOSS_WEIGHT,
                                   gs_map_size=config.MODEL.VIT_UPCYCLING.GS_MAP_SIZE,
                                   gs_window_size=config.MODEL.VIT_UPCYCLING.GS_WINDOW_SIZE,
                                   gs_stride=config.MODEL.VIT_UPCYCLING.GS_STRIDE,
                                   gs_padding_mode=config.MODEL.VIT_UPCYCLING.GS_PADDING_MODE,
                                   gs_loss_mode=config.MODEL.VIT_UPCYCLING.GS_LOSS_MODE,
        )

        fused_experts_network = target_module.mlp._moe_layer.experts

        moe_fc1_weight = fused_experts_network.batched_fc1_w.data
        moe_fc2_weight = fused_experts_network.batched_fc2_w.data
        moe_fc1_bias = fused_experts_network.batched_fc1_bias.data
        moe_fc2_bias = fused_experts_network.batched_fc2_bias.data

        for i in range(moe_fc1_weight.size(0)):
            moe_fc1_weight[i, :, :] = dense_fc1_weight
            moe_fc1_bias[i, :] = dense_fc1_bias
            moe_fc2_weight[i, :, :] = dense_fc2_weight.T
            moe_fc2_bias[i, :] = dense_fc2_bias

        fused_experts_network.batched_fc1_w = torch.nn.Parameter(moe_fc1_weight)
        fused_experts_network.batched_fc2_w = torch.nn.Parameter(moe_fc2_weight)
        fused_experts_network.batched_fc1_bias = torch.nn.Parameter(moe_fc1_bias)
        fused_experts_network.batched_fc2_bias = torch.nn.Parameter(moe_fc2_bias)

    return model