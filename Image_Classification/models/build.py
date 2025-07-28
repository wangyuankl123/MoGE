from .swin_transformer import SwinTransformer
from .swin_transformer_upcycle import SwinTransformerUpcycle
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_transformer_moge import SwinTransformerMoGE
from .vision_transformer import VisionTransformer
from .vision_transformer_moe import VisionTransformerMoE
from .vision_transformer_moge import VisionTransformerMoGE
from .vision_transformer_upcycle import VisionTransformerUpcycle


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swin_upcycle':
        model = SwinTransformerUpcycle(img_size=config.DATA.IMG_SIZE,
                                       patch_size=config.MODEL.SWIN_UPCYCLING.PATCH_SIZE,
                                       in_chans=config.MODEL.SWIN_UPCYCLING.IN_CHANS,
                                       num_classes=config.MODEL.NUM_CLASSES,
                                       embed_dim=config.MODEL.SWIN_UPCYCLING.EMBED_DIM,
                                       depths=config.MODEL.SWIN_UPCYCLING.DEPTHS,
                                       num_heads=config.MODEL.SWIN_UPCYCLING.NUM_HEADS,
                                       window_size=config.MODEL.SWIN_UPCYCLING.WINDOW_SIZE,
                                       mlp_ratio=config.MODEL.SWIN_UPCYCLING.MLP_RATIO,
                                       qkv_bias=config.MODEL.SWIN_UPCYCLING.QKV_BIAS,
                                       qk_scale=config.MODEL.SWIN_UPCYCLING.QK_SCALE,
                                       drop_rate=config.MODEL.DROP_RATE,
                                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                       ape=config.MODEL.SWIN_UPCYCLING.APE,
                                       norm_layer=layernorm,
                                       patch_norm=config.MODEL.SWIN_UPCYCLING.PATCH_NORM,
                                       use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                       fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'vit':
        model = VisionTransformer(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.VIT.PATCH_SIZE,
                                  in_chans=config.MODEL.VIT.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.VIT.EMBED_DIM,
                                  depth=config.MODEL.VIT.DEPTH,
                                  num_heads=config.MODEL.VIT.NUM_HEADS,
                                  mlp_ratio=config.MODEL.VIT.MLP_RATIO,
                                  qkv_bias=config.MODEL.VIT.QKV_BIAS,
                                  representation_size=config.MODEL.VIT.REPRESENTATION_SIZE,
                                  distilled=config.MODEL.VIT.DISTILLED,
                                  drop_rate=config.MODEL.VIT.DROP_RATE,
                                  attn_drop_rate=config.MODEL.VIT.ATTN_DROP_RATE,
                                  drop_path_rate=config.MODEL.VIT.DROP_PATH_RATE,
                                  norm_layer=config.MODEL.VIT.NORM_LAYER,
                                  act_layer=config.MODEL.VIT.ACT_LAYER,
                                  weight_init=config.MODEL.VIT.WEIGHT_INIT)
    elif model_type == 'vit_upcycle':
        model = VisionTransformerUpcycle(img_size=config.DATA.IMG_SIZE,
                                         patch_size=config.MODEL.VIT_UPCYCLING.PATCH_SIZE,
                                         in_chans=config.MODEL.VIT_UPCYCLING.IN_CHANS,
                                         num_classes=config.MODEL.NUM_CLASSES,
                                         embed_dim=config.MODEL.VIT_UPCYCLING.EMBED_DIM,
                                         depth=config.MODEL.VIT_UPCYCLING.DEPTH,
                                         num_heads=config.MODEL.VIT_UPCYCLING.NUM_HEADS,
                                         mlp_ratio=config.MODEL.VIT_UPCYCLING.MLP_RATIO,
                                         qkv_bias=config.MODEL.VIT_UPCYCLING.QKV_BIAS,
                                         representation_size=config.MODEL.VIT_UPCYCLING.REPRESENTATION_SIZE,
                                         distilled=config.MODEL.VIT_UPCYCLING.DISTILLED,
                                         drop_rate=config.MODEL.VIT_UPCYCLING.DROP_RATE,
                                         attn_drop_rate=config.MODEL.VIT_UPCYCLING.ATTN_DROP_RATE,
                                         drop_path_rate=config.MODEL.VIT_UPCYCLING.DROP_PATH_RATE,
                                         norm_layer=config.MODEL.VIT_UPCYCLING.NORM_LAYER,
                                         act_layer=config.MODEL.VIT_UPCYCLING.ACT_LAYER,
                                         weight_init=config.MODEL.VIT_UPCYCLING.WEIGHT_INIT)
    elif model_type == 'vit_moe':
        model = VisionTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.VIT_MOE.PATCH_SIZE,
                                     in_chans=config.MODEL.VIT_MOE.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.VIT_MOE.EMBED_DIM,
                                     depth=config.MODEL.VIT_MOE.DEPTH,
                                     num_heads=config.MODEL.VIT_MOE.NUM_HEADS,
                                     num_local_experts=config.MODEL.VIT_MOE.NUM_LOCAL_EXPERTS,
                                     top_value=config.MODEL.VIT_MOE.TOP_VALUE,
                                     capacity_factor=config.MODEL.VIT_MOE.CAPACITY_FACTOR,
                                     moe_drop=config.MODEL.VIT_MOE.MOE_DROP,
                                     aux_loss_weight=config.MODEL.VIT_MOE.AUX_LOSS_WEIGHT,
                                     mode=config.MODEL.VIT_MOE.MODE)
    elif model_type == 'vit_moge':
        model = VisionTransformerMoGE(img_size=config.DATA.IMG_SIZE,
                                      patch_size=config.MODEL.VIT_MOGE.PATCH_SIZE,
                                      in_chans=config.MODEL.VIT_MOGE.IN_CHANS,
                                      num_classes=config.MODEL.NUM_CLASSES,
                                      embed_dim=config.MODEL.VIT_MOGE.EMBED_DIM,
                                      depth=config.MODEL.VIT_MOGE.DEPTH,
                                      num_heads=config.MODEL.VIT_MOGE.NUM_HEADS,
                                      num_local_experts=config.MODEL.VIT_MOGE.NUM_LOCAL_EXPERTS,
                                      top_value=config.MODEL.VIT_MOGE.TOP_VALUE,
                                      capacity_factor=config.MODEL.VIT_MOGE.CAPACITY_FACTOR,
                                      moe_drop=config.MODEL.VIT_MOGE.MOE_DROP,
                                      aux_loss_weight=config.MODEL.VIT_MOGE.AUX_LOSS_WEIGHT,
                                      mode=config.MODEL.VIT_MOGE.MODE,

                                      is_group_sparsity_loss=config.MODEL.VIT_MOGE.IS_GROUP_SPARSITY_LOSS,
                                      gs_loss_weight=config.MODEL.VIT_MOGE.GS_LOSS_WEIGHT,
                                      gs_window_size=config.MODEL.VIT_MOGE.GS_WINDOW_SIZE,
                                      gs_stride=config.MODEL.VIT_MOGE.GS_STRIDE)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                   patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                                   in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                                   num_classes=config.MODEL.NUM_CLASSES,
                                   embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                                   depths=config.MODEL.SWIN_MOE.DEPTHS,
                                   num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                                   window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                                   mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                                   qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                                   qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                                   drop_rate=config.MODEL.DROP_RATE,
                                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                   ape=config.MODEL.SWIN_MOE.APE,
                                   patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                                   mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                                   init_std=config.MODEL.SWIN_MOE.INIT_STD,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                   pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                   moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                                   num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                                   capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                                   normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                                   is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                                   gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                                   cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                   moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                                   aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)
    elif model_type == 'swin_moge':
        model = SwinTransformerMoGE(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN_MOGE.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN_MOGE.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN_MOGE.EMBED_DIM,
                                    depths=config.MODEL.SWIN_MOGE.DEPTHS,
                                    num_heads=config.MODEL.SWIN_MOGE.NUM_HEADS,
                                    window_size=config.MODEL.SWIN_MOGE.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN_MOGE.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN_MOGE.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN_MOGE.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN_MOGE.APE,
                                    patch_norm=config.MODEL.SWIN_MOGE.PATCH_NORM,
                                    mlp_fc2_bias=config.MODEL.SWIN_MOGE.MLP_FC2_BIAS,
                                    init_std=config.MODEL.SWIN_MOGE.INIT_STD,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    pretrained_window_sizes=config.MODEL.SWIN_MOGE.PRETRAINED_WINDOW_SIZES,
                                    moe_blocks=config.MODEL.SWIN_MOGE.MOE_BLOCKS,
                                    num_local_experts=config.MODEL.SWIN_MOGE.NUM_LOCAL_EXPERTS,
                                    top_value=config.MODEL.SWIN_MOGE.TOP_VALUE,
                                    capacity_factor=config.MODEL.SWIN_MOGE.CAPACITY_FACTOR,
                                    cosine_router=config.MODEL.SWIN_MOGE.COSINE_ROUTER,
                                    normalize_gate=config.MODEL.SWIN_MOGE.NORMALIZE_GATE,
                                    use_bpr=config.MODEL.SWIN_MOGE.USE_BPR,
                                    is_gshard_loss=config.MODEL.SWIN_MOGE.IS_GSHARD_LOSS,
                                    gate_noise=config.MODEL.SWIN_MOGE.GATE_NOISE,
                                    cosine_router_dim=config.MODEL.SWIN_MOGE.COSINE_ROUTER_DIM,
                                    cosine_router_init_t=config.MODEL.SWIN_MOGE.COSINE_ROUTER_INIT_T,
                                    moe_drop=config.MODEL.SWIN_MOGE.MOE_DROP,
                                    aux_loss_weight=config.MODEL.SWIN_MOGE.AUX_LOSS_WEIGHT,

                                    is_group_sparsity_loss=config.MODEL.SWIN_MOGE.IS_GROUP_SPARSITY_LOSS,
                                    gs_loss_weight=config.MODEL.SWIN_MOGE.GS_LOSS_WEIGHT,
                                    gs_window_size=config.MODEL.SWIN_MOGE.GS_WINDOW_SIZE,
                                    gs_stride=config.MODEL.SWIN_MOGE.GS_STRIDE)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
