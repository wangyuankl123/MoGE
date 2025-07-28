# MoGE for Image Classification

This folder contains the implementation of the Mixture of Group Experts (MoGE) for image classification.

## 1. Installation

- Create a conda virtual environment:

```bash
conda create -n moge python=3.7 -y
conda activate moge
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install other requirements:

```bash
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main
```

- Install fused window process for acceleration, activated by passing `--fused_window_process` in the running script
```bash
cd kernels/window_process
python setup.py install #--user
```

## 2. Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

## 3. Reproducing Results

### Training from scratch on ImageNet-1K

To train **dense model** with 4 GPU on a single node for 150 epochs, run:

`ViT-S/16`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_dense.py \
--cfg configs/vitmoe/vit_moe_small_patch16_224_densebaseline_1k.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```

`SwinV2-T/12`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_dense.py \
--cfg configs/swinmoe/swin_moe_tiny_patch4_window12_192_densebaseline_1k.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```

To train **MoE models** with 4 GPU on a single node for 150 epochs, run:

`V-MoE-S/16`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_vitmoe.py \
--cfg configs/vitmoe/vit_moe_small_patch16_224_32expert_4gpu_1k_last2_C4.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```

`SwinV2-MoE-T/12`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_swinmoe.py \
--cfg configs/swinmoe/swin_moe_tiny_patch4_window12_192_32expert_4gpu_1k_last2_C4.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```

To train **MoGE models** with 4 GPU on a single node for 150 epochs, run:

`V-MoGE-S/16`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_vitmoge.py \
--cfg configs/vitmoe/vit_moge_small_patch16_224_32expert_4gpu_1k_last2_C4.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```

`SwinV2-MoGE-T/12`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --nnode=1 --node_rank=0 --master_port 12345  main_swinmoge.py \
--cfg configs/swinmoe/swin_moge_tiny_patch4_window12_192_32expert_4gpu_1k_last2_C4.yaml --data-path <imagenet-path> --batch-size 256 --fused_window_process
```


### Fine-tuning on ImageNet-1K

#### Step 1

Download the ImageNet-22K pretrained models and move them to the folder `./Image_Classification/checkpoints/`.

MoE and MoGE models perform sparse upcycling to initialize experts from dense checkpoints pretrained on ImageNet-22K.

#### Step 2

To fine-tune **dense models** with 4 GPU on a single node for 30 epochs, run:

`ViT-S/16`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_finetune_vit.py --cfg configs/vit/vit_small_patch16_224_22kto1k_finetune.yaml \
--pretrained ./checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

`SwinV1-T/7`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_finetune.py --cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml \
--pretrained ./checkpoints/swin_tiny_patch4_window7_224_22k.pth --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

To fine-tune **MoE models** with 4 GPU on a single node for 30 epochs, run:

`V-MoE-S/16`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_upcycle_vit.py --cfg configs/vit/vit_small_patch16_224_22kto1k_finetune_upcycling32e.yaml \
--pretrained ./checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

`SwinV1-MoE-T/7`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 36535  main_upcycle.py --cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune_upcycling32e.yaml \
--pretrained ./checkpoints/swin_tiny_patch4_window7_224_22k.pth --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

To fine-tune **MoGE models** with 4 GPU on a single node for 30 epochs, run:

`V-MoGE-S/16`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 38191  main_upcycle_vit.py --cfg configs/vit/vit_small_patch16_224_22kto1k_finetune_upcycling32e_moge.yaml \
--pretrained ./checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

`SwinV1-MoGE-T/7`:

```bashs
python -m torch.distributed.launch --nproc_per_node 4 --master_port 16875  main_upcycle.py --cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune_upcycling32e_moge.yaml \
--pretrained ./checkpoints/swin_tiny_patch4_window7_224_22k.pth --data-path <imagenet-path> --batch-size 64 --accumulation-steps 2
```

## Acknowledgment

This project is built upon the foundation laid by [Swin Transformer V2: Scaling Up Capacity and Resolution](https://github.com/microsoft/Swin-Transformer), [Tutel: Adaptive Mixture-of-Experts at Scale](https://github.com/microsoft/tutel) and [Scaling Vision with Sparse Mixture of Experts](https://github.com/google-research/vmoe). The original code from their project is licensed under the [MIT License](https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE), [MIT License](https://github.com/microsoft/Tutel/blob/main/LICENSE) and [Apache-2.0 License](https://github.com/google-research/vmoe/blob/main/LICENSE) respectively. We would like to thank the authors for their great work and contributions.