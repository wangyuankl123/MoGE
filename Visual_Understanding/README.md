# MoGE for Visual Understanding

This folder contains the implementation of the Mixture of Group Experts (MoGE) for visual understanding.

## 1. Requirements and Installation

We recommend the requirements as follows.
* Python == 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* **Transformers == 4.37.0**
* **Tokenizers==0.15.1**
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
cd MoE-LLaVA
conda create -n moellava python=3.10 -y
conda activate moellava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Below are optional. For Qwen model.
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```


## 2. Checkpoints

Download the MoE-LLaVA-1.6B×4-Top2 checkpoint from [LanguageBind/MoE-LLaVA-StableLM-Stage2](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-Stage2)







## 3. Training & Validating
The training & validating instruction is in [TRAIN.md](docs/TRAIN.md) & [EVAL.md](docs/EVAL.md).

## Acknowledgement

This project is built upon the foundation laid by [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](https://github.com/PKU-YuanGroup/MoE-LLaVA). The original code from this project is licensed under the [Apache-2.0 License](https://github.com/PKU-YuanGroup/MoE-LLaVA/blob/main/LICENSE). We would like to thank the authors for their great work and contributions.
