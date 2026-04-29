## Data preparation


- The LLaVA-FT is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1rwub9o0T3_7ZHbPZzCiLZw?pwd=0yhi), [Google Disk](https://drive.google.com/file/d/13YxtVowfhUIpGOCODhKFstoRBvogF4od/view?usp=sharing), [Peking University Disk](https://disk.pku.edu.cn/link/AA10683317FB824FB9B2427A6B268EAADB) or [Hugging Face](https://huggingface.co/datasets/LanguageBind/MoE-LLaVA/tree/main/train_json)


We also provide the processed data as follows. The link is to BaiDu Disk.

| Data group  | Usage | Link |
|----------|----------|-----------|
| LLaVA-FT | Stage 3 |  [LLaVA 1.5-mix-665k](https://pan.baidu.com/s/1xC9E6VuOOEBV5iieve0Z7A?pwd=2o0a) |

**For those who can not easily access to BaiDu Disk**, you can download data from [Hugging Face](https://huggingface.co/datasets/LanguageBind/MoE-LLaVA).

After downloading all of them, organize the data as follows in ```IMAGE_FOLDER```. 

```Shell
IMAGE_FOLDER
└── llava_image_tune
```


## Training
Specify your `IMAGE_FOLDER` and `JSON_FOLDER` according to the data preparation.

### StableLM
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/stablelm/finetune_moe.sh).
