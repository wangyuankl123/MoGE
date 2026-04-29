# MoGE for Language Modeling

This folder contains the implementation of the Mixture of Group Experts (MoGE) for language modeling.

## 1. Installation

- Create a conda virtual environment:

```bash
conda create -n moge_lm python=3.9.18 -y
conda activate moge_lm
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

- Install other requirements:

```bash
pip install -i ./requirements.txt
```

## 2. Data preparation

- Download the WikiText-103 dataset from [here](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and preprocess the dataset:
```bash
unzip -q wikitext-103-v1.zip
cd wikitext-103
mv wiki.train.tokens train.txt
mv wiki.valid.tokens valid.txt
mv wiki.test.tokens test.txt
cd ..
```

- then change bash scripts based on your local data paths.
```bash
data_directory/
    └── wikitext-103
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

## 3. Reproducing Results
### Pretraining <u>SMoGE</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoge-s.sh
bash scripts/smoge-m.sh
```

### Pretraining <u>*Momentum*SMoGE</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoge-mom-s.sh
bash scripts/smoge-mom-m.sh
```

### Pretraining <u>*Adam*SMoGE</u> on WikiText-103: 

``` # WikiText-103 dataset: 
bash scripts/smoge-adam-m.sh
```

## Acknowledgment

This project is built upon the foundation laid by [MomentumSMoE: Integrating Momentum into Sparse Mixture of Experts](https://github.com/rachtsy/MomentumSMoE). The original code from this project is licensed under the [Apache-2.0 License](https://github.com/rachtsy/MomentumSMoE/blob/main/LICENSE.txt). We would like to thank the authors for their great work and contributions.
