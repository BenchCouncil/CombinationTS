# CombinationTS

Official implementation of **CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models** (ICML 2026).

## :fire:News
- [2026-05-01] Our **CombinationTS** has been accepted to **ICML 2026**! 🎉

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.12+

Place datasets under `./dataset/` (ETT-small, weather, exchange_rate, traffic, electricity, illness).

## Quick Start

```bash
# PatchTST
python run.py +model=patchtst +dataset=ETTh1 seq_len=96 pred_len=96

# iTransformer
python run.py +model=itransformer +dataset=Weather seq_len=96 pred_len=96

# DLinear
python run.py +model=dlinear +dataset=Exchange seq_len=336 pred_len=96

# TimesNet
python run.py +model=timesnet +dataset=ECL seq_len=96 pred_len=96

# FreTS
python run.py +model=frets +dataset=ETTh2 seq_len=96 pred_len=96

# TimeMixer
python run.py +model=timemixer +dataset=Traffic seq_len=96 pred_len=96

# Custom component combination
python run.py \
  +model/embedding=Patch16 \
  +model/encoder=Transformer \
  +model/decoder=Linear \
  +dataset=ETTh1 \
  model.use_norm=true \
  model.channel_independence=true \
  model.encoder.e_layers=1 \
  model.d_model=512
```

## Citation

```bibtex
@inproceedings{combinationts2026,
  title={CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models},
  author={Xiaorui Wang, Fanda Fan, Chenxi Wang, Yuxuan Yang, Rui Tang, Kuoyu Gao, simiao pang, Yuanfeng Shang, Zhipeng Liu, Wanling Gao, Lei Wang, Jianfeng Zhan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
