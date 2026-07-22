# CombinationTS
<p align="left"> <a href="https://arxiv.org/abs/2605.01231"> <img src="https://img.shields.io/badge/arXiv-2605.01231-b31b1b.svg"></a> <a href="https://creativecommons.org/licenses/by/4.0/"> <img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg"></a> </p>

CombinationTS is a modular framework that decomposes, recombines, and diagnoses time-series forecasting models.

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
@inproceedings{
  wang2026combinationts,
  title={Combination{TS}: A Modular Framework for Understanding Time-Series Forecasting Models},
  author={Xiaorui Wang and Fanda Fan and Chenxi Wang and Yuxuan Yang and Rui Tang and Kuoyu Gao and simiao pang and Yuanfeng Shang and Zhipeng Liu and Wanling Gao and Lei Wang and Jianfeng Zhan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=CwHRT46VmC}
}
```
