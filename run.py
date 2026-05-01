import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
from torch import nn
from exp import Exp
import random
import numpy as np
import logging
import os
import json
import fcntl

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    task_name = cfg.task_name
    logger = logging.getLogger(task_name)

    arch = {
        'dataset': cfg.dataset.name,
        'seq_len': cfg.seq_len,
        'pred_len': cfg.pred_len,
        'embedding': cfg.model.embedding.name,
        'encoder': cfg.model.encoder.name,
        'decoder': cfg.model.decoder.name,
        'lr': cfg.train.lr,
        'd_model': cfg.model.d_model,
        'e_layers': cfg.model.encoder.e_layers,
        'channel_independence': cfg.model.channel_independence
    }

    # setting = cfg.setting
    setting = f"{cfg.dataset.name}_{cfg.seq_len}_{cfg.pred_len}_{cfg.model.embedding.name}_{cfg.model.encoder.name}_{cfg.model.decoder.name}_lr{cfg.train.lr}_d{cfg.model.d_model}_el{cfg.model.encoder.e_layers}_ci{cfg.model.channel_independence}"
    
    if cfg.use_json:
        assert os.path.exists(cfg.json_path), f"JSON file {cfg.json_path} does not exist."
        with open(cfg.json_path, 'r') as f:
            kv = json.load(f)
        if setting in kv:
            logger.info(f"Setting {setting} found in JSON. Skipping experiment.")
            results = kv[setting]['results']
            line = 'success:{}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}, time:{}, early_stop:{}, best_epoch:{}'.format( \
                results.get('success'), results.get('mse'), results.get('mae'), results.get('rmse'), results.get('mape'), results.get('mspe'), results.get('dtw'), results.get('time'), results.get('early_stop'), results.get('best_epoch'))
            logger.info(line)
            return

    if cfg.model.get("learning_rate") is not None:
        cfg.train.lr = cfg.model.learning_rate
    if cfg.model.get("batch_size") is not None:
        cfg.train.batch_size = cfg.model.batch_size


    try:
        exp = Exp(cfg, logger)
        logger.info(f"Experiment setting: {setting}")
        exp.train(setting)
        results = exp.test(setting)

        line = 'mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}, time:{}, early_stop:{}, best_epoch:{}'.format( \
            results.get('mse'), results.get('mae'), results.get('rmse'), results.get('mape'), results.get('mspe'), results.get('dtw'), results.get('time'), results.get('early_stop'), results.get('best_epoch'))
        logger.info(line)
        f = open(cfg.output, 'a')
        f.write(setting + "  \n")
        f.write(line)
        f.write('\n')
        f.write('\n')
        f.close()

        def update_json(json_path, setting, arch, results):
            with open(json_path, 'r+') as f:
                # 加锁，阻塞直到获得锁
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    kv = json.load(f)
                    kv[setting] = {
                        **arch,
                        **results
                    }
                    # 回到文件开头并清空
                    f.seek(0)
                    f.truncate()
                    json.dump(kv, f, indent=4)
                finally:
                    # 释放锁
                    fcntl.flock(f, fcntl.LOCK_UN)

        if cfg.use_json:
            update_json(cfg.json_path, setting, arch, results)

        exp.empty()

    except Exception as e:
        # print(f"An error occurred: {e}")
        if not cfg.debug:
            logger.error(f"An error occurred: {e}")

            with open(cfg.output, 'a') as f:
                f.write(setting + "  \n")
                f.write('error')
                f.write('\n')
                f.write('\n')

            with open('error_log.txt', 'a') as f:
                f.write(f"Setting: {setting}\n")
                f.write(f"Error: {e}\n")
                f.write("\n")
        else:
            raise e

if __name__ == "__main__":
    print("version exp 1.3")
    main()
    os._exit(0)