import torch
from torch import nn
import numpy as np
from data_provider.data_factory import data_provider
from torch import optim
from utils.tools import EarlyStopping, adjust_learning_rate, visual, set_seed
from utils.metrics import metric
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.dtw_metric import accelerated_dtw
import os, time
from hydra.utils import instantiate
from arch.Model import Model
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import pynvml
import gc

class Exp(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        set_seed(cfg.seed)
        output_dir = HydraConfig.get().runtime.output_dir
        with open(os.path.join(output_dir, 'arch.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        self.logger = logger
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.model.to(self.device)

        self.early_stop_flag = False
        self.best_epoch = 0
    
    def _get_free_gpu(self):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                return i

        return None

    def _acquire_device(self):
        return torch.device('cuda')

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        return model_optim

    def _select_criterion(self, loss):
        loss = loss.lower()
        if loss == 'mse':
            criterion = nn.MSELoss()
        elif loss == 'mae':
            criterion = nn.L1Loss()
        elif loss == 'mape':
            criterion = mape_loss()
        elif loss == 'smape':
            criterion = smape_loss()
        elif loss == 'mase':
            criterion = mase_loss()
        else:
            raise ValueError("Unsupported loss type: {}".format(loss))
        return criterion
 
    def _build_model(self):
        model = Model(self.cfg.model, self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.cfg.dataset, flag)
        return data_set, data_loader

    def train(self, setting):
        self.setting = setting
        self.logger.info(f'setting: {setting}')
        self.start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.cfg.train.checkpoints, setting)
        if not os.path.exists(path):
            if self.cfg.train.save_model:
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.cfg.train.patience, verbose=True, save_model=self.cfg.train.save_model)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.cfg.train.loss)

        if self.cfg.train.lradj == 'TST':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.cfg.train.pct_start,
                                                epochs=self.cfg.train.epochs,
                                                max_lr=self.cfg.train.lr)


        for epoch in range(self.cfg.train.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)

                f_dim = -1 if self.cfg.dataset.features == 'MS' else 0
                outputs = outputs[:, -self.cfg.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.cfg.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.cfg.train.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.cfg.train.lradj == 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.cfg.train, scheduler, printout=False)
                    scheduler.step()
            cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info("Epoch: {0}, Steps: {1}, time: {2:.3f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, cost_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, epoch+1)
            self.best_epoch = early_stopping.best_epoch
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                self.early_stop_flag = True
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.cfg.train.lradj, self.cfg.train.lr, self.cfg.train.epochs)
            if self.cfg.train.lradj == 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.cfg.train, scheduler)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.cfg.train)

        if self.cfg.train.save_model:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(early_stopping.best_model)

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                f_dim = -1 if self.cfg.dataset.features == 'MS' else 0
                outputs = outputs[:, -self.cfg.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.cfg.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        test_result_path = os.path.join(HydraConfig.get().runtime.output_dir, 'test_results/')
        if not os.path.exists(test_result_path):
            os.makedirs(test_result_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_cycle = batch_cycle.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)

                f_dim = -1 if self.cfg.dataset.features == 'MS' else 0
                outputs = outputs[:, -self.cfg.pred_len:, :]
                batch_y = batch_y[:, -self.cfg.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.cfg.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.cfg.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if self.cfg.train.save_res:
                        visual(gt, pd, os.path.join(test_result_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_path = os.path.join(HydraConfig.get().runtime.output_dir, 'results/')
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # dtw calculation
        if self.cfg.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.end_time = time.time()
        use_time = self.end_time - self.start_time

        results = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'dtw': float(dtw) if isinstance(dtw, (float, int)) else dtw,
            'time': float(use_time),
            'early_stop': bool(self.early_stop_flag),
            'best_epoch': int(self.best_epoch)
        }

        # line = 'mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, dtw:{}, time:{}, early_stop:{}, best_epoch:{}'.format( \
            # mse, mae, rmse, mape, mspe, dtw, use_time, self.early_stop_flag, self.best_epoch)
        # self.logger.info('mse:{}, mae:{}, time:{}'.format(mse, mae, use_time))
        # self.logger.info(line)
        # f = open("result_long_term_forecast.txt", 'a')
        np.save(result_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if self.cfg.train.save_res:
            np.save(result_path + 'pred.npy', preds)
            np.save(result_path + 'true.npy', trues)

        return results

    def empty(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        if self.cfg.train.del_model and self.cfg.train.save_model:
            output_dir = HydraConfig.get().runtime.output_dir
            # checkpoints_dir = os.path.join(output_dir, 'checkpoints')
            checkpoints_path = os.path.join(self.cfg.train.checkpoints, self.setting)
            if os.path.exists(checkpoints_path):
                import shutil
                shutil.rmtree(checkpoints_path)
            