import os
import time
import math
import shutil
import yaml
from pathlib import Path
import gc

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np
from einops import rearrange
from frouros.detectors.concept_drift import (
    ADWINConfig, KSWINConfig
)

from data.data_loader import *
from utils.augmentations import Augmenter
from utils.metrics import metric
from utils.caliper import CALIPER
from utils.tools import EarlyStopping
from utils.detectors import CustomADWIN, CustomKSWIN

SETTINGS_DIR = Path('./settings')
MODELS_CONFIG_PATH = SETTINGS_DIR / 'models.yaml'

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        self.verbose = args.verbose
        self.device = self._acquire_device()
        self.view_shape = args.verbose

        if self.args.drift_detection:
            self._initialize_drift_detector()

    def model_init(self, args):
        if self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
        self.model = self.net_class(args, device = self.device)

    def set_mode(self, mode):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

    def grid_search(self, config, setting):
        args = self.args
        best_lr = args.learning_rate
        best_vali_loss = np.inf
        if args.verbose: print('>>> Started grid search', flush=True)
        for lr in config['learnig_rate_list']:
            args.learning_rate = lr
            self.model_init(args)  # set experiments
            _, vali_loss = self.train(setting)
            if vali_loss < best_vali_loss:
                best_lr = lr
                best_vali_loss = vali_loss
        self.args.learning_rate = best_lr
        self.model_init(self.args)
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.args.use_amp = False
        return device
    
    def _initialize_drift_detector(self):
        """Initializes the concept drift detector based on arguments."""
        detector_factory = {
            'ADWIN': [CustomADWIN, ADWINConfig],
            'KSWIN': [CustomKSWIN, KSWINConfig],
        }
        
        if self.args.detector not in detector_factory:
            raise ValueError(f"Unsupported drift detector: {self.args.detector}")

        detector_class, config_class = detector_factory[self.args.detector]

        # Load detector-specific parameters from a YAML file.
        config_path = f'./settings/local/{self.args.detector}.yaml'
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        
        config = config_class(**params)
        self.detector = detector_class(config=config)
        
        # Initialize DySAW for dynamic sliding window adjustment.
        self.dsw = DySAW(d=self.args.enc_in, pred_len=self.args.pred_len, seed=self.args.exp_seed, n_jobs=self.args.DySAW_njob)
        
        # Minimum data length required to consider a restart.
        self.min_window = np.maximum(self.args.fix_win, (self.args.seq_len + self.args.pred_len + self.args.batch_size))

    def _get_data(self, flag: str):
        """
        Creates and returns a dataset and dataloader for a given mode.
        Args:
            flag: One of 'train', 'val', or 'test'.
        """
        dataset_map = {
            'MoCap': Dataset_custom, 
            'Dysts': Dataset_custom,
            'TEP': Dataset_TEP, 
        }

        with open(MODELS_CONFIG_PATH, 'r') as yml:
            model_list = yaml.safe_load(yml)
        if self.args.data not in dataset_map:
            raise ValueError(f"Dataset '{self.args.data}' not found in dataset_map.")
        
        Data = dataset_map[self.args.data]

        # Configure dataloader parameters based on the flag.
        if (flag == 'test') or (self.args.method in model_list.get('Others', [])):
            shuffle_flag, drop_last, bsz, num_workers, pin = False, False, self.args.test_bsz, 0, True
        elif (flag == 'val'):
            shuffle_flag, drop_last, bsz, num_workers, pin = True, False, 4096, 0, True
        else:
            shuffle_flag, drop_last, bsz, num_workers, pin = True, True, self.args.batch_size, self.args.num_workers, True

        data_set = Data(
            root_path=self.args.root_path, data_path=self.args.data_path, flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features, target=self.args.target,
            train_rate=self.args.train_rate, test_rate=self.args.test_rate,
            scale = self.args.scale, inverse=self.args.inverse, timeenc=1, 
            freq=self.args.detail_freq, cols=self.args.cols
        )
        if self.view_shape:
            print(f"{flag} data size: {len(data_set)}")
            
        data_loader = DataLoader(
            data_set, batch_size=bsz, shuffle=shuffle_flag,
            num_workers=num_workers, drop_last=drop_last, pin_memory=pin,
        )
        return data_set, data_loader
    
    def get_inp(self, y):
        dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
        return torch.cat([y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

    def get_model_stats(self):
        cpu_memory = self.get_cpu_memory()
        gpu_memory = 0
        if self.args.use_gpu:
            gpu_memory = self.get_gpu_memory()
        return cpu_memory, gpu_memory, self.count_params()
    

    def count_params(self):
        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        return num_params


    def get_cpu_memory(self):
        total_memory = 0
        for param in self.model.parameters():
            total_memory += param.numel() * param.element_size()
        return total_memory
    

    def get_gpu_memory(self):
        return torch.cuda.max_memory_allocated()
    
    def get_augmenter(self, sample_batched):
    
        seq_len = sample_batched.shape[1]
        num_channel = sample_batched.shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        elif self.input_channels_dim == 1 and seq_len>1000: 
            self.augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25, crop_prob=1, dropout_prob=0.0)
            #we apply cutout 3 times in a row.
            self.augmenter.augmentations = [self.augmenter.history_cutout, self.augmenter.history_cutout, self.augmenter.history_cutout,
                                            self.augmenter.history_crop, self.augmenter.gaussian_noise, self.augmenter.spatial_dropout]
        #if there is only one channel but not long, we just need to make sure that we don't drop this only channel
        else:
            self.augmenter = Augmenter(cutout_length=cutout_len, dropout_prob=0.0)
            

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    

    def train(self, setting, train_loader=None, vali_loader=None, early_stop=True):
        if train_loader is None:
            train_data, train_loader = self._get_data(flag='train')
        if early_stop and (vali_loader is None):
            vali_data, vali_loader = self._get_data(flag='val')

        vali_loss_list = []

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=self.verbose)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.amp.GradScaler()

        scheduler = StepLR(self.opt, step_size=5, gamma=0.8)
        self.set_mode('train')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, update_x, update_y, update_x_mark, update_y_mark, uflag) in enumerate(train_loader):
                iter_count += 1
                data = [batch_x, batch_y, batch_x_mark, batch_y_mark, self.get_inp(batch_y)]
                self.opt.zero_grad()
                pred, true = self._process_one_batch(data=data)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if self.verbose:  print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    if self.verbose:  print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                
            if self.verbose:  print("Epoch: {}".format(epoch + 1))
            train_loss = np.average(train_loss)
            if early_stop:
                vali_loss = self.vali(vali_loader, criterion)
                vali_loss_list.append(vali_loss)
                #test_loss = self.vali(test_data, test_loader, criterion)
                if self.verbose:  print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    if self.verbose:  print("Early stopping")
                    break
            elif self.verbose:  
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

            scheduler.step()
        
        if early_stop:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        if os.path.abspath(path) != os.path.abspath(self.args.checkpoints):
            shutil.rmtree(path)
        
        if len(vali_loss_list) > 0:
            return self.model, min(vali_loss_list)
        else:
            return self.model, np.inf
        
    def vali(self, vali_loader, criterion):
        self.set_mode('eval')
        total_loss = []
        with torch.inference_mode():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, update_x, update_y, update_x_mark, update_y_mark, uflag) in enumerate(vali_loader):
                data = [batch_x, batch_y, batch_x_mark, batch_y_mark, self.get_inp(batch_y)]
                pred, true = self._process_one_batch(data=data)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.set_mode('train')
        return total_loss

    def test(self, setting):
        self.set_mode('eval')
        if self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
                
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        ct = []
        memory = []
        start = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, update_x, update_y, update_x_mark, update_y_mark, uflag) in enumerate(test_loader):
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            data = [batch_x, batch_y, batch_x_mark, batch_y_mark, self.get_inp(batch_y)]
            data_update = [update_x, update_y, update_x_mark, update_y_mark, self.get_inp(update_y)]
            torch.cuda.synchronize()
            start_one_batch = time.time()
            update=uflag.flatten()[0]
            pred, true = self._process_one_batch(data=data, uflag=update, data_update=data_update, mode='test')
            torch.cuda.synchronize()
            end_one_batch = time.time()
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            if update == 1:
                ct.append(end_one_batch-start_one_batch)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        if self.view_shape:  print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE = metric(preds, trues)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        cpu_memory, gpu_memory, num_params = self.get_model_stats()
        Results = {
            'MAE': MAE,
            'MSE': MSE,
            'preds': preds,
            'trues': trues
        }
        Stats={
            'comp_time': ct,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'total_memory': cpu_memory+gpu_memory,
            'num_params': num_params,
            'latency': [0],
            'selected_size': [0]
        }
        return [mae, mse, rmse, mape, mspe, exp_time], Results, Stats

    def _process_one_batch(self, data, uflag = 0, data_update=[None, None, None, None, None], mode='train'):
        if (uflag != 0) and (self.online != 'none'):
            return self._ol_one_batch(data, data_update)
        
        [batch_x, batch_y, batch_x_mark, batch_y_mark, x_dec] = data
        x_enc = batch_x.float().to(self.device)
        x_mark_enc = batch_x_mark.float().to(self.device)
        x_mark_dec = batch_y_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if self.args.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        if mode == 'train':
            return rearrange(outputs, 'b t d -> b (t d)'), rearrange(batch_y, 'b t d -> b (t d)')
        else:
            return outputs, batch_y
    
    def _ol_one_batch(self, data, data_update):
        # b, t, d = batch_y.shape
        criterion = self._select_criterion()
        [update_x, update_y, update_x_mark, update_y_mark, x_dec_update] = data_update
        x_enc_update = update_x.float().to(self.device)
        x_mark_enc_update = update_x_mark.float().to(self.device)
        x_mark_dec_update = update_y_mark.float().to(self.device)
        self.set_mode('train')

        for _ in range(self.n_inner):
            self.opt.zero_grad()
            if self.args.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs_update = self.model(x_enc_update, x_mark_enc_update, x_dec_update, x_mark_dec_update)
            else:
                outputs_update = self.model(x_enc_update, x_mark_enc_update, x_dec_update, x_mark_dec_update)
            outputs_update = rearrange(outputs_update, 'b t d -> b (t d)').float().to(self.device)
            loss =  criterion(outputs_update, rearrange(update_y, 'b t d -> b (t d)').float().to(self.device))
            loss.backward()
            self.opt.step()  
            self.model.store_grad()     
            self.set_mode('eval')

        with torch.inference_mode():
            [batch_x, batch_y, batch_x_mark, batch_y_mark, x_dec] = data
            x_enc = batch_x.float().to(self.device)
            x_mark_enc = batch_x_mark.float().to(self.device)
            x_mark_dec = batch_y_mark.float().to(self.device)
            batch_y = batch_y.float()
            if self.args.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            f_dim = -1 if self.args.features=='MS' else 0
            batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y 
    
    def _process_restart(self, setting, data_re, data_stamp_re):
        args = self.args
        self.model_init(args)
        train_data = data_re
        train_stamp = data_stamp_re

        train_set = Dataset_restart(
                        root_path=args.root_path,
                        train_rate=args.train_rate,
                        test_rate=args.test_rate,
                        data_path=args.data_path,
                        data=train_data,
                        data_stamp=train_stamp,
                        flag='train',
                        size=[args.seq_len, args.label_len, args.pred_len],
                        features=args.features,
                        target=args.target,
                        inverse=args.inverse,
                        timeenc=1,
                        freq=args.detail_freq,
                        cols=args.cols
                    )
        
        train_loader = DataLoader(train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)
        
        self.train(setting, train_loader, early_stop=False)
        self.detector.reset()
        self.dsw.reset()
        self.set_mode('eval')
        del train_set
        del train_loader
        return 

    

    def test_drift(self, setting):
        """Executes testing with online concept drift detection and adaptation."""
        test_data, test_loader = self._get_data(flag='test')
        self.set_mode('eval')
        preds, trues, compute_times = [], [], []
        drift_state = {'detected': False, 'win_start': 0, 'win_end': 0, 'time_point': 0}
        start = time.time()
        self.Latencies = []
        self.select_win = []
        self.drift_time = []
        self.train_start_time = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, update_x, update_y, update_x_mark, update_y_mark, is_update_batch) in enumerate(test_loader):
            
            # If it's an update batch, perform online learning and check for drift
            drift_state['time_point'] = i
            torch.cuda.synchronize()
            start_time = time.time()

            if is_update_batch.flatten()[0] == 1:
                update_inputs = [update_x, update_y, update_x_mark, update_y_mark, self.get_inp(update_y)]
                self._online_update_and_detect_drift(update_inputs, drift_state, i, test_data, setting)
            
            # Perform inference on the current batch
            
            with torch.inference_mode():
                model_inputs = [batch_x, batch_y, batch_x_mark, batch_y_mark, self.get_inp(batch_y)]
                pred, true = self._process_one_batch(model_inputs, mode='test')

            torch.cuda.synchronize()
            end_time = time.time()

            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            compute_times.append(end_time - start_time)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        if self.view_shape:  print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE = metric(preds, trues)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        cpu_memory, gpu_memory, num_params = self.get_model_stats()
        Results = {
            'MAE': MAE,
            'MSE': MSE,
            'preds': preds,
            'trues': trues
        }
        if len(self.Latencies) == 0:
            self.Latencies = [np.nan]
            self.select_win = [np.nan]
            self.drift_time = [np.nan]
            self.train_start_time = [np.nan]
        Stats={
            'comp_time': compute_times,
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'total_memory': cpu_memory+gpu_memory,
            'num_params': num_params,
            'latency': np.array(self.Latencies),
            'drift_time': np.array(self.drift_time),
            'train_start_time': np.array(self.train_start_time),
            'selected_size': np.array(self.select_win)
        }
        return [mae, mse, rmse, mape, mspe, exp_time], Results, Stats
    
    def _online_update_and_detect_drift(self, update_inputs, drift_state, current_step, test_data, setting):
        """Handles the logic for one online update step and drift detection."""
        # Calculate error on the update batch for the detector
        with torch.inference_mode():
            u_pred, u_true = self._process_one_batch(update_inputs, mode='test')
        error = torch.sqrt(((u_pred - u_true) ** 2).mean()).item()
        
        # Feed error to the detector if a drift has not already been flagged
        if not drift_state['detected']:
            self.detector.update(value=error)
            if self.detector.status["drift"]:
                drift_state['detected'] = True
                self.latency = 0
                drift_state['win_end'] = current_step + self.args.seq_len - 1
                drift_state['win_start'] = drift_state['win_end'] - self.detector.update_instances + 1
        
        # If drift is flagged, expand the window and check for restart
        if drift_state['detected']:
            drift_state['win_end'] += 1
            self._check_and_process_restart(drift_state, test_data, setting)

    def _check_and_process_restart(self, drift_state, test_data, setting):
        """Checks if conditions are met for a model restart and triggers it."""
        win_st, win_ed = drift_state['win_start'], drift_state['win_end']
        window_size = win_ed - win_st

        # A minimum window size is required before considering a restart
        if window_size <= self.min_window:
            return

        if self.args.use_DySAW:
            # Use DySAW to find an optimal restart point
            should_restart, restart_data, restart_stamps, _ = self.dsw.detect(test_data, win_st, win_ed)
        else:
            # Simple restart strategy: use the whole detected window
            restart_data, restart_stamps = test_data.get_restart_data(win_st, win_ed)
            should_restart = True
        
        # print(restart_data.shape)
        
        if should_restart:
            # print(f"Drift confirmed. Restarting model training with data from steps {win_st} to {win_ed}.")
            self._process_restart(setting, restart_data, restart_stamps)
            # Reset state after restart
            drift_state['detected'] = False
            self.Latencies.append(self.latency)
            self.select_win.append(win_ed-win_st)
            self.train_start_time.append(drift_state['time_point'])
            self.drift_time.append(drift_state['time_point'] - self.latency)
        else:
            self.latency += 1
    
    
