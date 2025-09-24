import sys
import numpy as np
import pandas as pd
import torch
import scipy
import re
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_base(Dataset):
    """
    Base class for time series datasets.
    Provides common functionalities like initialization and data access patterns.
    """

    def __init__(self, root_path, train_rate, test_rate, data_path,
                 flag='train', size=None, delay_fb=False, features='M',
                 target='OT', scale=True, inverse=False, timeenc=1, freq='s', cols=None):
        # size: [seq_len, label_len, pred_len]
        if size is None:
            raise ValueError("Size must be provided as a list: [seq_len, label_len, pred_len]")
        self.seq_len, self.label_len, self.pred_len = size

        assert 0.0 <= train_rate + test_rate <= 1.0, "The sum of train_rate and test_rate must be between 0 and 1."
        self.train_rate = train_rate
        self.test_rate = test_rate

        assert flag in ['train', 'test', 'val'], f"Invalid flag: {flag}"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.delay_fb = delay_fb
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        # Attributes to be populated by subclasses
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.scaler = None

    def _load_data(self):
        """
        Default data reading implementation for generic CSV files.
        Assumes the CSV has a 'date' column or one can be generated.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        if 'date' not in df_raw.columns:
            df_raw['date'] = pd.to_datetime(pd.date_range(start='2004-01-01', periods=len(df_raw), freq=self.freq))
        
        df_raw = df_raw[['date'] + [col for col in df_raw.columns if col != 'date']]

        if self.target:
            if self.cols:
                cols = self.cols.copy()
                cols.remove(self.target)
                df_raw = df_raw[cols + [self.target]]
            else:
                cols = list(df_raw.columns)
                cols.remove(self.target)
                df_raw = df_raw[cols + [self.target]]
                cols.remove('date')
        else:
            cols = list(df_raw.columns)
            df_raw = df_raw[cols]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError(f"Unknown features type: {self.features}")

        self._split_and_scale_generic(df_raw, df_data)

    def _split_and_scale_generic(self, df_raw, df_data):
        """Helper method to split, scale data, and set instance variables."""
        len_data = len(df_raw)
        num_train = int(len_data * self.train_rate)
        num_test = int(len_data * self.test_rate)
        
        border1s = [0, num_train - self.seq_len, len_data - num_test - self.seq_len]
        border2s = [num_train, num_train + (len_data - num_train - num_test), len_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = df_data.values[border1:border2] if self.inverse else data[border1:border2]

    def _split_and_scale_specialized(self, df_data, df_ctr):
        """Helper for specialized datasets with separate data and control/stamp dataframes."""
        len_data = len(df_data)
        num_train = int(len_data * self.train_rate)
        num_test = int(len_data * self.test_rate)
        
        border1s = [0, num_train - self.seq_len, len_data - num_test - self.seq_len]
        border2s = [num_train, num_train + (len_data - num_train - num_test), len_data]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            self.data_scaler = StandardScaler()
            self.data_scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data = self.data_scaler.transform(df_data.values)

            self.ctr_scaler = StandardScaler()
            self.ctr_scaler.fit(df_ctr.iloc[border1s[0]:border2s[0]].values)
            ctr = self.ctr_scaler.transform(df_ctr.values)
        else:
            data = df_data.values
            ctr = df_ctr.values

        self.data_x = data[border1:border2]
        self.data_stamp = ctr[border1:border2]
        self.data_y = df_data.values[border1:border2] if self.inverse else data[border1:border2]

    def __getitem__(self, index):
        s_begin, s_end, r_begin, r_end = self._get_indices(index)
        seq_x, seq_y, seq_x_mark, seq_y_mark = self._get_data_from_indices(s_begin, s_end, r_begin, r_end)

        if s_begin >= self.pred_len:
            us_begin = s_begin - self.pred_len
            us_end = us_begin + self.seq_len
            ur_begin = us_end - self.label_len
            ur_end = ur_begin + self.label_len + self.pred_len
            update_x, update_y, update_x_mark, update_y_mark = self._get_data_from_indices(us_begin, us_end, ur_begin, ur_end)
            update_flag = np.ones_like(seq_x)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, update_x, update_y, update_x_mark, update_y_mark, update_flag
        else:
            zeros = np.zeros_like(seq_x)
            zeros_y = np.zeros_like(seq_y)
            zeros_mark = np.zeros_like(seq_x_mark)
            zeros_y_mark = np.zeros_like(seq_y_mark)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, zeros, zeros_y, zeros_mark, zeros_y_mark, zeros

    def _get_indices(self, index):
        s_begin = index * self.pred_len if self.delay_fb and self.set_type == 2 else index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return s_begin, s_end, r_begin, r_end

    def _get_data_from_indices(self, s_begin, s_end, r_begin, r_end):
        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], axis=0) if self.inverse else self.data_y[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.delay_fb and self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len + 1) // self.pred_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        scaler = getattr(self, 'scaler', None) or getattr(self, 'data_scaler', None)
        if scaler:
            return scaler.inverse_transform(data)
        raise RuntimeError("Scaler has not been fitted.")

    def get_data_full(self):
        return self.data_x, self.data_y, self.data_stamp
    
    def get_restart_data(self, st, ed):
        return self.data_x[st:ed], self.data_stamp[st:ed]