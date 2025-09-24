import numpy as np
import pandas as pd
import yaml
import scipy
import re
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from data.base_loder import Dataset_base

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_TEP(Dataset_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_data()

    def _load_data(self):
        self.scaler = StandardScaler()
        if self.set_type == 0:
            df_raw = pd.read_csv('./dataset/TFP/train/normal_data.csv')
        elif self.set_type == 1:
            df_raw = pd.read_csv('./dataset/TFP/val/normal_data.csv')
        else:
            df_raw = pd.read_csv(self.data_path)

        df_raw = df_raw.iloc[:,[5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 19]]

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

        if self.scale:
            train_data = pd.read_csv('./dataset/TFP/train/normal_data.csv')
            train_data = train_data.iloc[:,[5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 19]].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data
        self.data_y = df_data.values if self.inverse else data


class Dataset_custom(Dataset_base):
    def __init__(self, root_path, train_rate, test_rate, data_path,
                 flag='train', size=None, delay_fb=False, features='M', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='s', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__(root_path, train_rate, test_rate, data_path, flag, size, delay_fb, 
                         features, target, scale, inverse, timeenc, freq, cols)
        
        self.__read_data__()


class Dataset_restart(Dataset_base):
    def __init__(self, root_path, train_rate, test_rate, data_path, data, data_stamp, 
                 flag='train', size=None, delay_fb=False, features='M', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='s', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        super().__init__(root_path, train_rate, test_rate, data_path, flag, size, delay_fb, 
                         features, target, scale, inverse, timeenc, freq, cols)
        
        self.data = data.copy()
        self.data_x = data.copy()
        self.data_y = data.copy()
        self.data_stamp = data_stamp.copy()