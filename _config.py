import argparse
import os
from collections import defaultdict

import torch
import yaml
from pathlib import Path

from utils.timefeatures import time_features_from_frequency_str

SETTINGS_DIR = Path('./settings')
DATA_PARSE_CONFIG_PATH = SETTINGS_DIR / 'data_parse.yaml'


class ExperimentConfig:
    """Class to manage experiment configurations."""

    def __init__(self):
        self.args = self._parse_arguments()
        self._load_and_merge_configs()
        self._finalize_config()
        self.print_arguments_by_group()

    def _add_argument_groups(self, parser: argparse.ArgumentParser):
        # =====================================================================================
        # Group 1: Experiment Settings
        # =====================================================================================
        exp_group = parser.add_argument_group('Experiment Settings', 'Basic settings that define the experiment')
        exp_group.add_argument('--method', type=str, default='TimeXer', help='Model/method name to use')
        exp_group.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name (e.g., long_term_forecast, anomaly_detection)')
        exp_group.add_argument('--root_path', type=str, default='./dataset', help='Root path of the data file')
        exp_group.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
        exp_group.add_argument('--des', type=str, default='test', help='Experiment description suffix')
        exp_group.add_argument('--suffix', type=str, help='A custom suffix for the experiment id')
        exp_group.add_argument('--itr', type=int, default=1, help='Number of experiment iterations')
        exp_group.add_argument('--exp_seed', type=int, default=-1, help='Random seed for the experiment')


        # =====================================================================================
        # Group 2: Data Settings
        # =====================================================================================
        data_group = parser.add_argument_group('Data Settings', 'Settings for data loading, splitting, and preprocessing')
        data_group.add_argument('--data', type=str, default='ETTh2', help='Dataset name')
        data_group.add_argument('--data_path', type=str, default='ETTh2.csv', help='Data file name')
        data_group.add_argument('--test_type', type=str, default='full', help='Test type (e.g., each, full)')
        data_group.add_argument('--features', type=str, default='M', help='Forecasting task type: M, S, MS')
        data_group.add_argument('--target', type=str, default='OT', help='Target feature name for S or MS tasks')
        data_group.add_argument('--freq', type=str, default='s', help='Frequency for time features (s, t, h, d, b, w, m)')
        data_group.add_argument('--cols', type=str, nargs='+', help='Use only specific columns as input features')
        data_group.add_argument('--train_rate', type=float, default=0.7, help='Proportion of data for training')
        data_group.add_argument('--test_rate', type=float, default=0.2, help='Proportion of data for testing')
        data_group.add_argument('--use_norm', type=int, default=1, help='Whether to normalize the data (1=True, 0=False)')
        data_group.add_argument('--scale', type=int, default=1, help='Normalize the data (1=True, 0=False)')
        data_group.add_argument('--inverse', action='store_true', help='Inverse transform the output data', default=False)



        # =====================================================================================
        # Group 3: Training Settings
        # =====================================================================================
        train_group = parser.add_argument_group('Training Settings', 'Settings for the training loop and optimization')
        train_group.add_argument('--train_epochs', type=int, default=3, help='Number of training epochs')
        train_group.add_argument('--batch_size', type=int, default=32, help='Training batch size')
        train_group.add_argument('--test_bsz', type=int, default=1, help='Test batch size')
        train_group.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
        train_group.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate')
        train_group.add_argument('--loss', type=str, default='mse', help='Loss function')
        train_group.add_argument('--lradj', type=str, default='type1', help='Learning rate adjustment strategy')
        train_group.add_argument('--opt', type=str, default='adam', help='Optimizer to use')
        train_group.add_argument('--weight_decay', type=float, default=1e-3, help='Optimizer weight decay')
        train_group.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training', default=False)
        train_group.add_argument('--grid_search', type=int, default=1, help='Enable grid search for hyperparameters (1=True, 0=False)')
        train_group.add_argument('--finetune', action='store_true', default=False, help='Finetune the model')
        train_group.add_argument('--aug', type=int, default=0, help='Number of data augmentation iterations')
        train_group.add_argument('--loss_aug', type=float, default=0.5, help='Weight for the augmentation loss')


        # =====================================================================================
        # Group 4: Execution Settings
        # =====================================================================================
        exec_group = parser.add_argument_group('Execution Settings', 'Settings for the execution environment')
        exec_group.add_argument('--use_gpu', type=bool, default=True, help='Use GPU if available')
        exec_group.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
        exec_group.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs', default=False)
        exec_group.add_argument('--devices', type=str, default='0,1', help='Device IDs for multiple GPUs')
        exec_group.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda, cpu)')
        exec_group.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loader')
        exec_group.add_argument('--num_works', type=int, default=1, help='Number of parallel processes for experiments')
        exec_group.add_argument('--single_core', action='store_true', help='Force single-threaded execution for performance-critical libraries')
        exec_group.add_argument('--verbose', type=int, default=0, help='Enable verbose output (1=True, 0=False)')
        exec_group.add_argument('--const_memory', type=int, help='Use constant memory (if applicable)')


        # =====================================================================================
        # Group 5: Core Model Architecture
        # =====================================================================================
        arch_group = parser.add_argument_group('Core Model Architecture', 'Core hyperparameters defining the model structure')
        arch_group.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
        arch_group.add_argument('--label_len', type=int, default=0, help='Start token length for decoder')
        arch_group.add_argument('--pred_len', type=int, default=1, help='Prediction sequence length')
        arch_group.add_argument('--enc_in', type=int, default=7, help='Encoder input size (number of features)')
        arch_group.add_argument('--dec_in', type=int, default=7, help='Decoder input size')
        arch_group.add_argument('--c_out', type=int, default=7, help='Output size')
        arch_group.add_argument('--d_model', type=int, default=32, help='Dimension of the model')
        arch_group.add_argument('--d_ff', type=int, default=128, help='Dimension of the feed-forward network')
        arch_group.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
        arch_group.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
        arch_group.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
        arch_group.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
        arch_group.add_argument('--activation', type=str, default='gelu', help='Activation function')
        arch_group.add_argument('--embed', type=str, default='timeF', help='Time feature encoding type (timeF, fixed, learned)')
        arch_group.add_argument('--padding', type=int, default=0, help='Padding type')
        arch_group.add_argument('--output_attention', action='store_true', help='Whether to output attention in encoder')
        arch_group.add_argument('--do_predict', action='store_true', help='Whether to predict unseen future data')


        # =====================================================================================
        # Group 6: Online Learning & Drift Detection
        # =====================================================================================
        online_group = parser.add_argument_group('Online Learning & Drift Detection', 'Settings for continual learning')
        online_group.add_argument('--online_learning', type=str, default='full', help='Online learning strategy')
        online_group.add_argument('--teacher_forcing', action='store_true', help='Use teacher forcing during forecasting', default=False)
        online_group.add_argument('--lr_test', type=float, default=1e-3, help='Learning rate during online testing/adaptation')
        online_group.add_argument('--drift_detection', action='store_true', help='Enable concept drift detection')
        online_group.add_argument('--detector', type=str, default='ADWIN', help='Concept drift detector to use')
        online_group.add_argument('--use_CALIPER', action='store_true', help='Enable CALIPER for drift detection')
        online_group.add_argument('--CALIPER_njob', type=int, default=-1, help='Number of jobs for CALIPER')


        # =====================================================================================
        # Group 7: Model-Specific: Transformer Family (Informer, FEDformer, etc.)
        # =====================================================================================
        trans_group = parser.add_argument_group('Model-Specific: Transformer Family', 'Arguments for Informer, FEDformer, etc.')
        trans_group.add_argument('--factor', type=int, default=5, help='ProbSparse attention factor (for Informer)')
        trans_group.add_argument('--atten_dim', type=int, default=64, help='dimension of various attention')
        trans_group.add_argument('--distil', action='store_false', help='Disable distilling in encoder (for Informer)', default=True)
        trans_group.add_argument('--attn', type=str, default='prob', help='Attention mechanism (prob, full) (for Informer)')
        trans_group.add_argument('--mix', action='store_false', help='Disable mix attention in decoder (for Informer)', default=True)
        trans_group.add_argument('--moving_avg', default=25, help='Window size of moving average (for Autoformer, DLinear, FEDformer)')
        trans_group.add_argument('--version', type=str, default='Wavelets', help='FEDformer version (Fourier, Wavelets)')
        trans_group.add_argument('--mode_select', type=str, default='random', help='FEDformer mode selection (random, low)')
        trans_group.add_argument('--modes', type=int, default=64, help='Number of modes for FEDformer')
        trans_group.add_argument('--L', type=int, default=3, help='Ignore level for FEDformer MWT')
        trans_group.add_argument('--base', type=str, default='legendre', help='MWT base for FEDformer')
        trans_group.add_argument('--cross_activation', type=str, default='tanh', help='MWT cross-activation for FEDformer (tanh, softmax)')
        trans_group.add_argument('--win_size', type=int, default=2, help='Window size for Crossformer')


        # =====================================================================================
        # Group 8: Model-Specific: TimeMixer
        # =====================================================================================
        mixer_group = parser.add_argument_group('Model-Specific: TimeMixer', 'Arguments for the TimeMixer model')
        mixer_group.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
        mixer_group.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
        mixer_group.add_argument('--down_sampling_method', type=str, default='avg', help='down sampling method, only support avg, max, conv')
        mixer_group.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')


        # =====================================================================================
        # Group 9: Model-Specific: PatchTST
        # =====================================================================================
        patch_group = parser.add_argument_group('Model-Specific: PatchTST', 'Arguments for the PatchTST model')
        patch_group.add_argument('--patch_len', type=int, default=16, help='Patch length')
        patch_group.add_argument('--stride', type=int, default=8, help='Stride for creating patches')
        patch_group.add_argument('--padding_patch', default='end', help='Padding for patches (None, end)')
        patch_group.add_argument('--revin', type=int, default=0, help='Enable RevIN (1=True, 0=False)')
        patch_group.add_argument('--affine', type=int, default=0, help='Enable affine transformation in RevIN (1=True, 0=False)')
        patch_group.add_argument('--subtract_last', type=int, default=0, help='Subtract last value instead of mean in RevIN (1=True, 0=False)')
        patch_group.add_argument('--individual', type=int, default=1, help='Enable individual head for each channel (1=True, 0=False)')
        patch_group.add_argument('--fc_dropout', type=float, default=0.05, help='Fully connected layer dropout')
        patch_group.add_argument('--head_dropout', type=float, default=0.0, help='Attention head dropout')
        

        # =====================================================================================
        # Group 10: Model-Specific: Model-Specific: diffusion
        # =====================================================================================
        diffusion_group = parser.add_argument_group('Model-Specific: diffusion', 'Arguments for the diffusion model')
        diffusion_group.add_argument('--time_steps', type=int, default=1000, help='time step of diffusion')
        diffusion_group.add_argument('--beta_start', type=float, default=0.0001, help='start of diffusion beta')
        diffusion_group.add_argument('--beta_end', type=float, default=0.02, help='end of diffusion beta')
        
        # =====================================================================================
        # Group 11: Model-Specific: Model-Specific: scikit-learn
        # =====================================================================================
        sklearn_group = parser.add_argument_group('Model-Specific: scikit-learn.')
        sklearn_group.add_argument('--krr_alpha', type=float, default=1.0, help='alpha for KRR')
        sklearn_group.add_argument('--krr_gamma', type=float, default=None, help='gamma for KRR')
        sklearn_group.add_argument('--krr_degree', type=int, default=3, help='degree for KRR')
        sklearn_group.add_argument('--krr_coef0', type=float, default=1.0, help='coef0 for KRR')
        sklearn_group.add_argument('--krr_kernel', type=str, default='rbf', help='kernel for KRR')
        sklearn_group.add_argument('--ln_alpha', type=float, default=1.0, help='alpha for Linear')

        # =====================================================================================
        # Group 12: Model-Specific: Other Architectures
        # =====================================================================================
        other_arch_group = parser.add_argument_group('Model-Specific: Other Architectures', 'Arguments for TCN, ReLiNet, etc.')
        other_arch_group.add_argument('--top_k', type=int, default=5, help='Top-k frequencies for TimesBlock/TimesNet')
        other_arch_group.add_argument('--num_kernels', type=int, default=6, help='Number of kernels for InceptionTime')
        other_arch_group.add_argument('--channel_independence', type=int, default=1, help='Channel independence (1=True) or dependence (0=False)')
        other_arch_group.add_argument('--decomposition', type=int, default=0, help='Enable series decomposition (1=True, 0=False)')
        other_arch_group.add_argument('--kernel_size', type=int, default=25, help='Kernel size for decomposition')
        other_arch_group.add_argument('--recurrent_dim', type=int, default=64, help='Dimension of recurrent layers (for ReLiNet)')
        other_arch_group.add_argument('--num_recurrent_layers', type=int, default=2, help='Number of recurrent layers (for ReLiNet)')
        other_arch_group.add_argument('--tcn_output_dim', type=int, default=320, help='Output dimension for TCN block')
        other_arch_group.add_argument('--tcn_layer', type=int, default=2, help='Number of layers in TCN block')
        other_arch_group.add_argument('--tcn_hidden', type=int, default=32, help='Hidden dimension in TCN block')
        other_arch_group.add_argument('--tcn_ksize', type=int, default=3, help='kernel_size in TCN block')
        other_arch_group.add_argument('--tcn_head', type=str, default='last', help='head type in TCN')
        other_arch_group.add_argument('--t', type=int, default=500, help='time step of adding noise')
        other_arch_group.add_argument('--p', type=float, default=10.00, help='peak value of trend disturbance')
        other_arch_group.add_argument('--d', type=int, default=30, help='shift of period')
        other_arch_group.add_argument('--q', type=float, default=0.01, help='init anomaly probability of spot')
        other_arch_group.add_argument('--block_num', type=int, default=2, help='num of various block')
        other_arch_group.add_argument('--memory_len', type=int, default=2048, help="size of memory (for MemStream)")
        other_arch_group.add_argument('--mem_gamma', type=float, default=0, help="knn coefficient (for MemStream)")
        other_arch_group.add_argument('--mem_beta', type=float, default=0.1, help="(for MemStream)")
        other_arch_group.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate (OneNet)')
        other_arch_group.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate (OneNet)')
        other_arch_group.add_argument('--head', type=str, default='last', help='head type of TCN')


        # =====================================================================================
        # Group 13: Miscellaneous & Custom
        # =====================================================================================
        misc_group = parser.add_argument_group('Miscellaneous & Custom', 'Other custom or less common settings')
        misc_group.add_argument('--anomaly_ratio', type=float, default=0.25, help='Prior anomaly ratio for anomaly detection tasks')
        misc_group.add_argument('--rho', type=float, default=0.9, help='Forgetting rate')
        misc_group.add_argument('--R', type=int, default=-1, help='Target rank for matrix decomposition')
        misc_group.add_argument('--n_inner', type=int, default=1, help='Number of inner loops')
        misc_group.add_argument('--num_buffer', type=int, default=1, help='Number of buffers (for OneNet)')
        misc_group.add_argument('--use_adbfgs', action='store_true', help='Use the Adbfgs optimizer', default=True)
        misc_group.add_argument('--mlp_depth', type=int, default=3, help='MLP depth')
        misc_group.add_argument('--mlp_width', type=int, default=256, help='MLP width')
        misc_group.add_argument('--delay_dim', type=int, default=10, help='Delay embedding dimension (for modeplait)')
        misc_group.add_argument('--err_th', type=float, default=0.4, help='Error threshold (for modeplait)')
        misc_group.add_argument('--delay_fb', action='store_true', default=False, help='Use delayed feedback')
        misc_group.add_argument('--s', type=int, default=3, help='delayed steps')
        misc_group.add_argument('--num_lds', type=int, default=10, help='Number of LDS')
        misc_group.add_argument('--fix_win', type=int, default=1, help='min window rate')
        # ... Add any other very specific or experimental arguments here ...

    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(description='Time-Series Forecasting and Analysis Framework')
        self._add_argument_groups(parser)
        self.parser = parser
        return parser.parse_args()
    
    def _load_and_merge_configs(self):
        """Load settings from YAML files and merge them with arguments."""
        with open(DATA_PARSE_CONFIG_PATH, 'r') as f:
            data_parser = yaml.safe_load(f)

        data_info = data_parser.get(self.args.data)
        if not data_info:
            raise ValueError(f"Dataset '{self.args.data}' not found in {DATA_PARSE_CONFIG_PATH}")
        
        # Use defaultdict to safely get values.
        data_defaults = defaultdict(lambda: None, data_info)

        # Update arguments with settings from the YAML file.
        self.args.data_path = data_defaults['data_path']
        self.args.target = data_defaults['T']
        self.args.enc_in, self.args.dec_in, self.args.c_out = data_defaults[self.args.features]
        self.args.test_type = data_defaults['test_type']
        self.args.detail_freq = data_defaults['freq']
        self.args.freq = self.args.detail_freq[-1] if self.args.detail_freq else 'h'

        if data_defaults['enc_in_tf'] is None:
            self.args.enc_in_tf = len(time_features_from_frequency_str(self.args.detail_freq))
        else:
            self.args.enc_in_tf = data_defaults['enc_in_tf']

    def print_arguments_by_group(self):
        print("--- Experiment Configuration ---")
        for group in self.parser._action_groups:
            group_name = group.title
            if group_name == 'positional arguments':
                continue  
            print(f"=== {group_name} ===")
            for arg in group._group_actions:
                if not arg.option_strings:
                    continue
                dest = arg.dest
                opt_str = ', '.join(arg.option_strings)
                help_msg = arg.help or ""
                default = arg.default
                actual = getattr(self.args, dest, None)
                print(f"{opt_str:<25} | help: {help_msg:<45} | default: {str(default):<10} | value: {actual}")
            print()
        print("--------------------------------")

    def _finalize_config(self):
        """Perform final adjustments to the arguments."""
        self.args.use_gpu = torch.cuda.is_available() and self.args.use_gpu
        self.args.device = torch.device('cuda' if self.args.use_gpu else 'cpu')

        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ', '')
            self.args.device_ids = [int(id_) for id_ in self.args.devices.split(',')]
            self.args.gpu = self.args.device_ids[self.args.gpu]

        if hasattr(self.args, 's_layers') and isinstance(self.args.s_layers, str):
            self.args.s_layers = [int(s_l) for s_l in self.args.s_layers.replace(' ', '').split(',')]

        if self.args.single_core:
            self._set_threading_limits(1)


    @staticmethod
    def _set_threading_limits(num_threads: int = 1):
        """Limit the number of threads for major libraries."""
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
        os.environ["NUMBA_NUM_THREADS"] = str(num_threads)
        torch.set_num_threads(num_threads)