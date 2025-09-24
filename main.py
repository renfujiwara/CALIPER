import argparse
import gc
import importlib
import multiprocessing
import os
import time
import random
import socket
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from _config import ExperimentConfig

# --- Constants Definition ---
# Manage paths for configuration files and directories as constants.
SETTINGS_DIR = Path('./settings')
MODELS_CONFIG_PATH = SETTINGS_DIR / 'models.yaml'
LOCAL_CONFIG_PATH = SETTINGS_DIR / 'local' / 'config.yaml'
RESULTS_DIR = Path('./_results')
CHECKPOINTS_DIR = Path('./checkpoints')
ROOT_PATH = './dataset'


def set_seed(seed: int):
    """Set random seeds for various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_class(args: argparse.Namespace) -> Tuple[Any, bool]:
    """Dynamically import the appropriate Experiment class based on task and method name."""
    task_map = {
        'long_term_forecast': 'exp_forecasting',
        'anomaly_detection': 'exp_anomaly_detection',
        # 'short_term_forecast': 'exp_short_term_forecast', # etc.
    }
    exp_dir = task_map.get(args.task_name)
    if not exp_dir:
        raise ValueError(f"Unsupported task: {args.task_name}")

    # Branch the import source depending on whether the model is from TSLib or a custom model.
    with open(MODELS_CONFIG_PATH, 'r') as yml:
        model_list = yaml.safe_load(yml)
    
    class_name = 'Exp_STSLib'

    if args.method in model_list.get('TSLib', []):
        module_name = f'{exp_dir}.exp_TSLib'
    elif args.method in model_list.get('scikit-learn', []):
        module_name = f'{exp_dir}.exp_scikit_learn'
        class_name = f'Exp_{args.method}'
    else:
        module_name = f'{exp_dir}.exp_{args.method}'
    
    try:
        exp_module = importlib.import_module(module_name)
        return getattr(exp_module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Could not import experiment class from {module_name}.")
        raise e


def run_single_trial(args: argparse.Namespace, config: Dict, setting: str) -> Tuple[Dict, Dict, Dict]:
    """
    Run a single trial (training and evaluation for a single seed).
    
    Returns:
        Tuple containing (metrics_dict, time_metrics_dict, artifacts_dict)
    """
    Exp = get_experiment_class(args)
    exp_instance = Exp(args)
    
    if args.grid_search == 1:
        exp_instance.grid_search(config, setting)

    if args.verbose:
        print(f">>> Training started for setting: {setting}")
    exp_instance.train(setting)
    gc.collect()
    if args.verbose:
        print(f"<<< Testing started for setting: {setting}")
    if args.use_gpu: torch.cuda.empty_cache()

    # Branch the test method based on whether drift detection is enabled.
    test_func = exp_instance.test_drift if args.drift_detection else exp_instance.test
    results = test_func(setting)

    # Parse results according to the task.
    if args.task_name == 'long_term_forecast':
        metrics, Results, Stats = results
        metrics_dict = {'MAE': Results['MAE'], 'MSE': Results['MSE']}
        artifacts = {'preds': Results['preds'], 'trues': Results['trues']}
    elif args.task_name == 'anomaly_detection':
        Results, Stats = results
        accuracy, precision, recall, f_score, _ = metrics
        metrics_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f_score}
        artifacts = {}
    else:
        raise ValueError(f"Unsupported task name for result parsing: {args.task_name}")
    del exp_instance
    gc.collect()
    return metrics_dict, Stats, artifacts


def process_file_experiment(file_path: Path, config: Dict, args: argparse.Namespace) -> Dict:
    """
    Run experiments for all seeds on a single data file.
    """
    file_name = file_path.stem
    args.data_path = str(file_path)
    
    all_results = []
    all_time_results = []
    all_size_result=[]
    log_metrics = defaultdict(list)
    
    uid = uuid.uuid4().hex[:4]

    for seed in config['random_state']:
        print(f"--- Processing: {file_name} | Seed: {seed} ---")
        args.exp_seed = seed
        set_seed(seed)
        
        setting = (
            f'{args.method}_pl{args.pred_len}_seed{seed}_ol{args.online_learning}_'
            f'opt{args.opt}_sfx{args.suffix}_{uid}_{file_name}'
        )

        try:
            metrics, time_metrics, artifacts = run_single_trial(args, config, setting)
            
            # --- Format results into a DataFrame ---
            method_name = f"{args.method}"
            if args.drift_detection:
                method_name += f"+{args.detector}"
                if getattr(args, 'use_DySAW', False):
                    method_name += "+DySAW"
                else:
                    method_name += f"+W_{args.fix_win}"

            if args.task_name == 'long_term_forecast':
                n = len(metrics['MSE'])
                df_result = pd.DataFrame({
                    "method": [method_name] * n, "dataset_name": [args.data] * n,
                    "data_name": [file_name] * n, "MAE": metrics['MAE'], "MSE": metrics['MSE'],
                    "seed": [seed] * n, "pred_len": np.arange(1, n + 1)
                })
                log_metrics['MSE'].append(np.mean(metrics['MSE']))
                log_metrics['MAE'].append(np.mean(metrics['MAE']))
            
            elif args.task_name == 'anomaly_detection':
                df_result = pd.DataFrame({
                    "method": [method_name], "dataset_name": [args.data],
                    "data_name": [file_name], **metrics, "seed": [seed]
                })
                for k, v in metrics.items(): log_metrics[k].append(v)
            
            # Record time and resource consumption.
            n_time = len(time_metrics['comp_time'])
            df_time = pd.DataFrame({
                "method": [method_name] * n_time, "dataset_name": [args.data] * n_time,
                "data_name": [file_name] * n_time, "time": time_metrics['comp_time'],
                "memory(cpu)": [time_metrics['cpu_memory']] * n_time,
                "memory(gpu)": [time_metrics['gpu_memory']] * n_time,
                "memory(total)": [time_metrics['total_memory']] * n_time,
                "params": [time_metrics['num_params']] * n_time, "seed": [seed] * n_time,
                "time_point": np.arange(args.seq_len + args.pred_len, args.seq_len + args.pred_len + n_time)
            })

            n_time = len(time_metrics['latency'])
            df_data_size = pd.DataFrame({
                "method": [method_name] * n_time, "dataset_name": [args.data] * n_time, 
                "data_name": [file_name] * n_time, "seed": [seed] * n_time,
                "latency": time_metrics['latency'], 'N': time_metrics['selected_size'],
                "train_start_time": time_metrics.get('train_start_time', [np.nan] * n_time),
                "drift_time": time_metrics.get('drift_time', [np.nan] * n_time)
            })
            
            all_results.append(df_result)
            all_time_results.append(df_time)
            all_size_result.append(df_data_size)
            log_metrics['Time'].append(np.sum(time_metrics['comp_time']))
            log_metrics['CPU_Mem_MB'].append(time_metrics['cpu_memory'] / 1024)
            log_metrics['GPU_Mem_MB'].append(time_metrics['gpu_memory'] / 1024)
            log_metrics['Total_Mem_MB'].append(time_metrics['total_memory'] / (1024**2))
            
        except Exception as e:
            print(f"ERROR during experiment for {file_name} with seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # --- Generate log messages ---
    log1 = ", ".join([f"{key}:{np.mean(val):.3f}" for key, val in log_metrics.items() if key not in ['Time', 'CPU_Mem_MB', 'GPU_Mem_MB', 'Total_Mem_MB']])
    log2 = (f"Time:{np.mean(log_metrics['Time']):.3f}s, "
            f"CPU_Mem:{np.mean(log_metrics['CPU_Mem_MB']):.2f}KB, "
            f"GPU_Mem:{np.mean(log_metrics['GPU_Mem_MB']):.2f}KB, "
            f"Total_Mem:{np.mean(log_metrics['Total_Mem_MB']):.2f}MB")
            
    return {
        'result': pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
        'result_time': pd.concat(all_time_results, ignore_index=True) if all_time_results else pd.DataFrame(),
        'result_size': pd.concat(all_size_result, ignore_index=True) if all_size_result else pd.DataFrame(),
        'log1': log1,
        'log2': log2
    }


def save_results(results_list: List[pd.DataFrame], time_list: List[pd.DataFrame], size_list, args: argparse.Namespace):
    """Save experiment results to CSV files."""
    if not results_list or not time_list:
        print("No results to save.")
        return

    folder_path = RESULTS_DIR / args.data / args.suffix
    folder_path.mkdir(parents=True, exist_ok=True)
    
    method_name = args.method
    if args.drift_detection:
        method_name += f"+{args.detector}"
        if getattr(args, 'use_DySAW', False):
            method_name += "+DySAW"
        else:
            method_name += f"+W_{args.fix_win}"

    save_path_err = folder_path / f'Errs_{method_name}.csv'
    save_path_scale = folder_path / f'Scale_{method_name}.csv'
    save_path_size = folder_path / f'Data_size_{method_name}.csv'

    pd.concat(results_list, ignore_index=True).to_csv(save_path_err, index=False)
    pd.concat(time_list, ignore_index=True).to_csv(save_path_scale, index=False)
    pd.concat(size_list, ignore_index=True).to_csv(save_path_size, index=False)
    print(f"Results saved to {save_path_err} and {save_path_scale} and {save_path_size}")


def main():
    """Main execution function."""
    print(f"Hostname: {socket.gethostname()}")
    
    # --- Load Configuration ---
    cfg_manager = ExperimentConfig()
    args = cfg_manager.args
    
    with open(LOCAL_CONFIG_PATH, 'r') as yml:
        local_config = yaml.safe_load(yml)

    # --- List Files to Process ---
    if args.test_type == 'each':
        files = [Path(ROOT_PATH, args.data_path)]
    else:
        files = list(Path(ROOT_PATH, args.data_path).glob('*'))

    if not files:
        print(f"No files found at {Path(ROOT_PATH, args.data_path)}")
        return

    # --- Determine Number of Workers ---
    num_works = min(args.num_works, len(files))
    if num_works <= 0:
        num_works = os.cpu_count() // 2 if os.cpu_count() else 1
    # Disable parallel processing under specific conditions.
    if args.verbose or args.grid_search:
        num_works = 1

    print(f"Number of parallel processes: {num_works}")
    print(">>>>>>> Starting Experiments >>>>>>>>>", flush=True)

    all_results, all_results_time, all_results_size = [], [], []

    if num_works == 1:
        for i, file in enumerate(files):
            result_dict = process_file_experiment(file, local_config, args)
            all_results.append(result_dict['result'])
            all_results_time.append(result_dict['result_time'])
            all_results_size.append(result_dict['result_size'])
            print(f"====== Finished {i+1}/{len(files)}: {file.name} ======")
            print(result_dict['log1'], flush=True)
            print(result_dict['log2'], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=num_works, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(process_file_experiment, file, local_config, args): file for file in files}
            
            for i, future in enumerate(as_completed(futures)):
                file = futures[future]
                try:
                    result_dict = future.result()
                    all_results.append(result_dict['result'])
                    all_results_time.append(result_dict['result_time'])
                    all_results_size.append(result_dict['result_size'])
                    print(f"====== Finished {i+1}/{len(files)}: {file.name} ======")
                    print(result_dict['log1'], flush=True)
                    print(result_dict['log2'], flush=True)
                except Exception as exc:
                    print(f"ERROR processing file {file.name}: {exc}", flush=True)

    # --- Save All Results ---
    save_results(all_results, all_results_time, all_results_size, args)
    print("<<<<<<< All Experiments Finished <<<<<<<")


if __name__ == '__main__':
    main()
