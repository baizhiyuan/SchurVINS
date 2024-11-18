#!/usr/bin/python3
"""
@author: Christian Forster
"""

import os
import sys
import yaml
import rospkg
import argparse
import time
import ros_node
import evaluate
import shutil

import subprocess
import re
from colorama import init, Fore
init(autoreset=True)


def get_cpu_info():
    command = "cat /proc/cpuinfo"
    all_info = subprocess.check_output(command, shell=True).decode('utf-8').strip()
    for line in all_info.split("\n"):
        if "model name" in line:
            model_name = re.sub(".*model name.*:", "", line, 1).strip()
            return model_name.replace("(R)", "").replace("(TM)", "")


def copy_files_to_tracedir(dataset_dir, trace_dir, files_to_copy=None):
    essential_files = ['data/stamped_groundtruth.txt',
                       'data/groundtruth.txt',
                       'data/groundtruth_matches.txt',
                       'dataset.yaml',
                       'evaluation_scripts.yaml']
    for f in essential_files:
        if os.path.isfile(os.path.join(dataset_dir, f)):
            print(f"Copy {f} to the trace folder.")
            shutil.copyfile(os.path.join(dataset_dir, f),
                            os.path.join(trace_dir, os.path.basename(f)))

    if files_to_copy:
        print('Copy dataset specific files.')
        for fn in files_to_copy:
            if os.path.isfile(os.path.join(dataset_dir, fn)):
                print(f'Copy {fn} to trace directory.')
                shutil.copyfile(os.path.join(dataset_dir, fn),
                                os.path.join(trace_dir, os.path.basename(fn)))


def run_single_experiment(params, node='svo_ros', node_name='benchmark'):
    # Load dataset parameters
    params['dataset_is_blender'] = False
    dataset_params_file = os.path.join(params['dataset_directory'], 'dataset.yaml')
    if os.path.exists(dataset_params_file):
        with open(dataset_params_file, 'r') as file:
            dataset_params = yaml.load(file, Loader=yaml.FullLoader)
            params.update(dataset_params)

    # Check that calibration file exists
    calib_name = params.get('calib_name', 'calib.yaml')
    params['calib_file'] = os.path.join(params['dataset_directory'], calib_name)
    if not os.path.exists(params['calib_file']):
        raise Exception(f'File does not exist: {params["calib_file"]}')

    shutil.copyfile(params['calib_file'], os.path.join(params['trace_dir'], 'calib.yaml'))

    # execute ros node
    os.system('rosparam delete /svo')
    node = ros_node.RosNode(node, node_name)
    node.run(params, '', True, params['trace_dir'])
    os.system(f'rosparam dump {os.path.join(params["trace_dir"], "params_ros.yaml")}')

    # dump experiment params to file and copy the other parameter files:
    params['platform'] = get_cpu_info()
    params['ros_node'] = node
    params['ros_node_name'] = node_name
    params_dump_file = os.path.join(params['trace_dir'], 'params.yaml')
    with open(params_dump_file, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


def run_experiments(experiment_file, num_monte_carlo_runs=None, align_type='posyaw', n_aligned=-1):
    experiment_file = experiment_file.replace('.yaml', '')
    experiment_params_file = os.path.join(
        rospkg.RosPack().get_path('svo_benchmarking'),
        'experiments', experiment_file + '.yaml')

    # Load base algorithm parameters.
    with open(experiment_params_file, 'r') as file:
        experiment_params = yaml.load(file, Loader=yaml.FullLoader)

    base_params = {
        'experiment_label': experiment_params['experiment_label'],
        'time': time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        'cooldown_interval_sec': experiment_params.get('cooldown_interval_sec', 10)
    }

    # Pipeline parameters
    if 'settings' in experiment_params:
        base_params.update(experiment_params['settings'])

    # Benchmark node flag: verbosity, etc.
    base_params['flags'] = experiment_params.get('flags', dict())

    # Get base folder where trace-folder is created.
    trace_base_dir = os.path.join(
        rospkg.RosPack().get_path('svo_benchmarking'), 'results')
    if 'trace_base_dir' in experiment_params:
        trace_base_dir = experiment_params['trace_base_dir']
        print(f"Overwrite the default trace directory with {trace_base_dir}.")

    if num_monte_carlo_runs is None:
        num_monte_carlo_runs = int(experiment_params.get('num_monte_carlo_runs', 1))

    # Store the trace directories of all datasets
    trace_dirs = []

    # Loop over each dataset
    for dataset_cfg in experiment_params['datasets']:
        # Overwrite and extend base parameters with dataset specific settings:
        cur_params = base_params.copy()
        if 'settings' in dataset_cfg:
            cur_params.update(dataset_cfg['settings'])

        # Check if dataset directory exists
        cur_params['dataset_directory'] = os.path.join(
            rospkg.RosPack().get_path('svo_benchmarking'),
            'data', dataset_cfg['name'])
        assert os.path.exists(cur_params['dataset_directory']), \
            f"Dataset {cur_params['dataset_directory']} does not exist."

        # Load dataset parameters
        cur_params['dataset_is_blender'] = False
        dataset_params_file = os.path.join(cur_params['dataset_directory'], 'dataset.yaml')
        if os.path.exists(dataset_params_file):
            with open(dataset_params_file, 'r') as file:
                dataset_params = yaml.load(file, Loader=yaml.FullLoader)
                cur_params.update(dataset_params)
        else:
            # 如果未找到 dataset.yaml 文件，设置 dataset_name
            cur_params['dataset_name'] = dataset_cfg.get('name', 'default_dataset_name')

        # 在打印之前检查是否设置了 'dataset_name'
        if 'dataset_name' not in cur_params:
            cur_params['dataset_name'] = dataset_cfg.get('name', 'default_dataset_name')
            print("Warning: 'dataset_name' was not found, setting to default or using the name from dataset config.")

        # Trace directory
        trace_name = dataset_cfg.get('trace_name', dataset_cfg['name'].split('/')[-1])
        dataset_trace_dir = f"{cur_params['time']}_{experiment_file}/{trace_name}"
        cur_params['trace_dir'] = os.path.join(trace_base_dir, dataset_trace_dir)
        os.makedirs(cur_params['trace_dir'], exist_ok=True)
        trace_dirs.append((dataset_cfg['name'], cur_params['trace_dir']))

        print(Fore.RED + f">>>>> Start experiments for {cur_params['dataset_name']}...")
        print(Fore.RED + f">>>>> Will do {num_monte_carlo_runs} MC runs.")
        print(Fore.RED + f">>>>> Trace will be saved in {cur_params['trace_dir']}.")

        # Copy essential files
        shutil.copyfile(experiment_params_file,
                        os.path.join(cur_params['trace_dir'], 'experiment.yaml'))

        # Copy the groundtruth trajectory to trace dir
        files_to_copy = cur_params.get('files_to_copy', [])
        copy_files_to_tracedir(cur_params['dataset_directory'], cur_params['trace_dir'], files_to_copy)

        # Create evaluation config file
        eval_cfg = {'align_type': align_type, 'align_num_frames': n_aligned}
        print(f"Will write evaluation config: {eval_cfg}")
        with open(os.path.join(cur_params['trace_dir'], 'eval_cfg.yaml'), 'w') as f:
            yaml.dump(eval_cfg, f, default_flow_style=False)

        # Run Monte-Carlo Iterations
        for trial_idx in range(num_monte_carlo_runs):
            print(Fore.RED + f">>> Trial {trial_idx}...")

            params_i = cur_params.copy()
            params_i['flags']['trial_idx'] = -1 if num_monte_carlo_runs == 1 else trial_idx

            # Run experiment
            run_single_experiment(params_i, experiment_params['ros_node'], experiment_params['ros_node_name'])
            print("Cooling down...")
            s_total = params_i['cooldown_interval_sec']
            for sl_i in range(s_total):
                time.sleep(1)
                sys.stdout.write("\r")
                sys.stdout.write(f"{sl_i + 1}/{s_total}")

    return trace_dirs


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='Runs SVO with the dataset and parameters specified in the provided experiment file.')
    parser.add_argument('experiment_file', help='experiment file in svo_benchmarking/experiments folder')
    parser.add_argument('--n_trials', type=int, help='number of Monte-Carlo runs', default=None)
    parser.add_argument('--align_type', type=str, help='Alignment type in evaluation', default='posyaw')
    parser.add_argument('--n_aligned', type=int, help='Number of frames to be aligned', default=-1)
    args = parser.parse_args()

    run_experiments(args.experiment_file, args.n_trials, args.align_type)
