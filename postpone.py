#! /usr/bin/python3
import os
import time
import sys
from argparse import ArgumentParser
from pathlib import Path
from subprocess import call

parser = ArgumentParser('gpu mover')
parser.add_argument('--config','-c', type=str, default='experiment/tiny-imagenet_resnet.a1.yaml')
parser.add_argument('--n-gpus', '-n', type=int,default=3)
parser.add_argument('--interval', '-i', type=int, default=5,)
parser.add_argument('--dir', '-d', type=str, )
parser.add_argument('--query', '-q', nargs='*', type=str)

def _parse_args():
    args = parser.parse_args()

    return args


class GPUGet:
    def __init__(self,
                 min_gpu_number,
                 time_interval):
        self.min_gpu_number = min_gpu_number
        self.time_interval = time_interval

    def get_gpu_info(self):
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')[1:]
        gpu_dict = dict()
        for i in range(len(gpu_status) // 4):
            index = i * 4
            gpu_state = str(gpu_status[index].split('   ')[2].strip())
            gpu_power = int(gpu_status[index].split('   ')[-1].split('/')[0].split('W')[0].strip())
            gpu_memory = int(gpu_status[index + 1].split('/')[0].split('M')[0].strip())
            gpu_dict[i] = (gpu_state, gpu_power, gpu_memory)
        return gpu_dict

    def loop_monitor(self, prohibited_gpus=[]):
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                if gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000 and i not in prohibited_gpus:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                    gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                    sys.stdout.write(gpu_str)
                    sys.stdout.flush()
                    available_gpus.append(i)
            if len(available_gpus) >= self.min_gpu_number:
                return available_gpus
            else:
                available_gpus = []
                time.sleep(self.time_interval)

    def run(self, cmd_parameter, cmd_command, py_parameters, errFile=None,prohibited_gpus=[]):
        available_gpus = self.loop_monitor(prohibited_gpus)
        
        gpu_list_str = " ".join(map(str, available_gpus[:self.min_gpu_number]))
        # 构建终端命令
        cmd_parameter = fr"""{cmd_parameter} 
                          NUM_GPUS={len(available_gpus)}"""  # 一定要有 `; \ `
        #cmd_command = fr"""{cmd_command}"""
        command = fr"""{cmd_parameter} {cmd_command} {gpu_list_str} {py_parameters}"""
        print(command)
        try:
            ret = call(command, shell=True)
            f = None
            if errFile is not None:
                with open(errFile, 'a') as f:
                    if ret < 0:
                        print("Chind terminated by signal", -ret, py_parameters, file=f)
                    else:
                        print("Child retruned", ret, py_parameters, file=f)
        except OSError as e:
            print('file not exists', e)


def get_paths(path, query=None):
    candidate = os.listdir(path)
    result = []
    for config in candidate:
        if not config.endswith('.yaml'):
            continue
        flag = 0
        if query is not None:
            for q in query:
                if q not in config:
                    flag = 1
                    break
        if flag == 0:
            result.append(config)
    return result


if __name__ == '__main__':
    args = _parse_args()
    prohibited_gpus = [4,5,6,7]
    min_gpu_number = args.n_gpus  # 最小GPU数量，多于这个数值才会开始执行训练任务。
    time_interval = args.interval  # 监控GPU状态的频率，单位秒。
    gpu_get = GPUGet(min_gpu_number, time_interval)
    config_file = args.config
    cmd_parameter = r""""""  # 命令会使用到的参数，使用 `;` 连接。
    cmd_command = fr"""./distributed_train.sh"""
    err_file = 'errors.txt'
    success_files = []
    success_lines = []
    try:
        with open(err_file, 'r') as file_log:
            for line in file_log:
                entries = line.strip('\n').split()
                success = int(entries[2])
                if success == 0:
                    success_files.append(entries[4])
                    success_lines.append(line)
        with open(err_file, 'w') as file_log:
            for line in success_lines:
                file_log.write(line)
    except OSError as e:
           pass
    print('files successful passing:', success_files)
    if args.dir:
        clean_path = Path(args.dir)
        # if clean_path.is_dir():
        #     for config in [os.path.join(clean_path, x) for x in get_paths(clean_path, args.query)]:
        #         py_parameters = f'--config { config }'
        #         if config not in success_files:
        #             gpu_get.run(cmd_parameter, cmd_command, py_parameters, errFile=err_file,prohibited_gpus=prohibited_gpus)
        # if poison_path.is_dir():
        #     for config in [os.path.join(poison_path, x) for x in get_paths(poison_path, args.query)]:
        #         py_parameters = f'--config { config }'
        #         if config not in success_files:
        #             gpu_get.run(cmd_parameter, cmd_command, py_parameters, errFile=err_file,prohibited_gpus=prohibited_gpus)
        if clean_path.is_dir():
            for config in [os.path.join(clean_path, x) for x in get_paths(clean_path, args.query)]:
                py_parameters = f'--config { config }'
                if config not in success_files:
                    gpu_get.run(cmd_parameter, cmd_command, py_parameters, errFile=err_file,prohibited_gpus=prohibited_gpus)
    else:        
        py_parameters = f'--config {config_file}'
        gpu_get.run(cmd_parameter, cmd_command, py_parameters)

