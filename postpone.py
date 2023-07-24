#! /usr/bin/python3
import os
import time
import sys
from argparse import ArgumentParser


parser = ArgumentParser('gpu mover')
parser.add_argument('--config','-c', type=str, default='experiment/tiny-imagenet_resnet.a1.yaml')
parser.add_argument('--n-gpus', '-n', type=int,default=3)
parser.add_argument('--interval', '-i', type=int, default=5,)

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

    def loop_monitor(self):
        available_gpus = []
        while True:
            gpu_dict = self.get_gpu_info()
            for i, (gpu_state, gpu_power, gpu_memory) in gpu_dict.items():
                if gpu_state == "P8" and gpu_power <= 40 and gpu_memory <= 1000:  # 设置GPU选用条件，当前适配的是Nvidia-RTX3090
                    gpu_str = f"GPU/id: {i}, GPU/state: {gpu_state}, GPU/memory: {gpu_memory}MiB, GPU/power: {gpu_power}W\n "
                    sys.stdout.write(gpu_str)
                    sys.stdout.flush()
                    available_gpus.append(i)
            if len(available_gpus) >= self.min_gpu_number:
                return available_gpus
            else:
                available_gpus = []
                time.sleep(self.time_interval)

    def run(self, cmd_parameter, cmd_command, py_parameters):
        available_gpus = self.loop_monitor()
        
        gpu_list_str = " ".join(map(str, available_gpus[:self.min_gpu_number]))
        # 构建终端命令
        cmd_parameter = fr"""{cmd_parameter} 
                          NUM_GPUS={len(available_gpus)}"""  # 一定要有 `; \ `
        #cmd_command = fr"""{cmd_command}"""
        command = fr"""{cmd_parameter} {cmd_command} {gpu_list_str} {py_parameters}"""
        print(command)
        os.system(command)


if __name__ == '__main__':
    args = _parse_args()
    min_gpu_number = args.n_gpus  # 最小GPU数量，多于这个数值才会开始执行训练任务。
    time_interval = args.interval  # 监控GPU状态的频率，单位秒。
    gpu_get = GPUGet(min_gpu_number, time_interval)
    config_file = args.config
    py_parameters = f'--config {config_file}'
    cmd_parameter = r""""""  # 命令会使用到的参数，使用 `;` 连接。
    cmd_command = fr"""./distributed_train.sh"""
    gpu_get.run(cmd_parameter, cmd_command, py_parameters)
