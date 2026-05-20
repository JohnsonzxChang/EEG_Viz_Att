import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from conf import BaseConfig
import os
import json 

class LoggerFile:
    def __init__(self, log_dir: str, name: str, conf_class:BaseConfig = None):
        self.log_dir = log_dir
        full_path = os.path.join(log_dir, name)
        os.makedirs(full_path, exist_ok=True)
        self.writer = SummaryWriter(full_path)
        if conf_class is not None:
            save_dir = self.writer.log_dir
            with open(f'{save_dir}/dict.json', 'w') as f:
                json.dump(conf_class.json(), f)
            print('save json file complete ...')


    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def save_fig(self, fig, step: int, name='regFig'):
        self.writer.add_figure(name, fig, step)

    def save_model(self, dict_data: dict, step: int):
        th.save(dict_data, f"{self.writer.log_dir}/checkpoint-{step}.pth")

    def close(self):
        self.writer.close()