import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchlight

from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

class IO():
    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        p = parser.parse_args(argv)
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key
            parser.set_defaults(**default_arg)
        self.arg = parser.parse_args(argv)

    def init_environment(self):
        self.io = torchlight.IO(
            self.arg.work_dir,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log)
        self.io.save_arg(self.arg)

        if self.arg.use_gpu:
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))

    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights)

    def gpu(self):
        self.model = self.model.to(self.dev)
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help=False):

        parser = argparse.ArgumentParser( add_help=add_help, description='I/O Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/temp', help='Folder where to store results')
        parser.add_argument('-c', '--config', default=None, help='Configuration file path')

        parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='Indexes of GPUs for training or testing')

        parser.add_argument('--print_log', type=str2bool, default=True, help='Print logging or not')
        parser.add_argument('--save_log', type=str2bool, default=True, help='Save logging or not')

        parser.add_argument('--model', default=None, help='Model to be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='Arguments of model')
        parser.add_argument('--weights', default=None, help='Weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='Name of weights which will be ignored in the initialization')

        return parser