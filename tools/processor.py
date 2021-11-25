import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchlight

from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO

class Processor(IO):
    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        Loader = import_class(self.arg.loader)
        if 'debug' not in self.arg.train_loader_args:
            self.arg.train_loader_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Loader(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_loader_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Loader(**self.arg.test_loader_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log('Done.')
        elif self.arg.phase == 'test':

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))
            self.io.print_log('STGCN-SWMV Evaluation:')
            self.test()
            self.io.print_log('Done.\n')

            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        parser = argparse.ArgumentParser(add_help=add_help, description='STGCN-SWMV Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/temp', help='Folder where to store results')
        parser.add_argument('-c', '--config', default=None, help='Configuration file path')

        parser.add_argument('--phase', default='train', help='Train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='If True : Store the output of the model')
        parser.add_argument('--start_epoch', type=int, default=0, help='Start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='Stop training at which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='Indexes of GPUs for training or testing')

        parser.add_argument('--log_interval', type=int, default=100, help='Interval for printing messages')
        parser.add_argument('--save_interval', type=int, default=1, help='Interval for storing models')
        parser.add_argument('--eval_interval', type=int, default=1, help='Interval for evaluating models')
        parser.add_argument('--save_log', type=str2bool, default=True, help='Save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='Print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='Logging on pavi or not')

        parser.add_argument('--loader', default='tools.loader', help='Data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='Number of worker per gpu for data loader')
        parser.add_argument('--train_loader_args', action=DictAction, default=dict(), help='Arguments of data loader for training')
        parser.add_argument('--test_loader_args', action=DictAction, default=dict(), help='Arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='Test batch size')
        parser.add_argument('--debug', action="store_true", help='Less data, Faster loading')

        parser.add_argument('--model', default=None, help='Model to be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='Arguments of model')
        parser.add_argument('--weights', default=None, help='Weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='Name of weights that will be ignored in the initialization')

        return parser