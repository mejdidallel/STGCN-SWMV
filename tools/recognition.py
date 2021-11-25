import sys
import argparse
import yaml
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchlight
import matplotlib as mpl
import itertools

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from .processor import Processor
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from focal_loss import *

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
						  dataset='uow'):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.cm.rainbow
    if normalize:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float)
    if dataset=='oad': 
        plt.figure(figsize=(15, 12))
    else:
        plt.figure(figsize=(23, 18))
    plt.imshow(cm_perc, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15, weight='bold')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.clim(0,1) 
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=13, weight='bold')
        plt.yticks(tick_marks, target_names, rotation=45, fontsize=13, weight='bold')

    thresh = cm_perc.max() / 1.5 if normalize else cm_perc.max() / 2
    for i, j in itertools.product(range(cm_perc.shape[0]), range(cm_perc.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm_perc[i, j]),
                     horizontalalignment="center", fontsize=13, weight='bold',
                     color="white" if cm_perc[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm_perc[i, j]),
                     horizontalalignment="center", fontsize=13, weight='bold',
                     color="white" if cm_perc[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14, weight='bold')
    plt.xlabel('Predicted label\nAccuracy={:0.2f}%; Misclass={:0.2f}%'.format(accuracy*100, misclass*100), fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('rsc/'+title+'.png')
    plt.show()	
	
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
		
def f1score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')
	
	
def compute_iou(y_pred, y_true, labels, threshold=0.8):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    acc = intersection / (sum(cm))
    cm_sum=0
    for i in range(len(acc)):
        if (acc[i]>0):
            cm_sum += acc[i]
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union
    for i in range(len(IoU)):
        if (IoU[i]>threshold):
            IoU[i] = 1
    return round(np.mean(IoU)*100, 3)
		
class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = FocalLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr
			
    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.10f}'.format(self.optimizer.param_groups[0]['lr'])
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
        
        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        if self.arg.dataset=='OAD':
            labels_name = ['No action','Drinking','Eating','Writing','Opening cupboard','Washing hands','Opening microwave oven','Sweeping','Gargling','Throwing trash','Wiping']
            labels_ids = [0,1,2,3,4,5,6,7,8,9,10]
        elif self.arg.dataset=='UOW':
            labels_name = ['No Action','High arm wave','Horizontal arm wave','Hammer','Hand catch','Forward punch','High Throw','Draw X','Draw Tick','Draw circle','Hand clap','Two Hand wave','Side boxing','Bend','Forward kick','Side kick','Jogging','Tennis swing','Tennis serve','Golf swing','Pick up and Throw']
            labels_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        for data, label in loader:
            
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()
            conf_matrix = confusion_matrix(np.argmax(self.result, axis=1), self.label)
            rank = self.result.argsort()
            with open('rsc/'+self.arg.dataset+'_Prediction.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(rank[:,rank.shape[1]-1])
            with open('rsc/'+self.arg.dataset+'_GroundTruth.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self.label)
				
            prediction = rank[:,rank.shape[1]-1]
            groundtruth = self.label
            
            iou=compute_iou(prediction, groundtruth, labels_ids, threshold=0.8)	
				
            print("Non normalized confusion matrix :\n", conf_matrix)
            plot_confusion_matrix(cm= conf_matrix, normalize = True, target_names = labels_name, title= self.arg.dataset+" Confusion Matrix", dataset=self.arg.dataset)
            print("\nGroundtruth labels: ", self.label)
            print("\nPredicted labels: ", rank[:,rank.shape[1]-1])
            
            hit_top = [l in rank[i, -1:] for i, l in enumerate(self.label)]
            accuracy = sum(hit_top) * 1.0 / len(hit_top)
            self.io.print_log('Accuracy: {:.2f}%'.format(100 * accuracy))
            self.io.print_log('F1-Score: {:.2f}%'.format(100 * f1score(np.argmax(self.result, axis=1), self.label)))
            self.io.print_log('IoU: {:.2f}%'.format(iou))
            self.io.print_log('Confusion matrix, groundtruth and predicted labels are saved in /rsc folder.')

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='STGCN-SWMV')

        parser.add_argument('--dataset', default='OAD', help='Test dataset OAD or UOW')
        parser.add_argument('--base_lr', type=float, default=0.01, help='Initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='Epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='Type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='Use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')

        return parser