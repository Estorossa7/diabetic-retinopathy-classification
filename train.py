from datasetup import get_lists
from models import CNN
from hyparams import hyparams

from PIL import Image 
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

import torch
from torch import nn
import torch.backends.cudnn
import torchvision.transforms.functional as TF
import numpy as np

import random
random.seed(hyparams.seed)      # seed for random 

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

device = torch.device("cuda" if use_cuda else "cpu")


class DataSet(object):
    def __init__(self, split):
        self.datas = get_lists(split)[0]
        self.labels = get_lists(split)[1]

    def __len__(self):
        return len(self.datas)    

    def __getitem__(self, idx):
        while 1:
            label = np.zeros(5)
            idx = random.randint(0, len(self.datas) - 1)

            img = Image.open(self.datas[idx])
            #img = np.array(img)
            img = TF.to_tensor(img)

            raw_label = int( self.labels[idx])
            label[raw_label] = 1.0

            return img, label
        


# loss function cross entropy loss
loss_func = nn.CrossEntropyLoss()

def train(device, model, train_data_loader, val_data_loader, optimizer, total_epoch=hyparams.total_epoch):

    current_epoch = 0
    train_loss_list= []
    eval_loss_list= []
    print('Total Epoch: {}'.format(total_epoch))
    print('Total Eval Steps: {}'.format(len(val_data_loader)))

    while current_epoch < total_epoch:
        model.train()    
        current_epoch_loss = 0

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (img, label) in prog_bar:
            
            optimizer.zero_grad()

            # Move data to CUDA device
            img = img.to(device)
            label = label.to(device)

            pred = model(img)

            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()

            current_epoch_loss += loss.item()

            prog_bar.set_description('Train: Epoch: {}, Step: {}, Avg Loss: {:.6f}'.format(current_epoch, step, current_epoch_loss/( step+ 1)))

        current_epoch += 1
        avg_loss = current_epoch_loss/( step+ 1)
        train_loss_list.append(avg_loss)

        with torch.no_grad():
            eval_loss = eval(device, model, val_data_loader)
        eval_loss_list.append(eval_loss)

    return train_loss_list, current_epoch, eval_loss_list


def eval(device, model, val_data_loader):

    while 1:
        current_loss= 0
        count = 0
        model.eval()

        prog_bar = tqdm(enumerate(val_data_loader))

        for step, (img, label) in prog_bar:
            count += len(img)
            # Move data to CUDA device
            img = img.to(device)
            label = label.to(device)

            pred = model(img)

            loss = loss_func(pred, label)

            current_loss += loss.item()

            prog_bar.set_description('Eval: Step: {}, Avg Loss: {:.6f}'.format(step, current_loss/( step+ 1)))

        avg_loss = current_loss/( count)
        
        return avg_loss
    
