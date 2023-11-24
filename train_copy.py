from datasetup import get_lists, get_class_weight
from hyparams import hyparams
import torch.optim as optim

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

    def get_label(split):
        return get_lists(split)[1]

    def __len__(self):
        return len(self.datas)    

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.datas) - 1)

            img = Image.open(self.datas[idx])
            img = TF.to_tensor(img)

            #   one hot encoding of labels
            label = np.zeros(5)
            raw_label = self.labels[idx]
            label[raw_label] = 1.0
            label = np.float32(label)

            #   1d label array
            #label = self.labels[idx]

            return img, label
        


#   loss function cross entropy loss
#class_weight_train = get_class_weight('train')
#class_weight_val = get_class_weight('val')

#loss_func_train = nn.CrossEntropyLoss(weight= class_weight_train)
#loss_func_val = nn.CrossEntropyLoss(weight= class_weight_val)

#   loss function mean square loss
loss_func_train = nn.MSELoss()
loss_func_val = nn.MSELoss()


def train(device, model, train_data_loader, val_data_loader, optimizer, total_epoch=hyparams.total_epoch):

    current_epoch = 0
    train_loss_list= []
    eval_loss_list= []
    count = 0
    print('Total Epoch: {}'.format(total_epoch))
    print('Total Eval Steps: {}'.format(len(val_data_loader)))

    while current_epoch < total_epoch:
        
        model.train()

        current_epoch_loss = 0

        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (img, label) in prog_bar:
            
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                # Move data to CUDA device
                img = img.to(device)
                label = label.to(device)

                pred = model(img)

                loss = loss_func_train(pred, label)
                loss.backward()
                optimizer.step()

            current_epoch_loss += loss.item() 

            prog_bar.set_description('Train: Epoch: {}, Step: {}, Avg Loss: {:.6f}'.format(current_epoch, step, current_epoch_loss/( step+ 1)))

        avg_loss = current_epoch_loss/( step+ 1)
        train_loss_list.append(avg_loss)

        eval_loss = eval(device, model, val_data_loader)
        eval_loss_list.append(eval_loss)

        #control_idx = 0
        ##   lr update when loss stabilizers
        #if train_loss_list[control_idx] < train_loss_list[current_epoch]:
        #    count +=1
        #else:
        #    control_idx = current_epoch
        #    count = 0
        #
        #if current_epoch == 10:
        #    new_lr = 1e-4
        #    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr= new_lr, weight_decay=hyparams.weight_decay)
        #    control_idx = current_epoch
        #    print("lr changed 10")
        #
        #if current_epoch == 50:
        #    new_lr = 1e-5
        #    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr= new_lr, weight_decay=hyparams.weight_decay)
        #    control_idx = current_epoch
        #    print("lr changed 50")

        current_epoch += 1

    return train_loss_list, current_epoch, eval_loss_list


def eval(device, model, val_data_loader):

    while 1:
        current_loss= 0
        model.eval()

        prog_bar = tqdm(enumerate(val_data_loader))

        for step, (img, label) in prog_bar:

            with torch.no_grad():
                # Move data to CUDA device
                img = img.to(device)
                label = label.to(device)

                pred = model(img)

                loss = loss_func_val(pred, label)

            current_loss += loss.item() 

            prog_bar.set_description('Eval: Step: {}, Avg Loss: {:.6f}'.format(step, current_loss/( step+ 1)))

        avg_loss = current_loss/( step+ 1)
        
        return avg_loss
    
def test(device, model, test_data_loader):

    while 1:
        model.eval()
        pred_list = []
        label_list = []

        prog_bar = tqdm(enumerate(test_data_loader))

        for step, (img, label) in prog_bar:

            with torch.no_grad():
                # Move data to CUDA device
                img = img.to(device)

                pred = model(img)

                predictions = pred.argmax(dim=1).cpu().detach().tolist()

                pred_list.extend(predictions)

                label_list = label.argmax(dim=1).cpu().detach().tolist()

            prog_bar.set_description('Test: Step: {}'.format(step))
        
        return np.array(pred_list), np.array(label_list)