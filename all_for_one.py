from train import DataSet, train
from hyparams import hyparams
from models import CNN

import torch
from torch.utils import data as data_utils
from torch import optim
import numpy as np

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

def training_setup():
    # data loader setup
    train_dataset = DataSet('train')
    test_dataset = DataSet('test')

    train_data_loader = data_utils.DataLoader( train_dataset, batch_size=hyparams.batch_size, shuffle=True)
    test_data_loader = data_utils.DataLoader( test_dataset, batch_size=hyparams.batch_size)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ",device)

    # Model
    model = CNN().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr)

    loss_list = train(device, model, train_data_loader, test_data_loader, optimizer, total_epoch=hyparams.total_epoch)

    print("The End - train")
    return loss_list


"""
    need to add eval() to train.py
    need to create test.py similat to train,py
    create testing_setup() in all_for_one.py

"""

def plot(loss_l):
    epochs = [i for i in range(len(loss_l))]

    plt.subplot(2,2,1)
    plt.plot(epochs, loss_l, 'r', label= 'Training loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def main():
    loss_list = training_setup()
    losss = loss_list
#    losss = [1.1991665897698238, 1.154235893282397, 1.1485114714194988, 1.1389241177460243, 1.107125158967643, 1.0911626774689247, 1.0870850743918583, 1.073838706674247, 1.0620280709759942, 1.0425515174865723, 1.0226404132514164, 1.0084421367480838, 1.0027588893627297, 0.9970466515113567, 0.9776576716324379, 0.9760608282582514, 0.9694907336399473, 0.9624767529553381, 0.9623742432429873, 0.9620455071843904]
    plot(losss)

    print("The End - all_for_one")


if __name__ == "__main__":
    main()
