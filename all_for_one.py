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
    val_dataset = DataSet('val')

    train_data_loader = data_utils.DataLoader( train_dataset, batch_size=hyparams.batch_size, shuffle=True)
    val_data_loader = data_utils.DataLoader( val_dataset, batch_size=hyparams.eval_batch_size)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ",device)

    # Model
    model = CNN().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr)

    train_loss_list, current_epoch, eval_loss_list = train(device, model, train_data_loader, val_data_loader, optimizer, total_epoch=hyparams.total_epoch)

    print("The End - train")
    return train_loss_list, current_epoch, eval_loss_list


"""
    need to add eval() to train.py
    need to create test.py similat to train,py
    create testing_setup() in all_for_one.py

    can add accuracy:
        acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

"""

def plot(t_list, v_list, epoch, plt_name):
    x = [i for i in range(epoch)]

    plt.subplot(1,1,1)
    plt.plot(x, t_list, 'r')
    plt.plot(x, v_list, 'b')
    plt.title(plt_name)
    plt.xlabel('epochs')
    plt.ylabel(plt_name.split(' ')[-1])
    plt.legend()
    plt.show()

def main():
    train_loss_list, current_epoch, eval_loss_list = training_setup()
    print(train_loss_list)
    print(eval_loss_list)
    
    plot(train_loss_list, eval_loss_list, current_epoch, 'training and eval loss')

    print("The End - all_for_one")


if __name__ == "__main__":
    main()
