from hyparams import hyparams
from models import CNN
from train import DataSet, train, test

from models_copy import CNN as CNN_copy
from train_copy import train as train_copy
from train_copy import test as test_copy

import torch
from torch.utils import data as data_utils
from torch import optim
import numpy as np

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

# models
def training_setup():
    # data loader setup
    train_dataset = DataSet('train')
    val_dataset = DataSet('val')
    test_dataset = DataSet('test')

    train_data_loader = data_utils.DataLoader( train_dataset, batch_size=hyparams.batch_size, shuffle=True)
    val_data_loader = data_utils.DataLoader( val_dataset, batch_size=hyparams.eval_batch_size)
    test_data_loader = data_utils.DataLoader( test_dataset, batch_size=hyparams.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ",device)

    # Model
    model = CNN().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr, weight_decay=hyparams.weight_decay)
    #optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr)

    train_loss_list, current_epoch, eval_loss_list = train(device, model, train_data_loader, val_data_loader, optimizer, total_epoch=hyparams.total_epoch)

    pred_list, label_list = test(device, model, test_data_loader)

    print("The End - train")
    return train_loss_list, current_epoch, eval_loss_list, pred_list, label_list

#   models_copy
def training_setup_copy():
    # data loader setup
    train_dataset = DataSet('train')
    val_dataset = DataSet('val')
    test_dataset = DataSet('test')

    train_data_loader = data_utils.DataLoader( train_dataset, batch_size=hyparams.batch_size, shuffle=True)
    val_data_loader = data_utils.DataLoader( val_dataset, batch_size=hyparams.eval_batch_size)
    test_data_loader = data_utils.DataLoader( test_dataset, batch_size=hyparams.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ",device)

    # Model
    model = CNN_copy().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr, weight_decay=hyparams.weight_decay)
    #optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=hyparams.lr)

    train_loss_list_copy, current_epoch_copy, eval_loss_list_copy = train_copy(device, model, train_data_loader, val_data_loader, optimizer, total_epoch=hyparams.total_epoch)

    pred_list_copy, label_list_copy = test_copy(device, model, test_data_loader)

    print("The End - train_copy")
    return train_loss_list_copy, current_epoch_copy, eval_loss_list_copy, pred_list_copy, label_list_copy

"""
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

def plots(t_list, v_list, epoch):
    x = [i for i in range(1,epoch)]

    #plt.plot( t_list, label='training loss')
    #plt.plot( v_list, label='validation loss')
    #plt.xlabel('epochs')
    #plt.ylabel('loss')
    #plt.legend()
    #plt.show()
    return x, t_list

def accuracy(pred, label):

    total = 0
    correct = 0

    for i in range(len(pred)):
        if pred[i] == label[i]:
            correct += 1
            total += 1
        else:
            total +=1

    print("total:", total)
    print("correct: ", correct)

    return correct/total

def main():

    #this is for models
    train_loss_list, current_epoch, eval_loss_list, prediction_list, test_label_list = training_setup()

    #train_loss_list = [0.0788881455243814, 0.06139617374554129, 0.0533022528495824, 0.039417338760403715, 0.027739145308917405]
    #eval_loss_list = [0.08185563263084207, 0.07871838592525039, 0.0788090346114976, 0.07451074530503579, 0.07827994732984475]
    #current_epoch = 5

    #this is for models
    train_loss_list_copy, current_epoch_copy, eval_loss_list_copy, prediction_list_copy, test_label_list_copy = training_setup_copy()

    #train_loss_list_copy = [0.07790916402003553, 0.06693182645900415, 0.06419145356458013, 0.057267129956053035, 0.056410486431847706]
    #eval_loss_list_copy = [0.07942532801202365, 0.07482676633766719, 0.08818594925105572, 0.07516866762723241, 0.08216743916273117]
    #current_epoch_copy = 5

    #   accuracy
    print("Accuracy models: ", accuracy(prediction_list, test_label_list))
    print("Accuracy models_copy: ", accuracy(prediction_list_copy, test_label_list_copy))



    plot1 = plt.subplot2grid((2,2), (0, 0)) 
    plot2 = plt.subplot2grid((2,2), (0, 1)) 

    plot1.plot(train_loss_list, [i for i in range(1,current_epoch + 1)]) 
    plot1.set_title("training and eval loss")
    
    plot2.plot(train_loss_list_copy, [i for i in range(1,current_epoch_copy + 1)])
    plot2.set_title("training and eval loss_copy")

    plt.tight_layout() 
    plt.show()

    print("The End - all_for_one")


if __name__ == "__main__":
    main()
