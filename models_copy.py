import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
#       layer 1: input channel = 1, output channel = 10, kernel size = 5x5, maxpool size = 2x2
        self.layer1 = nn.Sequential(nn.Conv2d(3,32,kernel_size=3), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.BatchNorm2d(32))
        
#       layer 2: input channel = ouput channel of layer 1, output channel = 20, kernel size = 5x5, maxpool size = 2x2
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.BatchNorm2d(64))

#       layer 2: input channel = ouput channel of layer 1, output channel = 20, kernel size = 5x5, maxpool size = 2x2
        self.layer3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.BatchNorm2d(128))
        
#       A drop layer deletes 20% of the features to help prevent overfitting 
        self.drop = nn.Dropout2d(p=0.2)
        
#       layer 3: fully connected layer, input features = 320, output features = 50        
        self.fully_cl_1 = nn.Linear(24*24*128, 1000)
        
#       layer 3: fully connected layer, input features = 320, output features = 50        
        self.fully_cl_2 = nn.Linear(1000, 100)

#       layer 4: fully connected layer, input features = output feature of layer 3, output feature = num_of_classes        
        self.fully_cl_3 = nn.Linear(100, 5)
        
#   to send data forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.dropout(self.drop(x), training=self.training)
        x = x.view(-1, 24*24*128)
        x = self.fully_cl_1(x)
        x = F.relu(x)
        x = self.fully_cl_2(x)
        x = F.relu(x)
        x = self.fully_cl_3(x)
        output = F.softmax(x, dim= 1)
        return output 
