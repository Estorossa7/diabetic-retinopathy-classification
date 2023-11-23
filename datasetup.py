import os
import torch
import numpy as np
from sklearn.utils import class_weight

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_dir = "data\\archive\\images_combines"
list_dir = "data\\archive"


train_size = int(0.8 * 3662)                    # 3662
test_size = int(0.6 * (3662 - train_size))      # 429
val_size = 3662 - test_size - train_size        # 294

# returns file list and label list 
def get_lists(split):
    labellist = []
    filelist = []

    with open( list_dir + '/{}list.csv'.format(split) ) as f:
        iterf = iter(f)
        next(iterf)
        for line in iterf:
            line = line.strip()
            line = line.split(',')
            file = os.path.join(data_dir, line[0])
            filelist.append(file + ".png")
            labellist.append(int(line[1]))

    return filelist, labellist

def get_class_weight(split):

    labels= get_lists(split)[1]
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.array([0,1,2,3,4]),y= labels )
    class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
    return class_weights

print("The End - datasetup")