import sys, os

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
            labellist.append(line[1])

    return filelist, labellist

print("The End - datasetup")