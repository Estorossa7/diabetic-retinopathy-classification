from datasetup import get_lists
from PIL import Image 

import numpy as np 
import matplotlib.pyplot as plt 

filelist, labellist = get_lists('train')

img = Image.open(filelist[0])
print(img)
img = np.array(img)


label = int(labellist[0])

print(label)