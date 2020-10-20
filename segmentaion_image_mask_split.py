
from matplotlib import pyplot as plt
import cv2
import numpy as np

"""1 for necrosis
2 for edema
3 for non-enhancing tumor
4 for enhancing tumor
0 for everything else"""

#255 is for white
    
name=r'D:\BRATS2013\Image_Data\HG\0001\VSD.Brain_3more.XX.XX.OT\VSD.Brain_3more.XX.XX.OT.6560.mha\0.png'

img=cv2.imread(name,0)

values = list(set(img.flatten()))

#plt.imshow(img)

#plt.show()


