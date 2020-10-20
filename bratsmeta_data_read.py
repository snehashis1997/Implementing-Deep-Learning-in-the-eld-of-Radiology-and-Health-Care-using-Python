

name=r'C:\Users\NRAD.NRAD-28\Desktop\brats.mha'

from medpy.io.load import load
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io

image= io.imread(name, plugin='simpleitk')

img = sitk.ReadImage(name)

import SimpleITK as sitk

mhd = sitk.ReadImage(name)
origin = mhd.GetOrigin()
spacing = mhd.GetSpacing()
direction = mhd.GetDirection()

offset=mhd.GetOffset()

keys=mhd.GetMetaDataKeys()

name=r'C:\Users\NRAD.NRAD-28\Desktop\brats.mha'

import medpy

output=medpy.io.load(name)

image=output[0]

#header=output[1]

header = medpy.io.header.Header(name)

data=header.offset

