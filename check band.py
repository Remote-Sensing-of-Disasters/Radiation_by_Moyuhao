# 检查每个波段是不是有问题
import gdal
import os
import numpy as np
import glob

fns = glob.glob(r'F:\new_new_data\*\0440.tif')
for fn in fns:
    ary = gdal.Open(fn).ReadAsArray() #[band,height,width]
    ary_band = ary[10,:,:]
    print(fn)
    print(ary_band.max())
    print(ary_band.min())
    print(ary_band.std())


