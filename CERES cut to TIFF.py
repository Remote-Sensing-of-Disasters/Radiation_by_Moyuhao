# 把CERES图片先插成和葵花一样的分辨率，然后再裁剪它来适合葵花
import numpy as np
import torch
import netCDF4 as nc
import torchvision.transforms as tfm
import gdal
import os
import glob
import time as t
from tqdm import tqdm
data = nc.Dataset(r'E:\Radiative Effect\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20150801-20151031.nc')
fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\pixel classification results\Noon256batchsize3e4lrBN050Dropout\*\*.tif')
CERES_fns = ['adj_sfc_uva_all_1h']#[r'ini_sfc_lw_down_all_1h',r'ini_sfc_sw_down_all_1h',r'ini_toa_lw_all_1h',r'ini_toa_sw_all_1h']
for CERES_name in CERES_fns:
    #CERES_name = 'ini_sfc_sw_down_all_1h' #ini_sfc_lw_down_all_1h ini_sfc_sw_down_all_1h
    if not os.path.exists(r'E:\Radiative Effect\{}'.format(CERES_name)):
        os.mkdir(r'E:\Radiative Effect\{}'.format(CERES_name))
    CERES_data = np.array(data.variables[CERES_name])
    bar = tqdm(fns)
    for fn in bar:
        date = fn.split('\\')[-2]
        time = fn.split('\\')[-1][:-4]
        if not os.path.exists(r'E:\Radiative Effect\{}\{}'.format(CERES_name,date)):
            os.mkdir(r'E:\Radiative Effect\{}\{}'.format(CERES_name,date))
        result_pth = r'E:\Radiative Effect\{}\{}\{}.tif'.format(CERES_name,date,time)
        month,day = int(date[:2]),int(date[2:])
        if int(time[2])>3: #时刻,我选择让整点时刻上下半小时为此值05:40\05:50跟着06:00
            Time = int(time[1])+1
        else:
            Time = int(time[1])
        num = (month-8)*31*24+(day-1)*24+Time #仅限8、9月份，如果10月份还得改代码
        CERES_np = CERES_data[num] # 贼难算的时刻，月份、日期、时间一起决定了到底是多少
        CERES_torch = torch.from_numpy(CERES_np).unsqueeze(0)
        # print(CERES_torch.size())
        CERES_torch_rs = tfm.Resize([15*50,20*50],0)(CERES_torch)
        CERES_np_interp = CERES_torch_rs.squeeze().numpy()[::-1] #纬度是从南到北，得反过来看
        pic = gdal.Open(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\0808\0400.tif')
        img = pic.ReadAsArray()
        geotfm = pic.GetGeoTransform()
        col = int((geotfm[0]-np.array(data.variables['lon'][0]))/geotfm[1])
        rol = int((geotfm[3]-np.array(data.variables['lat'][-1]))/geotfm[-1])
        ary = CERES_np_interp[rol:rol+img.shape[1],col:col+img.shape[2]]
        driver = gdal.GetDriverByName('GTiff')
        result_pic = driver.Create(result_pth, ary.shape[1], ary.shape[0],1,gdal.GDT_Float32)  # 结果图
        result_pic.GetRasterBand(1).WriteArray(ary)
        result_pic.SetGeoTransform(geotfm)
        result_pic.SetProjection(pic.GetProjection())





















# a = torch.Tensor([[[1,2],[3,4]]])
# print(a)
# b = tfm.Resize([100,100],0)
# print(b(a))