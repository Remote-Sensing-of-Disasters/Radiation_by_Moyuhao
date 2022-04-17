# 这是一个把下下来的CERES文件nc变成规定经纬度之内的图。
# 输入两个经纬表和一个CERES的nc文件，输出一张经过插值的图片
import netCDF4 as nc
import glob
import gdal
import numpy as np
import os

def output_one_type(CERES_np, pic, longitude, latitude, result_pth):
    '''
    输入nc文件中的一个类型的numpy数据（一个时段），以及一张裁剪的葵花图（gdal格式，它的长宽是你要的最终结果），还有nc文件的经纬表。
    输出按照图片中那样的长宽、空间分辨率和对应经纬的图片
    '''
    def interpret(lonlat):
        '''
        对特定的经纬点进行插值，输入特定经纬点，以数组形式(lonlat=[lon,lat])，其它的都调用上层output_one_type的形式参数
        '''
        def findpix_lonlat(lonlat):
            '''
            找到这个经纬点在Ceres表上对应的左上\左下\右上\右下经纬度，以4个数组的形式输出
            '''
            for lon in longitude:
                if np.abs(lon - lonlat[0])<1: #从小往大数，第一个绝对值差小于1的肯定是左边
                    right = lon + 1
                    left = lon
                    break
            for lat in latitude:
                if np.abs(lat-lonlat[1])<1:
                    # if lat == -4.5:
                    #     down = -4.5
                    #     up = -4.5
                    #     break
                    down = lat
                    up = lat + 1
                    break
            idx = []
            idx.append([left,up])
            idx.append([left,down])
            idx.append([right,up])
            idx.append([right,down])
            return idx

        def findvalue(fourpoints):
            '''
            找到这4个经纬点在Ceres上对应的值:左上\左下\右上\右下
            '''
            points_value = []
            for [lon,lat] in fourpoints:
                points_value.append(CERES_np[np.where(lat==latitude),np.where(lon==longitude)][0,0])
            return points_value

        def neighbour_interpretation(lonlat, four_points,four_values):
            '''
            输入待插值的经纬点(lonlat)、以及找到的4个原图的经纬点(four_points)，根据这些信息在4个经纬点的值(four_values)中选个最近的插值
            输出一个值
            '''
            longitude_pix = lonlat[0]
            latitude_pix = lonlat[1]
            ul,dl,ur,dr = four_points
            if np.abs(longitude_pix-ul[0]) > np.abs(longitude_pix-ur[0]) and np.abs(latitude_pix-ul[1]) > np.abs(latitude_pix-dl[1]):
                return four_values[3] #右下
            if np.abs(longitude_pix - ul[0]) > np.abs(longitude_pix - ur[0]) and np.abs(latitude_pix - ul[1]) < np.abs(
                latitude_pix - dl[1]):
                return four_values[2] #右上
            if np.abs(longitude_pix - ul[0]) < np.abs(longitude_pix - ur[0]) and np.abs(latitude_pix - ul[1]) < np.abs(
                latitude_pix - dl[1]):
                return four_values[0] #左上
            if np.abs(longitude_pix - ul[0]) < np.abs(longitude_pix - ur[0]) and np.abs(latitude_pix - ul[1]) > np.abs(
                latitude_pix - dl[1]):
                return four_values[1] #左下

        four_points = findpix_lonlat(lonlat) #lonlat_ul,lonlat_dl,lonlat_ur, lonlat_dr 四个经纬点
        # 此处的接口有问题，我如果要考虑到日后的双线性插值，就得改变接口，这里的最近邻插值其实只需要一个像元就行。
        # 但是如果只输入一个像元，那我这就不需要单独建立一个函数了。
        # 所以我决定更改一下，左上右下的经纬度坐标得输入，然后那4个值也得输入，我还得做个函数用于查找
        four_values = findvalue(four_points) # 四个经纬点对应在CERES数据上的值
        value = neighbour_interpretation(lonlat, four_points,four_values) # 想插值的点的经纬度、对应CERES四个点的经纬度、对应CERES四个点的值

        return value

    geotfm = pic.GetGeoTransform()
    ary = np.zeros([pic.ReadAsArray().shape[0],pic.ReadAsArray().shape[1]]) # 结果图的numpy格式,shape0纬度，图像的height(501);shape1经度，width(582)
    count = 0
    for i in range(ary.shape[0]): # 纬度
        for j in range(ary.shape[1]): # 经度
            [lon_pix,lat_pix] = [geotfm[0]+geotfm[1]*j, geotfm[3]+geotfm[-1]*i]  #[经度，纬度]
            ary[i,j] = interpret([lon_pix, lat_pix])
            count+=1
            print(count)
    # 制作图片
    driver = gdal.GetDriverByName('GTiff')
    result_pic = driver.Create(result_pth, ary.shape[1], ary.shape[0],1,gdal.GDT_Float32)  # 结果图
    result_pic.GetRasterBand(1).WriteArray(ary)
    result_pic.SetGeoTransform(geotfm)
    result_pic.SetProjection(pic.GetProjection())
    del result_pic


if __name__ == '__main__':
    #(CERES_np, pic, longitude, latitude, result_pth)
    import time
    print(time.asctime())
    data = nc.Dataset(r'E:\Radiative Effect\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20150801-20151031.nc')
    nc_name = 'ini_sfc_lw_up_clr_1h'
    CERES_np = np.array(data.variables[nc_name])[0]
    pic = gdal.Open(r'E:\SmokeDetection\source\semi-supervised learning\pixel classification results\Noon256batchsize3e4lrBN050Dropout\0826\0300.tif')
    longitude = data.variables['lon']
    latitude = data.variables['lat']
    result_pth = r'E:\Radiative Effect\result\test.tif'
    high_res_pic = output_one_type(CERES_np, pic, longitude, latitude, result_pth)
    print(time.asctime())


