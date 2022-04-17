#在一个文件夹中的所有的天内，比较这一系列时段的图的指定类像元的同一波段在不同时刻的均值方差
# 20220105 本代码为在clipednewnewdata里头做的代码，裁剪之后的加里曼丹东南角
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import gdal
import glob
def output_one_line(fns,band,kind):
    '''
    输入一组图的tif文件名，以及需要保存的txt文件路径名
    输出一组图的单个波段各个时段的均值方差和最大最小值
    输入label一个判断标准，只包括0和1，图幅和输入的图像大小相同
    band是你想分析的波段
    Kind是你赋予的这个标签的类型，比如烟、无烟、海洋、陆地，英文字符串格式
    返回两个数组，均值和方差数组
    '''
    land_sea_label = gdal.Open(r'E:\SmokeDetection\source\semi-supervised learning\初探的实验结果\cliped1.tif').ReadAsArray()#海洋陆地标签
    # 在这里进行一个操作，把fns中的所有日期的对应时刻的文件合并，举例：UTC=00:00的所有图片和为一张
    imgs=[]
    obj_origin = glob.glob(r'{}\0830\*.tif'.format(fns)) #先找一个日子，把这个日子的所有文件当做基准
    for obj in obj_origin: #时序索引从0000到0800
        utc = obj[-8:-4]
        same_time_files = glob.glob(r'{}\*\{}.tif'.format(fns,utc)) #相同时间的文件索引
        img = np.array([])
        ary = gdal.Open(same_time_files[0]).ReadAsArray()
        ary[-2,:,:] = land_sea_label # 改写风向数据为海陆标签，然后存在第18个波段通道中
        for file in same_time_files[1:]:
            t = gdal.Open(file).ReadAsArray()
            t[-2,:,:] = land_sea_label
            ary = np.c_[ary,t]
        imgs.append(ary)


    stds = []
    means = []
    b = band - 1 #机器语言，0为初始
    for temp in imgs:
        if kind == 'smoke':label = temp[-1,:,:]
        if kind == 'cloud':
            label = np.where(temp[0,:,:]>0.2,1,0)
            label = np.where(temp[-1,:,:]==1,0,label)
        if kind == 'land':
            label = np.where(temp[0, :, :] > 0.2, 0, 1) #先把没云的区域设为1，作为label
            land = temp[-2,:,:]
            label = np.where(land == 1, label, 0) #再把有地的地方设为1，其它为0
        if kind == 'sea':
            label = np.where(temp[0, :, :] > 0.2, 0, 1) #先把没云的区域设为1，作为label
            land = temp[-2,:,:]
            label = np.where(land == 0, label, 0)  # 再把有海的地方设为1，其它为0
        temp = np.where(label == 1, temp, 0)
        mean = temp[b, :, :].sum() / label.sum()
        temp_std = np.where(label == 1, (temp[b, :, :] - mean) ** 2, 0)
        std = np.sqrt(temp_std.sum() / (label.sum() - 1))
        stds.append(std)
        means.append(mean)
    mk_one_line(means,stds,kind)

    return means,stds

def mk_one_line(means,stds,kind):
    '''
    在pyplot上画一条带误差棒的线
    '''

    Times = np.arange(len(means))
    # print(len(means))
    # print(len(stds))
    error_limit = [stds, stds]
    plt.errorbar(Times,means,error_limit,fmt=":o",elinewidth=1,ms=5,capsize=3, capthick=1,label = kind)

# fns的参考格式是cliped_new_new_data\0817\0000.tif
fns = r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data'
#连续波段出图
for band in range(1,17):
    times = []
    for i in range(7):
        times.append('0{}:00'.format(i))
    output_one_line(fns,band,'smoke')
    output_one_line(fns,band,'cloud')
    output_one_line(fns,band,'sea')
    output_one_line(fns,band,'land')
    plt.xlabel('UTC')
    if band>=7:
        plt.ylabel('Brightness')
    else:
        plt.ylabel('Albedo')
    plt.xticks(np.arange(0,len(times)*6,6),times)
    plt.title('Band {}'.format(band))
    plt.legend()  # bbox_to_anchor=(0.1,0.5)
    plt.savefig(r'E:\Radiative Effect\picture_all_5days\band {}.jpg'.format(band))
    plt.close()
    print('Successfully output')
    #plt.show()