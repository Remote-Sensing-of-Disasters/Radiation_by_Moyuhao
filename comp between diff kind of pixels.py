#比较这一系列时段的图的指定类像元的同一波段在不同时刻的均值方差
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
    stds = []
    means = []
    b = band - 1 #机器语言，0为初始
    for fn in fns:
        temp = gdal.Open(fn).ReadAsArray()
        if kind == 'smoke':label = temp[-1,:,:]
        if kind == 'cloud':
            label = np.where(temp[0,:,:]>0.2,1,0)
            label = np.where(temp[-1,:,:]==1,0,label)
        if kind == 'land':
            label = np.where(temp[0, :, :] > 0.2, 0, 1) #先把没云的区域设为1，作为label
            land = gdal.Open(r'E:\SmokeDetection\source\semi-supervised learning\初探的实验结果\cliped1.tif').ReadAsArray()
            label = np.where(land == 1, label, 0) #再把有地的地方设为1，其它为0
        if kind == 'sea':
            label = np.where(temp[0, :, :] > 0.2, 0, 1) #先把没云的区域设为1，作为label
            land = gdal.Open(r'E:\SmokeDetection\source\semi-supervised learning\初探的实验结果\cliped1.tif').ReadAsArray()
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


fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\cliped_new_new_data\0823\*.tif')
#连续波段出图
for band in range(1,17):
    output_one_line(fns,band,'smoke')
    output_one_line(fns,band,'cloud')
    output_one_line(fns,band,'sea')
    output_one_line(fns,band,'land')
    plt.xlabel('UTC')
    if band>=7:
        plt.ylabel('Brightness')
    else:
        plt.ylabel('Albedo')
    times = []
    for i in range(7):
        times.append('0{}:00'.format(i))
    plt.xticks(np.arange(0,len(times)*6,6),times)
    plt.title('Band {}'.format(band))
    plt.legend()  # bbox_to_anchor=(0.1,0.5)
    plt.savefig(r'E:\Radiative Effect\picture_0823\band {}.jpg'.format(band))
    plt.close()
    print('Successfully output')
    #plt.show()