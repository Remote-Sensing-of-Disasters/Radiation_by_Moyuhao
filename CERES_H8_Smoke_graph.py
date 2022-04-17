#做一张巨他妈难的图片，要求根据CERES的时间序列，分析葵花8分辨率的类别分布，得到一个分布柱状图和折线图
import gdal
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from tqdm import tqdm
import copy
class graph():
    """
    现在类里头定义好CERES的类型（graph_type)以及存储路径(result_pth)
    """
    def __init__(self, graph_type, result_pth):
        self.gt = graph_type
        self.pth = result_pth
        self.H8shape = [] #葵花图幅大小
        self.CERESshape=[] #CERES图幅大小
        self.H8fns=[]
        self.CERESfns=[]
        self.Smokefns=[]
    def load_H8(self,special_day = None):
        """
        加载'E:\SmokeDetection\source\semi-supervised learning\new_new_data'中所有的葵花数据，除了0808，special_day可以只读取那一天的数据
        造了一个数组，里头存放了所有的数据，根据日期时间从小到大，从旧到新排列
        """
        if special_day and len(glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\{}\*.tif'.format(special_day)))==0:
            raise ValueError('看看你输了什么好日子，格式如下：0831,字符串哦')
        try:
            self.H8fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\*\*.tif')
            if special_day:
                self.H8fns = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\{}\*.tif'.format(special_day))
            H8fns = copy.deepcopy(self.H8fns) # 这个需要深度复制一个才能用哦，不然就删除(remove)不了数据了呢
            '''
            s =['a','b','v','d','e','f']
            for ss in s:
              s.remove(ss)
            s会剩下一半的值，如果有三个就剩一个，必须用for遍历一个被深度复制的数组而不是原来的s
            '''
            for fn in H8fns: # 检测三个数据源是否同源
                date = fn.split('\\')[-2]
                time = fn.split('\\')[-1][:-4]
                if not (os.path.exists(r'E:\Radiative Effect\{}\{}\{}.tif'.format(self.gt,date,time)) and os.path.exists(r'E:\SmokeDetection\source\semi-supervised learning\pixel classification results\Noon256batchsize3e4lrBN050Dropout\{}\{}.tif'.format(date,time))):
                    self.H8fns.remove(fn)
            H8_datasets = []
            for fn in tqdm(self.H8fns):
                ary = gdal.Open(fn).ReadAsArray()
                H8_datasets.append(ary)
            self.H8shape = H8_datasets[2].shape
            return H8_datasets
        except:
            ImportError('LoadH8导入出错了,错在加载数据上面，你干了什么呀？原代码是没错哒！')


    def load_CERES(self):
        """
        加载‘E:\Radiative Effect’中特定的CERES影像，目前有5个，选一个类，全都加载完哦
        """
        self.CERESfns = glob.glob(r'E:\Radiative Effect\{}\*\*.tif'.format(self.gt))
        CERESfns = copy.deepcopy(self.CERESfns)
        for fn in CERESfns:  # 检测三个数据源是否同源
            date = fn.split('\\')[-2]
            time = fn.split('\\')[-1][:-4]
            if not (os.path.exists(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\{}\{}.tif'.format(date,time)) and os.path.exists(r'E:\SmokeDetection\source\semi-supervised learning\pixel classification results\Noon256batchsize3e4lrBN050Dropout\{}\{}.tif'.format(date, time))):
                self.CERESfns.remove(fn)

        if len(self.CERESfns)==0:
            raise ValueError('这个CERES数据没有，是名字写错了还是数据没了呢？')
        try:

            CERES_datasets = []
            for fn in tqdm(self.CERESfns):
                CERES_datasets.append(gdal.Open(fn).ReadAsArray())
            self.CERESshape = CERES_datasets[2].shape
        except:
            raise ImportError('load_CERES导入gdal出错了,盲猜gdal的原因，我用的是GDAL3.4.1')
        return CERES_datasets

    def load_Smoke(self):
        """
        加载烟检测数据，在'E:\\SmokeDetection\\source\\semi-supervised learning\\pixel classification results\\Noon256batchsize3e4lrBN050Dropout'哦哦
        """
        self.Smokefns=glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\pixel classification results\Noon256batchsize3e4lrBN050Dropout\*\*.tif')
        if len(self.Smokefns)==0:
            raise ValueError('烟检测的结果（01值图）没加载，看看是不是路径写错了！')
        try:
            Smokefns = copy.deepcopy(self.Smokefns)
            for fn in Smokefns: # 检测三个数据源是否同源
                date = fn.split('\\')[-2]
                time = fn.split('\\')[-1][:-4]
                if not (os.path.exists(r'E:\Radiative Effect\{}\{}\{}.tif'.format(self.gt,date,time)) and os.path.exists(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\{}\{}.tif'.format(date,time))):
                    self.Smokefns.remove(fn)
            Smoke_datasets = []
            for fn in tqdm(self.Smokefns):
                Smoke_datasets.append(gdal.Open(fn).ReadAsArray())
        except:
            raise ImportError('load_Smoke时候导入文件出错了，盲猜gdal的原因，我用的是GDAL3.4.1')
        return Smoke_datasets

    def load_classification(self, himawari_datasets, smoke_datasets):
        """
        叠加烟检测数据和葵花数据的云结果，得到一个相同分辨率的像元类别结果，海为0，地为1,烟为3，云为4。烟的优先级在云上，云不盖烟。
        """
        classification = []
        for i,temp in enumerate(himawari_datasets):
            if kind == 'smoke': label = temp[-1] / 255  # 这里的label得改掉，不能是最后一个通道了
            if kind == 'cloud':
                label = np.where(temp[0, :, :] > 0.2, 1, 0)
                label = np.where(temp[-1, :, :] == 1, 0, label)


    def sample(self):
        """
        这是最恶心的代码，要定位到具体的行列号，然后采样对应的CERES数据和Classification数据
        """
        pass

    def trace_single(self):
        """
        找一天来分析出图
        """
        pass

    def trace_whole(self):
        """
        把这28天都找了出图
        """
        pass

if __name__ == '__main__':
    d = graph('adj_sfc_uva_all_1h','12')
    d.load_CERES()
    d.load_H8()
    d.load_Smoke()
    print(d.Smokefns)

    print(d.H8fns)
    print(d.CERESfns)
