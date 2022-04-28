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
    def __init__(self, graph_type):
        self.gt = graph_type
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
                elif not time[0]=='0' or time[1]=='9':
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
            elif not time[0]=='0' or time[1]=='9':
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
                elif not time[0] == '0' or time[1] == '9':

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
        try:
            land_sea_label = gdal.Open(r'E:\Radiative Effect\landsea501_582.tif').ReadAsArray()  # 海洋陆地标签
            classification = []
            for i,temp in enumerate(himawari_datasets):
                seaLand = np.copy(land_sea_label)
                seaLandCloud = np.where(temp[0, :, :] > 0.2, 4, seaLand) # 设云
                seaLandCloudSmoke = np.where(smoke_datasets[i]==255, 3, seaLandCloud) # 设烟
                classification.append(seaLandCloudSmoke)
            return classification
        except:
            raise ImportError('load_classification出错误了，很可能是文件加载的问题，看看land_sea_label的文件还在吗，还是[501,582]的shape吗')
    def sample(self,classification, CERES_dataset, row,col):
        """
        这是最恶心的代码，要定位到具体的行列号，然后采样对应的CERES数据和Classification数据,row和col是采样区左上角的点，50、50的像元跨度
        """
        #row,col =  300,358 # 这里写索引
        graph_element_CERES = []
        graph_element_classification=[]
        for i,c in enumerate(classification): #写到这里强迫症犯了，写不下去，为了让自己写下去就随便命名了
            test = CERES_dataset[i][row:row+50,col:col+50]
            if not round(test.sum()/test.mean()) == test.shape[0]*test.shape[1]:
                raise ValueError('{}数据错啦！切片里的值不是同一个值！检测一下行列号'.format(self.CERESfns[i]))
            graph_element_CERES.append(CERES_dataset[i][row:row+50,col:col+50])
            graph_element_classification.append(classification[i][row:row+50,col:col+50])
        return graph_element_classification, graph_element_CERES

    def trace_single(self):
        """
        找一天来分析出图
        """
        pass

    def trace_whole(self,graph_element_classification,graph_element_CERES):
        """
        把这28天都找了出图，输出的是按照0100-0800排列好的CERES平均数据，以及分类数据，分类数据从左到右海地烟云
        """
        obj_origin = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\0818\*.tif')[4:]  # 先找一个日子，把这个日子的所有文件当做基准，去掉从00:40开始

        ceres_totall = []
        class_totall = []
        for obj in obj_origin:
            utc = obj[-8:-4]
            same_time_files = glob.glob(r'E:\SmokeDetection\source\semi-supervised learning\new_new_data\*\{}.tif'.format(utc))
            idx= []
            for file in same_time_files:
                try:
                    i=self.H8fns.index(file)
                    idx.append(i)
                except:
                    '随便写一点啥'
            ceres_time = 0
            for j in idx:
                ceres_time+=graph_element_CERES[j][0,0]
            ceres_totall.append(ceres_time/len(idx))
            num_smoke, num_cloud, num_land, num_sea = 0,0,0,0 #烟云地海 四个类别每个时刻的数值 海为0，地为1,烟为3，云为4
            for k in idx:
                num_smoke += np.where(graph_element_classification[k]==3,1,0).sum()
                num_cloud += np.where(graph_element_classification[k]==4,1,0).sum()
                num_land+= np.where(graph_element_classification[k]==1,1,0).sum()
                num_sea+= np.where(graph_element_classification[k]==0,1,0).sum()
            if not num_sea+num_land+num_cloud+num_smoke==2500*len(idx):
                raise ValueError('哪里出错了呢？导致类别的和不全')
            class_totall.append([num_sea, num_land, num_smoke, num_cloud])
        return ceres_totall,class_totall
    def mk_a_graph(self,ceres_totall,class_totall,result_pth):
        x_axis = []
        class_totall = np.array(class_totall)
        ceres_totall = np.array(ceres_totall)
        class_hours = []
        ceres_hours = []
        for i in range(1,8):# 01：00-07：00，精准时刻是从00：40-07：30
            x_axis.append('{}:00'.format(i+8))
            #把每个时刻的整合在1小时里
            class_hours.append([class_totall[(i-1)*6:i*6,0].sum(),class_totall[(i-1)*6:i*6,1].sum(),class_totall[(i-1)*6:i*6,2].sum(),class_totall[(i-1)*6:i*6,3].sum()])
            ceres_hours.append(ceres_totall[(i-1)*6:i*6].sum()/6)
        class_hours = np.array(class_hours)
        ceres_hours = np.array(ceres_hours)
        fig ,ax = plt.subplots()
        ax.bar(x_axis, class_hours[:, 1],label = 'Land')
        ax.bar(x_axis, class_hours[:,2], bottom=class_hours[:,1], label = 'Smoke')
        ax.bar(x_axis, class_hours[:, 3], bottom=class_hours[:, 2],label = 'Cloud')
        ax.set_ylabel('pix_num')
        plt.legend()
        # plt.bar(x_axis, class_hours[:, 3], bottom=class_hours[:, 2])
        #plt.plot(x_axis,ceres_hours)
        ax2 = plt.twinx()
        ax2.plot(x_axis,ceres_hours,color='red')
        for i in range(7):
            ax2.text(x_axis[i],ceres_hours[i],'%.0f' % ceres_hours[i],fontdict={'fontsize':14})
        ax2.set_ylabel('W/m²')
        plt.title(result_pth.split('\\')[-1][:-4])

        plt.savefig(result_pth)
        #plt.show()

if __name__ == '__main__':
    d = graph('ini_toa_lw_all_1h') #ini_toa_lw_all_1h |ini_sfc_lw_down_all_1h
    CERES = d.load_CERES()
    h8 = d.load_H8()
    smoke = d.load_Smoke()
    c = d.load_classification(h8,smoke)
    # 自定义部分
    classfication, ceres= d.sample(c,CERES,300,358)
    c_t,cls_t = d.trace_whole(classfication,ceres)
    d.mk_a_graph(c_t,cls_t,r'E:\Radiative Effect\Picture\test1_lw_all_300_358.png')

    classfication, ceres = d.sample(c, CERES, 350, 358)
    c_t, cls_t = d.trace_whole(classfication, ceres)
    d.mk_a_graph(c_t, cls_t, r'E:\Radiative Effect\Picture\test1_lw_all_350_358.png')

    classfication, ceres = d.sample(c, CERES, 300, 308)
    c_t, cls_t = d.trace_whole(classfication, ceres)
    d.mk_a_graph(c_t, cls_t, r'E:\Radiative Effect\Picture\test1_lw_all_300_308.png')

    classfication, ceres = d.sample(c, CERES, 350, 308)
    c_t, cls_t = d.trace_whole(classfication, ceres)
    d.mk_a_graph(c_t, cls_t, r'E:\Radiative Effect\Picture\test1_lw_all_350_308.png')
