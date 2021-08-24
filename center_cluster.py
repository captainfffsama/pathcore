# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 16:39:57 CST
@Description:   用来做聚类选点的方法
'''
import random

import numpy as np
from sklearn.metrics import pairwise_distances

# FIXME: 计算效率很低,需要优化
class FacilityLocationMethod(object):
    def __init__(self):
        # self.min_distance中记录了每个点到自己对应的center的最近距离
        self.min_distance=None


    def _update_min_distance(self,center_idx,embedding):
        if not isinstance(center_idx,list):
            center_idx=[center_idx,]
        center_vector=embedding[center_idx]

        pair_dist=pairwise_distances(embedding,center_vector,metric='euclidean')
        
        if self.min_distance is None:
            self.min_distance=np.min(pair_dist,axis=1).reshape(-1,1)
        else:
            # 保证self.min_distance 中对应位置的元素记录的是该元素距离最近的center的距离
            self.min_distance=np.minimum(self.min_distance,pair_dist)

    def __call__(self,x:np.ndarray,n:int):
        """get n center from m data,

        Args:
            x: np.ndarray
                需要聚类的数据,行是数据量轴,列是特征轴
            n: int
                需要多少个聚类中心
        """
        center=[]
        for i in range(n):
            if self.min_distance is None:
                start_idx= random.randint(0,x.shape[0]-1)
            else:
                start_idx=np.argmax(self.min_distance)
            self._update_min_distance(start_idx,x)
            center.append(start_idx)

        print("max distance is:{}".format(max(self.min_distance)))

        return center

        
        

