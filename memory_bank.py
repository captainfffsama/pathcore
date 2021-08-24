# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 16:22:07 CST
@Description: memory bank
'''
import os

import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
import faiss
from sklearn.metrics import pairwise_distances

from center_cluster import FacilityLocationMethod

from utils import timeblock

import debug_tools as DT

class MemoryBank(object):
    def __init__(self,npy_path="",input_size=(52,52)):
        self.dim_reduction=SparseRandomProjection(n_components='auto',eps=0.9)
        self.center_clus=FacilityLocationMethod()
        if os.path.exists(npy_path):
            self.memory_bank:np.ndarray=np.load(npy_path)

        self.memory_bank_dataset=None
        self.input_size=input_size
        self.pair_dist=None

    @property
    def max_dis(self):
        assert hasattr(self,"memory_bank"),"must init self.memory_bank first"
        if self.pair_dist is None:
            print("first ask max distance,computing")
            self.pair_dist=pairwise_distances(self.memory_bank,self.memory_bank,metric='euclidean')
        return self.pair_dist.max()

    @property
    def mean_dis(self):
        assert hasattr(self,"memory_bank"),"must init self.memory_bank first"
        if self.pair_dist is None:
            print("first ask max distance,computing")
            self.pair_dist=pairwise_distances(self.memory_bank,self.memory_bank,metric='euclidean')
        return np.mean(self.pair_dist)

    @property
    def min_dis(self):
        assert hasattr(self,"memory_bank"),"must init self.memory_bank first"
        if self.pair_dist is None:
            print("first ask max distance,computing")
            self.pair_dist=pairwise_distances(self.memory_bank,self.memory_bank,metric='euclidean')
        return self.pair_dist[self.pair_dist!=0].min()


    def bank_generate(self,embedding,center_num,save_dir):
        if isinstance(embedding,list):
            embedding=np.array(embedding)
        # 生成映射矩阵
        with timeblock("dim_reduce spend time:"):
            self.features=self.dim_reduction.fit_transform(embedding)
        with timeblock("center clus spend time:"):
            centers_idx_list=self.center_clus(self.features,center_num)
        self.memory_bank=embedding[centers_idx_list]

        np.save(os.path.join(save_dir,'memory_bank.npy'),self.memory_bank)

    def _generate_score_map(self,D,x):
        anomaly_map=np.reshape(D[:,0],(-1,*self.input_size))
        max_distance=np.max(D[:,0])
        max_distance_idx=np.argmax(D[:,0])

        N_b = D[max_distance_idx]
        w = (1 - (np.exp(max_distance)/np.sum(np.exp(N_b))))
        score = w*max_distance  # Image-level score
        return score,anomaly_map


    def get_score_map(self,query:np.ndarray,k=20):
        if self.memory_bank_dataset is None:
            self.memory_bank_dataset= faiss.IndexFlatL2(self.memory_bank.shape[-1])
            self.memory_bank_dataset.add(self.memory_bank)
            # self.memory_bank_dataset = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',
            #                     metric='minkowski', p=2).fit(self.memory_bank)


        if not query.flags.c_contiguous:
            query=np.ascontiguousarray(query)
        D,I=self.memory_bank_dataset.search(query,k)
        D=np.sqrt(D)
        # D, _ = self.memory_bank_dataset.kneighbors(query)
        score,anomaly_map= self._generate_score_map(D,query)
        return score,anomaly_map
