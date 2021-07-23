# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 16:22:07 CST
@Description: memory bank
'''
import os

import numpy as np
from sklearn.random_projection import SparseRandomProjection

from center_cluster import FacilityLocationMethod

class MemoryBank(object):
    def __init__(self):
        self.dim_reduction=SparseRandomProjection(n_components='auto',eps=0.9)
        self.center_clus=FacilityLocationMethod()



    def bank_generate(self,embedding,center_num,save_dir):
        if isinstance(embedding,list):
            embedding=np.array(embedding)
        # 生成映射矩阵
        self.features=self.dim_reduction.fit_transform(embedding)
        centers_idx_list=self.center_clus(self.features,center_num)
        self.memory_bank=embedding[centers_idx_list]

        np.save(os.path.join(save_dir,'memory_bank.npy'),self.memory_bank)

