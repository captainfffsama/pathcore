# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 10:32:12 CST
@Description: path core 训练
'''
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import faiss

from data import HxqData
from model import WideResnet502
from memory_bank import MemoryBank
from utils import reshape_embedding

import debug_tools as D
"""
/home/chiebotgpuhq/MyCode/dataset/mvtec/data/hazelnut/test/crack
/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/hxq_tiantai/test_crop/
/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/hxq_tiantai/memory_bank.npy
"""

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="/home/chiebotgpuhq/MyCode/dataset/mvtec/1")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--memory_bank",type=str,default="/home/chiebotgpuhq/MyCode/python/pytorch/pathcore/compare/memory_bank.npy")
    args=parser.parse_args()
    return args

def train(args):
    dataset=HxqData(args.img_dir)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

    memory_bank_dealer=MemoryBank(args.memory_bank,(28,28))
    net=WideResnet502()
    net.eval()
    print("model init done")

    print("start get features")
    with torch.no_grad():
        for data in tqdm(dataloader):
            z=net(data)
            z=z.cpu().numpy()
            feature=np.load("/home/chiebotgpuhq/MyCode/python/pytorch/pathcore/compare/features.npy")
            print("feature diff mean is:",np.mean(feature-z))
            img_z=np.transpose(z,(0,2,3,1)).reshape((-1,z.shape[1]))
            score,anomaly_map=memory_bank_dealer.get_score_map(img_z,9)
            print("score is:",score)
            D.show_img([data,anomaly_map])



if __name__=="__main__":
    args=parse_args()
    train(args)



