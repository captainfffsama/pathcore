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

from data import HxqData
from model import WideResnet502

from memory_bank import MemoryBank
"""

/home/chiebotgpuhq/MyCode/dataset/anomaly_hxq/hxq_tiantai/crop/
"""

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="/home/chiebotgpuhq/MyCode/dataset/mvtec/data/hazelnut/train/good/")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--compress_rate",type=float,default=0.001)
    parser.add_argument("--save_dir",type=str,default="/home/chiebotgpuhq/MyCode/python/pytorch/pathcore/compare/")
    args=parser.parse_args()
    return args

def train(args):
    dataset=HxqData(args.img_dir)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

    net=WideResnet502()
    net.eval()
    print("model init done")

    embedding_bank=[]
    print("start get features")
    with torch.no_grad():
        for data in tqdm(dataloader):
            z=net(data)
            features=z.detach().cpu().numpy()
            features=np.transpose(features,(0,2,3,1))
            embedding_bank.append(features.reshape(-1,features.shape[-1]))

    print("get features done,generate memory bank")
    embedding_bank=np.concatenate(embedding_bank,axis=0)
    memory_bank_dealer=MemoryBank()
    memory_bank_dealer.bank_generate(embedding_bank,int(embedding_bank.shape[0]*args.compress_rate),args.save_dir)
    print("generate memory bank done")


if __name__=="__main__":
    args=parse_args()
    train(args)



