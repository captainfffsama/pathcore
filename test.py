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
import cv2

from data import HxqData
from model import WideResnet502
from memory_bank import MemoryBank
from utils import norm2npimg

import debug_tools as D
"""
/home/chiebotgpuhq/MyCode/dataset/mvtec/1
/home/chiebotgpuhq/MyCode/dataset/mvtec/data/hazelnut/test/crack
"""

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str,default="/data/anomaly_hxq/anomaly210804/annomaly/hxq")
    parser.add_argument("--batch_size",type=int,default=2)
    parser.add_argument("--memory_bank",type=str,default="/home/chiebotgpuhq/MyCode/python/pytorch/pathcore/compare/memory_bank.npy")
    parser.add_argument("--input_size",type=tuple,default=(416,416))
    args=parser.parse_args()
    return args

def generate_result_show_img(img,anomaly_map:np.ndarray):
    anomaly_map=anomaly_map.squeeze()
    anomaly_map=cv2.resize(anomaly_map,(img.shape[1],img.shape[0]))
    anomaly_map=norm2npimg(anomaly_map)
    anomaly_map=cv2.GaussianBlur(anomaly_map,(0,0),4,4)
    anomaly_color_map=cv2.applyColorMap(anomaly_map,cv2.COLORMAP_JET)
    rate=0.8
    result=cv2.addWeighted(img,rate,anomaly_color_map,1-rate,0.0)
    return result

def test(args):
    dataset=HxqData(args.img_dir,args.input_size)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

    memory_bank_dealer=MemoryBank(args.memory_bank,(args.input_size[0]//8,args.input_size[1]//8))
    print("max distance is:",memory_bank_dealer.max_dis)
    print("min distance is:",memory_bank_dealer.min_dis)
    print("mean distance is:",memory_bank_dealer.mean_dis)
    net=WideResnet502()
    net.eval()
    print("model init done")

    print("start get features")
    with torch.no_grad():
        for data in tqdm(dataloader):
            z=net(data).detach().permute(0,2,3,1)
            for idx,one_sample in enumerate(z):
                img_z=one_sample.reshape((-1,z.shape[-1])).cpu().numpy()
                score,anomaly_map=memory_bank_dealer.get_score_map(img_z,9)
                print("score is:",score)
                result=generate_result_show_img(norm2npimg(data[idx]),anomaly_map)
                D.show_img([result,norm2npimg(data[idx])[:,:,::-1]])



if __name__=="__main__":
    args=parse_args()
    test(args)



