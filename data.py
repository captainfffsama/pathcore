# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 07月 20日 星期二 10:34:57 CST
@Description: pathcore data
'''
import os
from typing import Union,List, Tuple

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

def get_all_file_path(file_dir:str,filter_:tuple=('.jpg',)) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
            for filename in file_name_list \
            if os.path.splitext(filename)[1] in filter_ ]

class HxqData(Dataset):
    def __init__(self,img_dir:str,img_shape:Union[list,int,tuple]=(416,416)):
        super(HxqData, self).__init__()
        self.all_img_list=get_all_file_path(img_dir)
        self.img_transforms=transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        
    def __len__(self):
        return len(self.all_img_list)

    def __getitem__(self,idx):
        img=Image.open(self.all_img_list[idx])
        return self.img_transforms(img)

