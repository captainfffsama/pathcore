import math
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as Norm
import numpy as np
import torch
from torchvision.utils import make_grid



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range *255


def show_img(img_ori,cvreader=True):
    img=deepcopy(img_ori)
    if isinstance(img, list) or isinstance(img, tuple):
        img_num = len(img)
        row_n = math.ceil(math.sqrt(img_num))
        col_n = max(math.ceil(img_num / row_n), 1)
        fig, axs = plt.subplots(row_n, col_n, figsize=(15 * row_n, 15 * col_n))
        for idx, img_ in enumerate(img):
            if isinstance(img_,torch.Tensor) or isinstance(img_,np.ndarray):
                img_=show_tensor(img_,cvreader)
            if 2 == len(axs.shape):
                axs[idx % row_n][idx // row_n].imshow(img_)
                axs[idx % row_n][idx // row_n].set_title(str(idx))
            else:
                axs[idx % row_n].imshow(img_)
                axs[idx % row_n].set_title(str(idx))
        plt.show()
    elif isinstance(img,torch.Tensor) or isinstance(img,np.ndarray):
        img=show_tensor(img,cvreader)
        plt.imshow(img)
        plt.show()
    else:
        if hasattr(img,'show'):
            img.show()


def show_tensor(img,cvreader):
    if len(img.shape) == 4 and img.shape[0] !=1:
        if isinstance(img,np.ndarray):
            img=torch.Tensor(img)
            img=make_grid(img)
        else:
            img=make_grid(img)
        img: np.ndarray = img.detach().cpu().numpy().squeeze()
    if isinstance(img,torch.Tensor):
        img=img.detach().cpu().numpy().squeeze()
    if 1==img.max():
        img=img.astype(np.float) 
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.min() <0 or img.max() >255:
        img=normalization(img)
        print("img have norm")
        if img.shape[-1] == 3:
            img=img.astype(np.uint8)
    if cvreader and len(img.shape)==3:
        img=img[:,:,::-1]
    return img