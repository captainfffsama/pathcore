import time
from contextlib import contextmanager
from typing import Union
import torch
import torch.nn.functional as F
import numpy as np

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

@contextmanager
def timeblock(label:str = '\033[1;34mSpend time:\033[0m'):
    r'''上下文管理测试代码块运行时间,需要
        import time
        from contextlib import contextmanager
    '''
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('\033[1;34m{} : {}\033[0m'.format(label, end - start))

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range *255

def norm2npimg(x:Union[torch.Tensor,np.ndarray]) -> np.ndarray:
    if isinstance(x,torch.Tensor):
        x=x.detach().cpu().numpy()
    x=x.squeeze()
    assert len(x.shape) <4,"x must be have less 4 dim"
    if 3==len(x.shape) and (x.shape[0]==1 or x.shape[0]==3):
        x=np.transpose(x,(1,2,0))
    x=normalization(x)
    x=x.astype(np.uint8)
    return x

