
# Towards Total Recall in Industrial Anomaly Detection

Here just a unofficial implement of PatchCore.

The original paper can be find on: <https://arxiv.org/abs/2106.08265>

# requirements

opencv
pytorch
faiss
scikit-learn
tqdm
matplotlib

# how to use
把正常图片放一个文件夹,然后运行 `train.py` 生成memory bank 到compare文件夹,比如:
```python
python train.py --img_dir yourImageDir --save_dir memoryBankSaveDir
```

然后运行 `test.py` 加载上一步的memory_bank.npy 测试测试数据
```python
python test.py --img_dir testImageDir --memory_bank memoryBankSaveDir
```

# Code Reference

<https://github.com/hcw-00/PatchCore_anomaly_detection>
