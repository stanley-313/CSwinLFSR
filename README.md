# Requirements:
* **PyTorch 1.12.1, torchvision 0.13.1. The code is tested with python=3.8, cuda=10.2.**
* **Matlab (For result image generation)**

#Train
## 1. Prepare training data
* Download the EPFL, HCInew, HCIold, INRIA and STFgantry datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder **`./datasets/`**.
* Run **`Generate_Data_for_Training.py`** to generate training data
## 2. Begin to train
```
  python train.py --upscale_factor $2/4$
```

# Test
## 1. Prepare test data
* Run **`Generate_Data_for_Test.py`** to generate test data
## 2. Begin to test
```
  python test.py
```
## 3. Generate result image
Run **`GenerateResultImages.m`** to generate result images

# Acknowledgement
This code borrows from [LFT](https://github.com/ZhengyuLiang24/LFT) and [CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer)

# Contact
For any questions welcome to contact us ([3220221027@bit.edu.cn](3220221027@bit.edu.cn), [3220200966@bit.edu.cn](3220200966@bit.edu.cn))