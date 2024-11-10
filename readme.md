# DGCNN 3D Point Cloud Classification

English / [中文](readme_zh.md)

↑ 点击切换语言

This project implements a highly accurate 3D point cloud classification task using the **DGCNN** framework.

Due to hardware limitations, the project did not follow the official recommended parameters of DGCNN. Instead, we reduced the k-value and randomized the sampling points, which helped reduce the model's computational complexity. However, with a maximum accuracy of 89.59% after 50 epochs, it is expected that with optimized parameter settings, the accuracy could reach around 91-92%.

The graph below shows the trend of training loss during the training process.

![Demo](plot/loss_plot.png)

The graph below shows the trend of test accuracy during the training process.

![Demo](plot/acc_plot.png)

**Training Duration Reference**

## Table of Contents

- [Environment Setup](#Environment-Setup)
- [Multilingual Support](#Multilingual-Support)
- [Dataset](#Dataset)
- [File Structure](#File-Structure)
- [License](#License)
- [Contributions](#Contributions)

## Environment Setup

```
CUDA 12.1
Python 3.9.13
PyTorch 2.1.0
torch-geometric 1.6.3
torch-cluster 1.6.3
h5py 3.12.1
numpy 1.24.3
pandas 2.2.3
matplotlib 3.9.2
```

## Multilingual Support

To make the code easier to understand for developers from different linguistic backgrounds, this project provides comments in both English and Chinese.

## Dataset

The ModelNet40 dataset used in this project is a processed version stored in HDF5 format, sourced from [GitHub](https://github.com/antao97/PointCloudDatasets).

## File Structure

The project file structure is as follows:

```c++
3D_Classification_DGCNN/
│
├── model/ 
│   └── model.pt
│
├── plot/ 
│   ├── acc_plot.png
│   ├── loss_plot.png
│   └── plot.ipynb
│
├── utils(en/zh)/ 
│   ├── data/ 
│   │    └── modelnet40_hdf5_2048 (please download it yourself)
│   ├── dataloader.py
│   ├── DGCNN.py
│   ├── EdgeConvfeature.ipynb
│   └── train.py
│
├── train.csv
├── readme.md
└── main.py 
```

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Contributions

Contributions of any kind are welcome! Whether it's reporting issues or making suggestions, thank you very much!!