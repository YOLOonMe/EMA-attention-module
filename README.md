# EMA-attention-module


# Results 

- Training on CIFAR-100 with ResNet.

| Name                     | Resolution | #Params | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|:------------------:|
| ResNet50                 |     32     | 23.71M  |   77.26    |    93.63   |          -         | 
| [+ CBAM](https://github.com/Christian-lyc/NAM)            |     32     | 26.24M  |   80.56    |    95.34   |          -         |
| + SA              |     32     | 23.71M  |   79.92    |    95.00   |          -         | 
| + ECA             |     32     | 23.71M  |   79.68    |    95.05   |          -         |
| [+ NAM](https://github.com/Christian-lyc/NAM)             |     32     | 23.71M  |   80.62    |    95.28   |          -         |
| + CA  |     32     | 25.57M  |   80.17    |    94.94   |          -         |
| + EMA             |     32     | 23.85M  |   80.69    |    95.59   |          [ema_model_best.pth.tar](https://pan.baidu.com/s/14CdNiGyou1sLGcRYLYOVKg?pwd=1234)         |
| ResNet101                |     32     | 42.70M  |   77.78    |    94.39   |          -         |
| + CA |     32     | 46.22M  |   80.01    |    94.78   |          -         |
| + EMA            |     32     | 42.96M  |   80.86    |    95.75   |          -         |

- Training on ImageNet-1k with [MobileNetv2](https://github.com/huggingface/pytorch-image-models).

| Name                          | Resolution | #Params |   MFLOPs   | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| MobileNetv2                   |     224    |  3.50M  |     300    |    72.3    |   91.02    | 
| [+ SE](https://github.com/houqb/CoordAttention)           |     224    |  3.89M  |     300    |    73.5    |     -      |          -         | 
| [+ CBAM](https://github.com/houqb/CoordAttention)          |     224    |  3.89M  |     300    |    73.6    |     -      |          -         | 
| [+ CA](https://github.com/houqb/CoordAttention)|     224    |  3.95M  |     310    |    74.3    |     -      |          -         | 
| + EMA               |     224    |  3.55M  |     306    |    74.32   |   91.82    |          [model_best.pth.tar](https://pan.baidu.com/s/1a1p30h-ZkDUSzKJLTGJSnw?pwd=1234)         | 


- Training on ImageNet-1k with [MobileNetv2](https://github.com/d-li14/mobilenetv2.pytorch).

| Name                     | Resolution | #Params |    MFLOPs   |Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|:------------------:|:------------:|
| MobileNetv2                 |     224     | 3.504M  | 300.79  |   72.192    |    90.534   |          -         | 
| + EMA             |     224     | -  | 302     |    72.55    |    90.89    |        [model_best.pth.tar](https://pan.baidu.com/s/18MS8u9_P-KG9OfpIunRyKA?pwd=1234)         | 


- Training on COCO 2017 with [YOLOv5s](https://github.com/ultralytics/yolov5/tree/v6.0).

| Name                          | Resolution | #Params |   MFLOPs   | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| YOLOv5s (v6.0) |     640    |  7.23M  |     16.5    |    56.0    |   37.2    |       -      | 
| + CBAM         |     640    |  7.27M  |     16.6    |    57.1    |     37.7      |       -      |  
| + SA|     640    |  7.23M  |     16.5    |    56.8      |       37.4      |       -      | 
| + ECA|     640    |  7.23M  |     16.5    |    57.1      |       37.6      |       -      | 
| + CA|     640    |  7.26M  |     16.50    |    57.5    |     38.1      |       -      | 
| + EMA               |     640    |  7.24M  |     16.53    |    57.8   |   38.4    |      [yolov5s.pt](https://pan.baidu.com/s/1_jmjIidvZ2hbMo-m-skmBg?pwd=1234)      | 

- Training on VisDrone 2019 with [YOLOv5x](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression).

| Name                          | Resolution | #Params |   MFLOPs   | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| YOLOv5x (v6.0)               |     640    |  90.96M  |     314.2    |    49.29    |   30.0    |       -      |
| [+ CBAM](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)|     640    |  91.31M  |     315.1    |    49.40      |      30.1      |       -      |
| + CA|     640    |  91.28M  |     315.2    |    49.30    |     30.1      |       -      |
| + EMA               |     640    |  91.18M  |     315.0    |    49.70   |   30.4    |       [yolov5x.pt](https://pan.baidu.com/s/1p-1763222pb3FuXhVtIzbA?pwd=1234)      |


## References
- [NAM](https://github.com/Christian-lyc/NAM)
- [MobileNetv2](https://github.com/huggingface/pytorch-image-models) 
- [MobileNetv2](https://github.com/d-li14/mobilenetv2.pytorch) 
- [YOLOv5s](https://github.com/ultralytics/yolov5/tree/v6.0)
- [YOLOv5x](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)
- [CoordAttention](https://github.com/houqb/CoordAttention)

```
@INPROCEEDINGS{10096516,
  author={Ouyang, Daliang and He, Su and Zhang, Guozhong and Luo, Mingzhu and Guo, Huaiyong and Zhan, Jian and Huang, Zhijie},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Efficient Multi-Scale Attention Module with Cross-Spatial Learning}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096516}}
```
