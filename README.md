# EMA-attention-module


# Results 

- Training on CIFAR-100 with ResNet50 and ResNet101 for 200 epochs.

| Name                     | Resolution | #Params | Top-1 Acc. | Top-5 Acc. |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|
| ResNet50                 |     32     | 23.71M  |   77.26    |    93.63   | 
| ResNet50-CBAM            |     32     | 26.24M  |   80.56    |    95.34   |
| ResNet50-SE              |     32     | 23.71M  |   79.92    |    95.00   | 
| ResNet50-ECA             |     32     | 23.71M  |   79.68    |    95.05   |
| ResNet50-NAM             |     32     | 23.71M  |   80.62    |    95.28   |
| ResNet50-CoordAttention  |     32     | 25.57M  |   80.17    |    94.94   |
| ResNet50-EMA             |     32     | 23.85M  |   80.69    |    95.59   |
| ResNet101                |     32     | 42.70M  |   77.78    |    94.39   |
| ResNet101-CoordAttention |     32     | 46.22M  |   80.01    |    94.78   |
| ResNet101-EMA            |     32     | 42.96M  |   80.86    |    95.75   |

- Training on ImageNet-1k with [MobileNetv2](https://github.com/huggingface/pytorch-image-models) for 400 epochs.

| Name                          | Resolution | #Params |   MFLOPs   | Top-1 Acc. | Top-5 Acc. |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|
| MobileNetv2                   |     224    |  3.50M  |     300    |    72.3    |   91.02    | 
| MobileNetv2-[SE](https://github.com/houqb/CoordAttention)           |     224    |  3.89M  |     300    |    73.5    |     -      |
| MobileNetv2-[CBAM](https://github.com/houqb/CoordAttention)          |     224    |  3.89M  |     300    |    73.6    |     -      |
| MobileNetv2-[CoordAttention](https://github.com/houqb/CoordAttention)|     224    |  3.95M  |     310    |    74.3    |     -      |
| MobileNetv2-EMA               |     224    |  3.55M  |     306    |    74.32   |   91.82    |


- Training on ImageNet-1k with [MobileNetv2](https://github.com/d-li14/mobilenetv2.pytorch) for 200 epochs.

| Name                     | Resolution | #Params |    MFLOPs   |Top-1 Acc. | Top-5 Acc. |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|:------------------:|
| [MobileNetv2](https://github.com/d-li14/mobilenetv2.pytorch)                 |     224     | 3.504M  | 300.79  |   72.192    |    90.534   |
| MobileNetv2-EMA             |     224     | -  | 302     |    72.55    |    90.89    |

## Results on COCO 2017 with Yolov5s (v6.0）

## Results on VisDrone 2019 with Yolov5X (v6.0）



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
