# EMA-attention-module


# Results 

## Results on CIFAR-100 with ResNet50 and ResNet101
| name | resolution | #params | Top-1 Acc. | Top-5 Acc. |
| :---: | :---: | :---: | :---: | :---: |
| ResNet50 | 32 | 23.71M | 77.26 | 93.63 |
| ResNet50-CBAM [16]| 32 | 26.24M | 80.56 | 95.34 |
| ResNet50-SE| 32 | 23.71M | 79.92 | 95.00 |
| ResNet50-ECA| 32 | 23.71M | 79.68 | 95.05 |
| ResNet50-NAM [16]| 32 | 23.71M | 80.62 | 95.28 |
| ResNet50-CoordAttention [8]| 32 | 25.57M | 80.17 | 94.94 |
| ResNet50-EMA| 32 | 23.85M | 80.69 | 95.59 |
| ResNet101| 32 | 42.70M | 77.78 | 94.39 |
| ResNet101-CoordAttention| 32 | 46.22M | 80.01 | 94.78 |
| ResNet101-EMA| 32 | 42.96M | 80.86 | 95.75 |

## Results on ImageNet-1k with MobileNetv2 
| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetv2 | 224 | 300M | 3.50 | 72.3 | 91.02 |
| MobileNetv2-SE [8]| 224 | 3.89M | 300 | 73.5 | - |
| MobileNetv2-CBAM [8]| 224 | 3.89M | 300 | 73.6 | - |
| MobileNetv2-CoordAttention [8]| 224 | 3.95M | 310 | 74.3 | - |
| MobileNetv2-EMA| 224 | 3.55M | 306 | 74.32 | 91.82 |

## Results on COCO 2017 with Yolov5s(v6.0）

## Results on VisDrone 2019 with Yolov5X (v6.0）
