# EMA-attention-module

 
# Results 

## Training on CIFAR-100 with ResNet for 200 epochs.
- Train
  ```shell
  CUDA_VISIBLE_DEVICES=0 python train_cifar100.py --b 128
  ```
| Name                     | Resolution | #Params | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|:------------------:|
| ResNet50                 |     32     | 23.71M  |   77.26    |    93.63   |          -         | 
| [+ CBAM](https://github.com/Christian-lyc/NAM)            |     32     | 26.24M  |   80.56    |    95.34   |          -         |
| + SA              |     32     | 23.71M  |   79.92    |    95.00   |          -         | 
| + ECA             |     32     | 23.71M  |   79.68    |    95.05   |          -         |
| [+ NAM](https://github.com/Christian-lyc/NAM)             |     32     | 23.71M  |   80.62    |    95.28   |          -         |
| + CA  |     32     | 25.57M  |   80.17    |    94.94   |          -         |
| + EMA             |     32     | 23.85M  |   80.69    |    95.59   |          [ema](https://pan.baidu.com/s/14CdNiGyou1sLGcRYLYOVKg?pwd=1234)         |
| + SSA-32 |     32     | 25.82M  |   80.91    |    95.53   |          -         |

| ResNet101                |     32     | 42.70M  |   77.78    |    94.39   |          -         |
| + CA |     32    | 46.22M  |   80.01    |    94.78   |          -         |
| + EMA            |     32     | 42.96M  |   80.86    |    95.75   |          -         |
| + SSA-32         |     32     | 51.37M  |   80.61    |    95.26   |          -         |

## Training on ImageNet-1k with [MobileNetv2](https://github.com/huggingface/pytorch-image-models)  for 400 epochs.
- Train
  ```shell
  ./distributed_train.sh 2 ./ILSVRC2012/ --model mobilenetv2_100 -b 256 --sched cosine --epochs 400 --decay-epochs 2.4 --decay-rate .97 --opt-eps .001 -j 16 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --amp --lr 0.4 --warmup-epochs 5 
  ```
- Val
  ```shell
  python validate.py ./ILSVRC2012/ --model mobilenetv2_100 --checkpoint model_best.pth.tar --use-ema
  ```
| Name                          | Resolution | #Params |   MFLOPs   | Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| MobileNetv2                   |     224    |  3.50M  |     300    |    72.3    |   91.02    | 
| [+ SE](https://github.com/houqb/CoordAttention)           |     224    |  3.89M  |     300    |    73.5    |     -      |          -         | 
| [+ CBAM](https://github.com/houqb/CoordAttention)          |     224    |  3.89M  |     300    |    73.6    |     -      |          -         | 
| [+ CA](https://github.com/houqb/CoordAttention)|     224    |  3.95M  |     310    |    74.3    |     -      |          -         | 
| + EMA               |     224    |  3.55M  |     306    |    74.32   |   91.82    |          [ema](https://pan.baidu.com/s/1a1p30h-ZkDUSzKJLTGJSnw?pwd=1234)         | 


## Training on ImageNet-1k with [MobileNetv2](https://github.com/d-li14/mobilenetv2.pytorch)  for 200 epochs.
- Train
  ```shell
  python imagenet.py  -a mobilenetv2  -d <path-to-ILSVRC2012-data> --epochs 200 --lr-decay cos --lr 0.05 --wd 4e-5   -c <path-to-save-checkpoints>   --input-size 224 
  ```
  
| Name                     | Resolution | #Params |    MFLOPs   |Top-1 Acc. | Top-5 Acc. | BaiduDrive(models) |
|:------------------------:|:----------:|:-------:|:----------:|:----------:|:------------------:|:------------:|
| MobileNetv2                 |     224     | 3.504M  | 300.79  |   72.192    |    90.534   |          -         | 
| + EMA             |     224     | -  | 302     |    72.55    |    90.89    |        [ema](https://pan.baidu.com/s/18MS8u9_P-KG9OfpIunRyKA?pwd=1234)         | 


## Training on COCO 2017 with [YOLOv5s](https://github.com/ultralytics/yolov5/tree/v6.0)  for 300 epochs.
- Train
  ```shell
  python train.py --data coco.yaml --cfg yolov5s_EMA.yaml --weights yolov5s.pt --batch-size 64 --device 0
  ```
- Val
  ```shell
  python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65 --weights yolov5s.pt 
  ```
  
| Name                          | Resolution | #Params |   MFLOPs   | mAP@.5 | mAP@.5:.95 | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| YOLOv5s  |     640    |  7.23M  |     16.5    |    56.0    |   37.2    |       [yolov5s(v6.0)](https://github.com/ultralytics/yolov5/releases/tag/v6.0)      | 
| + CBAM         |     640    |  7.27M  |     16.6    |    57.1    |     37.7      |        [cbam](https://pan.baidu.com/s/1qj4y9lrgO1DNI2W38IP6Vg?pwd=1234)      |  
| + SA|     640    |  7.23M  |     16.5    |    56.8      |       37.4      |        [sa](https://pan.baidu.com/s/1A_hF7F86VPdtEA8s660nJw?pwd=7wf6)      | 
| + ECA|     640    |  7.23M  |     16.5    |    57.1      |       37.6      |        [eca](https://pan.baidu.com/s/1COOK_ltxTfEpwgu4ieTwAQ?pwd=mu94)      | 
| + CA|     640    |  7.26M  |     16.50    |    57.5    |     38.1      |       [ca](https://pan.baidu.com/s/1coWhu_Ba5OuBtvNyNnCQEg?pwd=bg8u)      | 
| + EMA               |     640    |  7.24M  |     16.53    |    57.8   |   38.4    |      [ema](https://pan.baidu.com/s/110v-K1CmsHDsR2PylarZgA?pwd=qamz)      | 
| + SSA-32               |     640    |  7.27M  |     0    |    58.7   |   38.4    |           | 
| + SSA-16               |     640    |  7.31M  |     0    |    58.1   |   38.5    |        | 
| + SSA-2               |     640    |  8.55M  |     0    |    58.3   |   38.8    |            | 
| + SSA-1               |     640    |  11.50M  |     0    |    58.8   |   39.1    |            | 
## Training on VisDrone 2019 with [YOLOv5x](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression).
- Train
  ```shell
  python train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/accModels/yolov5xP2CBAM.yaml --epochs 300 --batch-size 6 --img 640 --device 0

  ```
- Val
  ```shell
  python val.py --data VisDrone.yaml --img 640 --weights best.pt
  ```
  
| Name                          | Resolution | #Params |   MFLOPs   | mAP@.5 | mAP@.5:.95 | BaiduDrive(models) |
|:-----------------------------:|:----------:|:-------:|:----------:|:----------:|:----------:|:------------------:|
| YOLOv5x (v6.0)               |     640    |  90.96M  |     314.2    |    49.29    |   30.0    |       -      |
| [+ CBAM](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression)|     640    |  91.31M  |     315.1    |    49.40      |      30.1      |       -      |
| + CA|     640    |  91.28M  |     315.2    |    49.30    |     30.1      |       -      |
| + EMA               |     640    |  91.18M  |     315.0    |    49.70   |   30.4    |       [ema](https://pan.baidu.com/s/1p-1763222pb3FuXhVtIzbA?pwd=1234)      |
| + SSA-32               |     640    |  91.18M  |     315.8    |    49.80   |   30.7    |           |

## References
- [NAM](https://github.com/Christian-lyc/NAM)
- [MobileNetv2](https://github.com/huggingface/pytorch-image-models) 
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
