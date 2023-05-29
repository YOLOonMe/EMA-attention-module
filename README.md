# EMA-attention-module


# Results on ImageNet-1k with MobileNetv2 
| name | resolution | #params | FLOPs | Top-1 Acc. | Top-5 Acc. |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetv2 | 224 | 300M | 3.50 | 72.3 | 91.02 |
| MobileNetv2-SE [8]| 224 | 3.89M | 300 | 73.5 | - |
| MobileNetv2-CBAM [8]| 224 | 3.89M | 300 | 73.6 | - |
| MobileNetv2-CoordAttention [8]| 224 | 3.95M | 310 | 74.3 | - |
| MobileNetv2-EMA| 224 | 3.55M | 306 | 74.32 | 91.82 |
