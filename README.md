# Vision3D: A 3D Vision Library built with PyTorch

Vision3D is a 3D vision library built with PyTorch, in order to reproduce the state-of-the-art 3D vision models.

## Features

1. State-of-the-art models:
   1. Point cloud processing:
      1. PointNet series: PointNet, PointNet++.
      2. DGCNN.
      3. PointCNN.
      4. Point Transformer.
      5. KPConv.
   2. Point cloud registration:
      1. FCGF.
      2. D3Feat.
      3. Predator.
      4. GeoTransformer
      5. RPM-Net.
2. Multiple 3D vision tasks:
   1. Object Classification.
   2. Semantic Segmentation.
   3. Registration.
   4. Completion.
3. Multiple datasets:
   1. Object Classification: ModelNet40.
   2. Part Semantic Segmentation: ShapeNetPart.
   3. Semantic Segmentation: S3DIS.
   4. Point Cloud Registration: ModelNet40, 3DMatch, KITTI odometry, 4DMatch, DeepDeform, CAPE, Augmented ICL-NUIM.
4. Multi-GPU Training with DDP.

## Installation

Vision3D is tested on Python 3.8, PyTorch 1.13.1, Ubuntu 22.04, GCC 11.3 and CUDA 11.7, but it should work with other configurations. Currently, Vision3d only support training and testing on GPUs.

Install Vision3D with the following command:

```bash
python setup.py develop
```

All requirements will be automatically installed.

## Acknowledgements

- [PointNet](https://github.com/charlesq34/pointnet)
- [PointNet2](https://github.com/charlesq34/pointnet2)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [PointCNN](https://github.com/yangyanli/PointCNN)
- [MMCV](https://github.com/open-mmlab/mmcv)
- [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
- [FCGF](https://github.com/chrischoy/FCGF)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
- [Predator](https://github.com/prs-eth/OverlapPredator)
- [RPM-Net](https://github.com/yewzijian/RPMNet)
- [UnsupervisedR&R](https://github.com/mbanani/unsupervisedRR)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [VectorNeuron](https://github.com/FlyingGiraffe/vnn)
- And more.
