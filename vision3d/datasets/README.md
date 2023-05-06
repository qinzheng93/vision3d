# Data Preparation

## Object Classification

### ModelNet40

Vision3D uses pre-processed ModelNet40 dataset from PointNet++.
Download the dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).
Modify `data_root` in `tools/datasets/modelnet40.py` and run the file for further pre-processing.
The script will generate 5 hdf5 files for training and 2 hdf5 files for testing with the corresponding id json files.
You can also use the dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).

## Part Semantic Segmentation

### ShapeNetPart

Vision3D uses pre-processed ShapeNetPart dataset with normal from PointNet++.
Download the dataset from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).
Modify `data_root` in `tools/datasets/shapenetpart.py` and run the file for further pre-processing.
The script will generate 4 pickle files for train, val, trainval and test, respectively.

## Semantic Segmentation

### S3DIS

Vision3D supports both on-the-fly training augmentation and preprocessed data.
Before using S3DIS in Vision3D, you need to generate `.npy` files following the instructions in [pointnet](https://github.com/charlesq34/pointnet).
For preprocessed hdf5 training data, download from [here](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip).
Whole-scene evaluation should be used for testing.
Modify `data_root` in `tools/datastes/s3dis.py` and run the file to generate whole-scene testing hdf5 files for fast evaluation.

### ScanNet-v2

## Point Cloud Registration

### 3DMatch

### KITTI odometry

### ModelNet40
