# 阿里巴巴点云分割(PointNet)

本项目是点云分割模型PointNet在DataFountain平台上阿里巴巴自动驾驶三维点云分割比赛
https://www.datafountain.cn/competitions/314/details/rule
数据集上的代码。模型和代码结构基本跟PointNet保持一致，主要修改了train部分和inference部分的代码。结果提交了训练赛，效果一般，大概在复赛水平十几名的样子，仅通过此深入学习经典的点云分割网络。

## Requirements

* Python==3.x
* TensorFlow==1.13.0

## Training

- STEP 1. 将csv数据转成npy数据，方便读取，此处直接借鉴https://github.com/kiclent/pointSeg 的代码，将每100帧的的数据合并到一个6列的矩阵里，保存npy格式

  ```python
  python data_merge.py
  ```

- STEP 2. 训练模型

  ```python
  python train.py
  ```

- STEP 3. 修改hParam.py中的is_training参数，改为False进行Inference

## Results

分割效果如图所示，可见车辆效果较好，其他效果不理想

![阿里PointNet结果2](https://github.com/Summit11/PointNet_AliData/blob/master/img/res1.png)



![阿里PointNet数据集结果](https://github.com/Summit11/PointNet_AliData/blob/master/img/res2.png)
