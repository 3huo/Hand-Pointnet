The released code implements the Hand Pose Regression Network of the following paper:  
"Liuhao Ge, Yujun Cai, Junwu Weng and Junsong Yuan. Hand PointNet: 3D Hand Pose Estimation using Point Sets. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2018."
Please cite this paper if you use our released code.

1. Database

Download the MSRA Hand Gesture database at:
https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0

Unzip it and put folders "P0"~"P9" in the "data/cvpr15_MSRAHandGestureDB" directory.

Please cite "Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang and Jian Sun. Cascaded Hand Pose Regression. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015", if you use this database.

2. Installation

Install PyTorch 0.3.0. You may also need to install tqdm and progressbar. The code has been tested with Python 2.7, PyTorch 0.3.0, CUDA 8.0 and cuDNN 5.1 on Ubuntu 16.04.

3. Usage

3.1 Data Preprocessing

cd preprocess
matlab preprocess.m

3.2 Train and evaluation

cd train_eval
python train.py
python eval.py

I put one pretrained model in 'train_eval/results/P0' which is trained on "P1"~"P9", and is evaluated on "P0".