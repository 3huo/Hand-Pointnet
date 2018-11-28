'''
evaluation
author: Liuhao Ge
'''
import argparse
import os
import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import HandPointDataset
from dataset import subject_names
from dataset import gesture_names
from network import PointNet_Plus
from utils import group_points

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default = 42,  help='number of PCA components')
parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015, help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04, help='square of radius for ball query in level 2')

parser.add_argument('--test_index', type=int, default = 0,  help='test index for cross validation, range: 0~8')
parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = 'pretrained_net.pth',  help='model name for training resume')

opt = parser.parse_args()
print (opt)

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

# 1. Load data                                         
test_data = HandPointDataset(root_path='../preprocess', opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers), pin_memory=False)
                                          
print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
netR = PointNet_Plus(opt)
if opt.ngpu > 1:
    netR.netR_1 = torch.nn.DataParallel(netR.netR_1, range(opt.ngpu))
    netR.netR_2 = torch.nn.DataParallel(netR.netR_2, range(opt.ngpu))
    netR.netR_3 = torch.nn.DataParallel(netR.netR_3, range(opt.ngpu))
if opt.model != '':
    netR.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    
netR.cuda()
print(netR)

criterion = nn.MSELoss(size_average=True).cuda()

# 3. evaluation
torch.cuda.synchronize()

netR.eval()
test_mse = 0.0
test_wld_err = 0.0
timer = time.time()
for i, data in enumerate(tqdm(test_dataloader, 0)):
	torch.cuda.synchronize()
	# 3.1 load inputs and targets
	points, volume_length, gt_pca, gt_xyz = data
	gt_pca = Variable(gt_pca, volatile=True).cuda()
	points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
	
	# points: B * 1024 * 6
	inputs_level1, inputs_level1_center = group_points(points, opt)
	inputs_level1, inputs_level1_center = Variable(inputs_level1, volatile=True), Variable(inputs_level1_center, volatile=True)
	
	# 3.2 compute output
	estimation = netR(inputs_level1, inputs_level1_center)
	loss = criterion(estimation, gt_pca)*opt.PCA_SZ
	torch.cuda.synchronize()
	test_mse = test_mse + loss.data[0]*len(points)

	# 3.3 compute error in world cs        
	outputs_xyz = test_data.PCA_mean.expand(estimation.data.size(0), test_data.PCA_mean.size(1))
	outputs_xyz = torch.addmm(outputs_xyz, estimation.data, test_data.PCA_coeff)
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
	diff_sum = torch.sum(diff,2)
	diff_sum_sqrt = torch.sqrt(diff_sum)
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
	diff_mean_wld = torch.mul(diff_mean,volume_length)
	test_wld_err = test_wld_err + diff_mean_wld.sum()
	
# time taken
torch.cuda.synchronize()
timer = time.time() - timer
timer = timer / len(test_data)
print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

# print mse
test_wld_err = test_wld_err / len(test_data)
print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))