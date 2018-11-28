'''
training
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
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay (SGD only)')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

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
parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')

opt = parser.parse_args()
print (opt)

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

# 1. Load data
train_data = HandPointDataset(root_path='../preprocess', opt=opt, train = True)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
										  shuffle=True, num_workers=int(opt.workers), pin_memory=False)
										  
test_data = HandPointDataset(root_path='../preprocess', opt=opt, train = False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)
										  
print('#Train data:', len(train_data), '#Test data:', len(test_data))
print (opt)

# 2. Define model, loss and optimizer
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
optimizer = optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# 3. Training and testing
for epoch in range(opt.nepoch):
	scheduler.step(epoch)
	print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %(epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
	# 3.1 switch to train mode
	torch.cuda.synchronize()
	netR.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()

	for i, data in enumerate(tqdm(train_dataloader, 0)):
		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()       
		# 3.1.1 load inputs and targets
		points, volume_length, gt_pca, gt_xyz = data
		gt_pca = Variable(gt_pca, requires_grad=False).cuda()
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()

		# points: B * 1024 * 6; target: B * 42
		inputs_level1, inputs_level1_center = group_points(points, opt)
		inputs_level1, inputs_level1_center = Variable(inputs_level1, requires_grad=False), Variable(inputs_level1_center, requires_grad=False)

		# 3.1.2 compute output
		optimizer.zero_grad()
		estimation = netR(inputs_level1, inputs_level1_center)
		loss = criterion(estimation, gt_pca)*opt.PCA_SZ

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		optimizer.step()
		torch.cuda.synchronize()
		
		# 3.1.4 update training error
		train_mse = train_mse + loss.data[0]*len(points)
		
		# 3.1.5 compute error in world cs      
		outputs_xyz = train_data.PCA_mean.expand(estimation.data.size(0), train_data.PCA_mean.size(1))
		outputs_xyz = torch.addmm(outputs_xyz, estimation.data, train_data.PCA_coeff)
		diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM,3)
		diff_sum = torch.sum(diff,2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
		diff_mean_wld = torch.mul(diff_mean,volume_length)
		train_mse_wld = train_mse_wld + diff_mean_wld.sum()
		
	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

	# print mse
	train_mse = train_mse / len(train_data)
	train_mse_wld = train_mse_wld / len(train_data)
	print('mean-square error of 1 sample: %f, #train_data = %d' %(train_mse, len(train_data)))
	print('average estimation error in world coordinate system: %f (mm)' %(train_mse_wld))

	torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
	torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
	
	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	netR.eval()
	test_mse = 0.0
	test_wld_err = 0.0
	timer = time.time()
	for i, data in enumerate(tqdm(test_dataloader, 0)):
		torch.cuda.synchronize()
		# 3.2.1 load inputs and targets
		points, volume_length, gt_pca, gt_xyz = data
		gt_pca = Variable(gt_pca, volatile=True).cuda()
		points, volume_length, gt_xyz = points.cuda(), volume_length.cuda(), gt_xyz.cuda()
		
		# points: B * 1024 * 6; target: B * 42
		inputs_level1, inputs_level1_center = group_points(points, opt)
		inputs_level1, inputs_level1_center = Variable(inputs_level1, volatile=True), Variable(inputs_level1_center, volatile=True)
		
		# 3.2.2 compute output
		estimation = netR(inputs_level1, inputs_level1_center)
		loss = criterion(estimation, gt_pca)*opt.PCA_SZ

		torch.cuda.synchronize()
		test_mse = test_mse + loss.data[0]*len(points)

		# 3.2.3 compute error in world cs        
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
	test_mse = test_mse / len(test_data)
	print('mean-square error of 1 sample: %f, #test_data = %d' %(test_mse, len(test_data)))
	test_wld_err = test_wld_err / len(test_data)
	print('average estimation error in world coordinate system: %f (mm)' %(test_wld_err))
	# log
	logging.info('Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, lr = %f' %(epoch, train_mse, train_mse_wld, test_mse, test_wld_err, scheduler.get_lr()[0]))