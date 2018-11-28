'''
load hand point data
author: Liuhao Ge
'''
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import scipy.io as sio
import pdb

SAMPLE_NUM = 1024
JOINT_NUM = 21

subject_names = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
gesture_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"]

class HandPointDataset(data.Dataset):
    def __init__(self, root_path, opt, train=True):
        self.root_path = root_path
        self.train = train
        self.size = opt.size
        self.test_index = opt.test_index

        self.PCA_SZ = opt.PCA_SZ
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.JOINT_NUM = opt.JOINT_NUM

        if self.size == 'full':
            self.SUBJECT_NUM = 9
            self.GESTURE_NUM = 17
        elif self.size == 'small':
            self.SUBJECT_NUM = 3
            self.GESTURE_NUM = 2

        self.total_frame_num = self.__total_frmae_num()

        self.point_clouds = np.empty(shape=[self.total_frame_num, self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],
                                     dtype=np.float32)
        self.volume_length = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)
        self.gt_xyz = np.empty(shape=[self.total_frame_num, self.JOINT_NUM, 3], dtype=np.float32)
        self.valid = np.empty(shape=[self.total_frame_num, 1], dtype=np.float32)

        self.start_index = 0
        self.end_index = 0
        if self.train:  # train
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject != self.test_index:
                    for i_gesture in range(self.GESTURE_NUM):
                        cur_data_dir = os.path.join(self.root_path, subject_names[i_subject], gesture_names[i_gesture])
                        print("Training: " + cur_data_dir)
                        self.__loaddata(cur_data_dir)
        else:  # test
            for i_gesture in range(self.GESTURE_NUM):
                cur_data_dir = os.path.join(self.root_path, subject_names[self.test_index], gesture_names[i_gesture])
                print("Testing: " + cur_data_dir)
                self.__loaddata(cur_data_dir)

        self.point_clouds = torch.from_numpy(self.point_clouds)
        self.volume_length = torch.from_numpy(self.volume_length)
        self.gt_xyz = torch.from_numpy(self.gt_xyz)
        self.valid = torch.from_numpy(self.valid)

        self.gt_xyz = self.gt_xyz.view(self.total_frame_num, -1)
        valid_ind = torch.nonzero(self.valid)
        valid_ind = valid_ind.select(1, 0)

        self.point_clouds = self.point_clouds.index_select(0, valid_ind.long())
        self.volume_length = self.volume_length.index_select(0, valid_ind.long())
        self.gt_xyz = self.gt_xyz.index_select(0, valid_ind.long())
        self.total_frame_num = self.point_clouds.size(0)

        # load PCA coeff
        PCA_data_path = os.path.join(self.root_path, subject_names[self.test_index])
        print("PCA_data_path: " + PCA_data_path)
        PCA_coeff_mat = sio.loadmat(os.path.join(PCA_data_path, 'PCA_coeff.mat'))
        self.PCA_coeff = torch.from_numpy(PCA_coeff_mat['PCA_coeff'][:, 0:self.PCA_SZ].astype(np.float32))
        PCA_mean_mat = sio.loadmat(os.path.join(PCA_data_path, 'PCA_mean_xyz.mat'))
        self.PCA_mean = torch.from_numpy(PCA_mean_mat['PCA_mean_xyz'].astype(np.float32))

        tmp = self.PCA_mean.expand(self.total_frame_num, self.JOINT_NUM * 3)
        tmp_demean = self.gt_xyz - tmp
        self.gt_pca = torch.mm(tmp_demean, self.PCA_coeff)

        self.PCA_coeff = self.PCA_coeff.transpose(0, 1).cuda()
        self.PCA_mean = self.PCA_mean.cuda()

    def __getitem__(self, index):
        return self.point_clouds[index, :, :], self.volume_length[index], self.gt_pca[index, :], self.gt_xyz[index, :]

    def __len__(self):
        return self.point_clouds.size(0)

    def __loaddata(self, data_dir):
        point_cloud = sio.loadmat(os.path.join(data_dir, 'Point_Cloud_FPS.mat'))
        gt_data = sio.loadmat(os.path.join(data_dir, "Volume_GT_XYZ.mat"))
        volume_length = sio.loadmat(os.path.join(data_dir, "Volume_length.mat"))
        valid = sio.loadmat(os.path.join(data_dir, "valid.mat"))

        self.start_index = self.end_index + 1
        self.end_index = self.end_index + len(point_cloud['Point_Cloud_FPS'])

        self.point_clouds[(self.start_index - 1):self.end_index, :, :] = point_cloud['Point_Cloud_FPS'].astype(
            np.float32)
        self.gt_xyz[(self.start_index - 1):self.end_index, :, :] = gt_data['Volume_GT_XYZ'].astype(np.float32)
        self.volume_length[(self.start_index - 1):self.end_index, :] = volume_length['Volume_length'].astype(np.float32)
        self.valid[(self.start_index - 1):self.end_index, :] = valid['valid'].astype(np.float32)

    def __total_frmae_num(self):
        frame_num = 0
        if self.train:  # train
            for i_subject in range(self.SUBJECT_NUM):
                if i_subject != self.test_index:
                    for i_gesture in range(self.GESTURE_NUM):
                        cur_data_dir = os.path.join(self.root_path, subject_names[i_subject], gesture_names[i_gesture])
                        frame_num = frame_num + self.__get_frmae_num(cur_data_dir)
        else:  # test
            for i_gesture in range(self.GESTURE_NUM):
                cur_data_dir = os.path.join(self.root_path, subject_names[self.test_index], gesture_names[i_gesture])
                frame_num = frame_num + self.__get_frmae_num(cur_data_dir)
        return frame_num

    def __get_frmae_num(self, data_dir):
        volume_length = sio.loadmat(os.path.join(data_dir, "Volume_length.mat"))
        return len(volume_length['Volume_length'])