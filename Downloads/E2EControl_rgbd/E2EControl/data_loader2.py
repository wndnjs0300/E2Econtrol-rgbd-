import cv2 
import os
import glob
#from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
import numpy as np
import PIL
import utils as utils

import gc
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
DIM = 2
'''
class EreonDataset(Dataset):
    def __init__(self, data_dir, flag_train, seqs, num_points,
                 img_transform=None, lidar_transform=None, random_order=True,
                 by_quaternion=False):   # 2022.08.08. revised by JIN2
        self.dir = data_dir
        self.flag_train = flag_train
        self.seqs = seqs
        self.num_points = num_points
        self.img_transform = img_transform
        self.lidar_transform = lidar_transform
        self.random_order = random_order   # 2022.07.11. revised by JIN2
        self.by_quaternion = by_quaternion   # 2022.08.08. revised by JIN2
        print('Loading...')
        # ------ 2022.06.28. revised by JIN2: start ------ #
        self.ptclds, self.images, self.poses = [], [], []
        self.mean_pose_xy, self.std_pose_xy = 0, 0
        self.roll_pitch = []

        self.load_dataset()
        self.ptclds = torch.Tensor(np.array(self.ptclds))
        self.images = torch.stack(self.images)
        self.poses = torch.Tensor(np.array(self.poses))
        self.save_load_mean_std_files()
        self.poses[:, :DIM] -= self.mean_pose_xy
        self.poses[:, :DIM] /= self.std_pose_xy

    def load_dataset(self):
        tf = transforms.ToTensor()
        for folder_name in self.seqs:
            print(folder_name)
            target = self.dir + '/' + folder_name + '/*.bin'
            for file_name in glob.glob(target):
                # Lidar - 2022.06.28. revised by JIN2
                pt_temp = load_velodyne_binary(file_name)
                pt_temp = self.select_lidar_random(pt_temp)
                self.ptclds.append(pt_temp)
                # Remove all zero columns - 2022.06.20. revised by JIN2
                # self.ptclds.append(pt_temp[:, ~np.all(pt_temp == 0, axis=0)])

                # Image
                file_name_img = file_name[0:-4] + '.jpg'
                img = PIL.Image.open(file_name_img)
                img_t = tf(img)
                self.images.append(img_t)

                # Pose
                file_name_pose = file_name[0:-4] + '.txt'
                pose_temp = np.loadtxt(file_name_pose, delimiter=',')
                angle_quat = np.array([pose_temp[-1]] + pose_temp[3:-1].tolist())
                if self.by_quaternion:
                    angle_rpy = utils.q2rpy(angle_quat)
                    angle_yaw = angle_rpy[DIM:]
                    self.roll_pitch.append(angle_rpy[:DIM])
                else:
                    angle_yaw = utils.q2yaw(angle_quat)
                pose_t = np.concatenate((pose_temp[0:2], angle_yaw))
                self.poses.append(pose_t)


    def select_lidar_random(self, _data):   # 2022.07.11. revised by JIN2
        num_pts_total = _data.shape[-1]
        if self.random_order:
            order = np.random.permutation(num_pts_total)
        else:
            np.random.seed(100)
            order = np.random.permutation(num_pts_total)
        _data = _data[:3, order[:self.num_points]]           ## 2 or 3
        return _data

    def cal_mean_std(self, _data, _divisor, dim=3):
        _data = _data.numpy()
        total_sum = np.sum(_data, axis=0)
        total_sqsum = np.sum(_data ** 2, axis=0)
        mean_p = np.asarray([np.sum(total_sum[c]) for c in range(dim)])
        mean_p /= _divisor
        # var = E[x^2] - E[x]^2
        var_p = np.asarray([np.sum(total_sqsum[c]) for c in range(dim)])
        var_p /= _divisor
        var_p -= (mean_p ** 2)
        return mean_p, np.sqrt(var_p)

    def save_load_mean_std_files(self):
        pose_mean_std_file_name = os.path.join(self.dir, 'tr_pose_xy_mean_std.txt')
        lidar_mean_std_file_name = os.path.join(self.dir, 'tr_lidar_xyz_mean_std_' + str(self.num_points) + '.txt')     ## xy or xyz
        image_mean_std_file_name = os.path.join(self.dir, 'tr_img_mean_std_aug.txt')
        if self.flag_train:
            if not os.path.isfile(pose_mean_std_file_name):
                self.mean_pose_xy, self.std_pose_xy = self.cal_mean_std(self.poses, len(self.poses), dim=DIM)
                np.savetxt(pose_mean_std_file_name, np.vstack((self.mean_pose_xy, self.std_pose_xy)), fmt='%8.7f')
            else:
                self.mean_pose_xy, self.std_pose_xy = np.float32(np.loadtxt(pose_mean_std_file_name))
            if not os.path.isfile(lidar_mean_std_file_name):
                mean_lidar, std_lidar = self.cal_mean_std(self.ptclds, len(self.ptclds) * self.ptclds.shape[-1], dim=3) ## DIM
                np.savetxt(lidar_mean_std_file_name,
                           np.vstack((mean_lidar, std_lidar)), fmt='%8.7f')
            if not os.path.isfile(image_mean_std_file_name):
                mean_image, std_image = self.cal_mean_std(self.images,
                                                          len(self.images) * self.images.shape[-2] * self.images.shape[-1])
                np.savetxt(image_mean_std_file_name,
                           np.vstack((mean_image, std_image)), fmt='%8.7f')
        else:  # 2022.05.02. revised -> not used problem, 2022.06.27. revised
            self.mean_pose_xy, self.std_pose_xy = np.float32(np.loadtxt(pose_mean_std_file_name))
    # ------ 2022.06.28. revised by JIN2: end ------ #

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # 2022.05.02. & 2022.05.10. revised
        ptcld, image, pose = self.ptclds[idx], self.images[idx], self.poses[idx]

        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.lidar_transform is not None:
            ptcld -= self.lidar_transform[0]
            ptcld[0:2] /= self.lidar_transform[1][0:2]

        if self.roll_pitch:
            return ptcld, image, pose, self.roll_pitch[idx]
        else:
            return ptcld, image, pose, self.roll_pitch

#------------------------------------------------------------------------------#
'''
class E2EControlPF(Dataset):
    def __init__(self, data_path, flag_train, seqs, img_transform=None):
        
        #self.angles = (np.arange(0,365,0.2842)) * np.pi / 180
        self.angles = (np.linspace(0,np.pi,1285))  
        # print((self.angles) ) 
        self.dir = data_path
        self.seqs = seqs  
        #print(100)
        print(self.seqs) 
        self.img_transform = img_transform

        self.scans_xy_total = []        
        self.images = []
        self.depths = []
        self.cmd = []
    
        self.flag_train = flag_train        

        self.load_dataset()
        self.scans_xy_total = torch.Tensor(np.array(self.scans_xy_total))
        self.cmd = torch.Tensor(np.array(self.cmd))
        self.images = torch.stack(self.images)
        self.depths = torch.stack(self.depths)

        # ********************** Add 2/16
        self.mean_cmd, self.std_cmd = 0, 0
        self.save_load_mean_std_files()
        # ********************** Add 2/16

    def load_scan_file(self, file_name):        
        data = np.genfromtxt(file_name, delimiter=',')        
        scans = data.astype(np.float32)
        return scans

    def polar2catersian(self, angles, ranges):  
        np.nan_to_num(ranges, copy=False) # need to validation
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        xy = np.vstack((x, y))
        return x, y, xy

    def load_dataset(self):
        tf = transforms.ToTensor()
        for seq in self.seqs:
            print(seq)
            #file_list = os.listdir(self.dir)
            
            file_list = os.listdir(os.path.join(self.dir, seq))  
            print(file_list)          
            for folder_name in file_list:
                

                data_dir = os.path.join(self.dir, seq, folder_name)
                
                if folder_name=='lidarsave': #lidar
                    file_names = glob.glob(os.path.join(data_dir, '*.csv'))
                    scans = self.load_scan_file(file_names[0])
                    for idx_scan in range(scans.shape[0]):
                        ranges = scans[idx_scan, :]   
                                
                        x, y, xy = self.polar2catersian(self.angles, ranges)
                        self.scans_xy_total.append(xy)

                elif folder_name=='cmdsave':
                    #file_names = sorted(glob.glob(data_dir + '/*.csv'))
                    file_names = sorted(glob.glob(os.path.join(data_dir, '*.csv')))                    
                    
                    data = np.genfromtxt(file_names[0], delimiter=',')
                    #print(data)
                    
                    cmddata=data.astype(np.float32)
                 
                    for i in range(len(cmddata)):
                        self.cmd.append(cmddata[i])
                    print(self.cmd[200])
                  
                        

                    

                elif folder_name=='imgsave':
                    #file_names = sorted(glob.glob(data_dir + '/*.jpg'))
                    file_names = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
                    for i in range(len(file_names)):
                        img = PIL.Image.open(file_names[i])
                        img_t = tf(img)
                        self.images.append(img_t)
                  


                elif folder_name=='depthsave':
                    #file_names = sorted(glob.glob(data_dir + '/*.png'))
                    file_names = sorted(glob.glob(os.path.join(data_dir, '*.png')))
                    #print(file_names)
                    for i in range(len(file_names)):
                        img = PIL.Image.open(file_names[i])
                        img_t = tf(img)
                        self.depths.append(img_t)
                        #print(2)
               
     # ********************** Add 2/16         
    def cal_mean_std(self, _data, _divisor, dim=3):
        _data = _data.numpy()
        total_sum = np.sum(_data, axis=0)
        total_sqsum = np.sum(_data ** 2, axis=0)
        mean_p = np.asarray([np.sum(total_sum[c]) for c in range(dim)])
        mean_p /= _divisor
        # var = E[x^2] - E[x]^2
        var_p = np.asarray([np.sum(total_sqsum[c]) for c in range(dim)])
        var_p /= _divisor
        var_p -= (mean_p ** 2)
        return mean_p, np.sqrt(var_p)

    def save_load_mean_std_files(self):
        image_mean_std_file_name = os.path.join(self.dir, 'tr_img_mean_std_aug.txt')
        cmd_mean_std_file_name = os.path.join(self.dir, 'tr_cmd_mean_std.txt')
        
        if self.flag_train:
            # image
            if not os.path.isfile(image_mean_std_file_name):
                mean_image, std_image = self.cal_mean_std(self.images, len(self.images) * self.images.shape[-2] * self.images.shape[-1])
                np.savetxt(image_mean_std_file_name, np.vstack((mean_image, std_image)), fmt='%8.7f')
            # cmd vel
            if not os.path.isfile(cmd_mean_std_file_name):
                self.mean_cmd, self.std_cmd = self.cal_mean_std(self.cmd, len(self.cmd), dim=2)
                np.savetxt(cmd_mean_std_file_name, np.vstack((self.mean_cmd, self.std_cmd)), fmt='%8.7f')
            else:
                self.mean_cmd, self.std_cmd = np.float32(np.loadtxt(cmd_mean_std_file_name))    
        else:
            self.mean_cmd, self.std_cmd = np.float32(np.loadtxt(cmd_mean_std_file_name))
    # ********************** Add 2/16
   
    
    def __len__(self):
        
        return len(self.scans_xy_total)
    
    def __getitem__(self, idx):   
        img = self.images[idx]
        depth = self.depths[idx]   
        scan = self.scans_xy_total[idx]
        cmdvel =self.cmd[idx]

        # ********************** Add 2/16
        cmdvel -= self.mean_cmd
        cmdvel /= self.std_cmd
        # ********************** Add 2/16

        if self.img_transform is not None:
            img = self.img_transform(img)
        
        return img, depth, scan, cmdvel


if __name__ == '__main__':
    #gc.collect()
    #torch.cuda.empty_cache()
    #data_path = 'DROWv2-data'
    data_path = '/data'    
    dataset_drow = E2EControlPF(dir=data_path, flag_train='train')
    #img, depth, ptcld, cmdvel = dataset_drow[50]
    #print(100)
    
    #print(ptcld)
    #print(500)
    #print(cmdvel)
    #print(5300)

    
    #plt.figure('input')
    #plt.imshow(img.permute(1, 2, 0))
    #plt.show()

    #img_pil = Image.open(img)
    #img_tensor = F.to_tensor(img_pil)
    
    #for folder_name in glob.glob("/home/jw/DATA/data01/lidarsave/*.csv"):
    # training_loader = torch.utils.data.DataLoader(dataset_drow, batch_size=opts.batch_size, shuffle=True)
    #    print(folder_name)
