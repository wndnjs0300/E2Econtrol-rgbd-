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
import csv
DIM = 2

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
        self.mean_depth, self.std_depth = 0, 0
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
                    print(file_names)
                    #data = np.genfromtxt(file_names[0], delimiter=',')
                    #print(data)
                    f=open(file_names[0])     
                    cmddata=csv.reader(f)
                   
                    sum_ang=[]
                    sum_spd=[]
                    #cmddata=data.astype(np.float32)
                 
                    #for i in range(len(cmddata)):
                    
                    #    self.cmd.append(cmddata[i])
                    for row in cmddata:
                        i=0
                        
                        angle=float(row[1].split(',')[0])
                        #speed=float(row[0].split(',')[0])
                        #print(speed)
                        #print(angle)
                        if angle>-0.000002 and angle<0.000002:

                            sum_ang.append(0)
                        elif angle<=-0.000002 and angle>-0.00001:
                            sum_ang.append(1)

                        elif angle<=-0.00001:
                            sum_ang.append(2)
                        elif angle>=0.000002 and angle<0.00001:
                            sum_ang.append(3)

                        elif angle>=0.00001:
                            sum_ang.append(4)
                    f.close()
                    
                    f=open(file_names[0])     
                    cmddata=csv.reader(f)    
                    for row2 in cmddata:
                        speed=float(row2[0].split(',')[0])
                        if speed<0.015:
                            sum_spd.append(0)
    
                        elif speed>=0.015 and speed<0.023:
                            sum_spd.append(1)
       
                        elif speed>=0.023:
                            sum_spd.append(2)
        
                      
                    f.close()
                    print(len(sum_ang))
                    print(len(sum_spd))
                    a=[]
                    b=[]
                  #[ 1.2877e-02  3.0000e-06]
                    '''
                    for i in range(len(sum_ang)):


                        if sum_ang[i]==0:
                            a.append([1, 0, 0, 0, 0])
                        elif sum_ang[i]==1:
                            a.append([0, 1, 0, 0, 0])
                        elif sum_ang[i]==2:
                            a.append([0, 0, 1, 0, 0])
                        elif sum_ang[i]==3:
                            a.append([0, 0, 0, 1, 0])
                        elif sum_ang[i]==4:
                            a.append([0, 0, 0, 0, 1])   
                    
                    for i in range(len(sum_ang)):


                        if sum_spd[i]==0:
                            b.append([1, 0, 0])
                        elif sum_spd[i]==1:
                            b.append([0, 1, 0])
                        elif sum_spd[i]==2:
                            b.append([0, 0, 1])
                        
                    data = np.array(b)
                    '''
                    for i in range(len(sum_ang)):
                        if sum_spd[i]==0:
                            if sum_ang[i]==0:
                                a.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==4:
                                a.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    

                        elif sum_spd[i]==1:
                            if sum_ang[i]==0:
                                a.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==4:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])   

                        elif sum_spd[i]==2:
                            if sum_ang[i]==0:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                            elif sum_ang[i]==4:
                        
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) 
                    
                    data = np.array(a) 
                    for j in range(len(sum_spd)): 
                           
                        self.cmd.append(data[j])
                        #self.cmd.append(sum_spd[i])
                    print(self.cmd[5])    
                        
                    

                elif folder_name=='imgsave':
                    #file_names = sorted(glob.glob(data_dir + '/*.jpg'))
                    file_names = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
                    for i in range(len(file_names)):
                        img = PIL.Image.open(file_names[i])
                        
                        img_t = tf(img)
                        self.images.append(img_t)
                        #print(img_t)
                  


                elif folder_name=='depthsave':
                    #file_names = sorted(glob.glob(data_dir + '/*.png'))
                    file_names = sorted(glob.glob(os.path.join(data_dir, '*.png')))
                    #print(file_names)
                    for i in range(len(file_names)):
                        img = PIL.Image.open(file_names[i])
                        img2=img.resize((224,224))
                        img3=np.array(img2)
                        img3[img3>10000] = 10000
                        img3=img3*255/10000
                        img4 = Image.fromarray(img3)
                        #img5=img4*255/10000
                        #print(img4)
                        img_t = tf(img4)
                        self.depths.append(img_t)
                        #print(2)
                        #print(img_t)
               
     # ********************** Add 2/16         
    def cal_mean_std(self, _data, _divisor, dim=3):
        _data = _data.numpy()
        total_sum = np.sum(_data, axis=0)
        total_sqsum = np.sum(_data ** 2, axis=0)
        mean_p = np.asarray([np.sum(total_sum[c]) for c in range(dim)])
        mean_p = mean_p / _divisor
        # var = E[x^2] - E[x]^2
        var_p = np.asarray([np.sum(total_sqsum[c]) for c in range(dim)])
        var_p = var_p / _divisor
        var_p = var_p - (mean_p ** 2)
        return mean_p, np.sqrt(var_p)

    def save_load_mean_std_files(self):
        image_mean_std_file_name = os.path.join(self.dir, 'tr_img_mean_std_aug.txt')
        cmd_mean_std_file_name = os.path.join(self.dir, 'tr_cmd_mean_std.txt')
        depth_mean_std_file_name = os.path.join(self.dir, 'tr_depth_mean_std.txt')
        
        if self.flag_train:
            # image
            if not os.path.isfile(image_mean_std_file_name):
                mean_image, std_image = self.cal_mean_std(self.images, len(self.images) * self.images.shape[-2] * self.images.shape[-1])
                np.savetxt(image_mean_std_file_name, np.vstack((mean_image, std_image)), fmt='%8.7f')

            #depth
            if not os.path.isfile(depth_mean_std_file_name):
                self.mean_depth, self.std_depth = self.cal_mean_std(self.depths, len(self.depths)* self.depths.shape[-2] * self.depths.shape[-1], dim=1)
                print('mdepth',self.mean_depth)
                print('sdepth',self.std_depth)
                np.savetxt(depth_mean_std_file_name, np.vstack((self.mean_depth, self.std_depth)), fmt='%8.7f')
                
            else:
                self.mean_depth, self.std_depth = np.float32(np.loadtxt(depth_mean_std_file_name))

            # cmd vel
            if not os.path.isfile(cmd_mean_std_file_name):
                self.mean_cmd, self.std_cmd = self.cal_mean_std(self.cmd, len(self.cmd), dim=2)
                np.savetxt(cmd_mean_std_file_name, np.vstack((self.mean_cmd, self.std_cmd)), fmt='%8.7f')
            else:
                self.mean_cmd, self.std_cmd = np.float32(np.loadtxt(cmd_mean_std_file_name))    
        else:
            self.mean_cmd, self.std_cmd = np.float32(np.loadtxt(cmd_mean_std_file_name))
            self.mean_depth, self.std_depth = np.float32(np.loadtxt(depth_mean_std_file_name))
    # ********************** Add 2/16
   
    
    def __len__(self):
        
        return len(self.scans_xy_total)
    
    def __getitem__(self, idx):   
        img = self.images[idx]
        depth = self.depths[idx]   
        scan = self.scans_xy_total[idx]
        cmdvel =self.cmd[idx]

        # ********************** Add 2/16
        #cmdvel -= self.mean_cmd
        #cmdvel /= self.std_cmd
        # ********************** Add 2/16

        # ********************** Add 2/16
        #depth=depth.float()
        
        depth-=self.mean_depth
        depth /= self.std_depth
        #depth=depth*255/10000
        #print(depth)
        # ********************** Add 2/16

        if self.img_transform is not None:
            img = self.img_transform(img)
            #print(img)
            #a= torch.stack(img, depth)
            #print(a.shape())
        
        return img, depth, scan, cmdvel


if __name__ == '__main__':
    #gc.collect()
    #torch.cuda.empty_cache()
    #data_path = 'DROWv2-data'
    # ****************************** 원래 값
    data_path = '/data'    
    
    dataset_drow = E2EControlPF(dir=data_path, flag_train='train')
    img, depth, ptcld, cmdvel = dataset_drow[50]
    print(100)
    
    #print(ptcld)
    #print(500)
    print(cmdvel)
    #print(5300)

    
    #plt.figure('input')
    #plt.imshow(img.permute(1, 2, 0))
    #plt.show()

    #img_pil = Image.open(img)
    #img_tensor = F.to_tensor(img_pil)
    
    #for folder_name in glob.glob("/home/jw/DATA/data01/lidarsave/*.csv"):
    # training_loader = torch.utils.data.DataLoader(dataset_drow, batch_size=opts.batch_size, shuffle=True)
    #    print(folder_name)
    '''
                    for i in range(len(sum_ang)):
                        if sum_spd[i]==0:
                            if sum_ang[i]==0:
                                a.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==4:
                                a.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])    

                        elif sum_spd[i]==1:
                            if sum_ang[i]==0:
                                a.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                            elif sum_ang[i]==4:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])   

                        elif sum_spd[i]==2:
                            if sum_ang[i]==0:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                            elif sum_ang[i]==1:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                            elif sum_ang[i]==2:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                            elif sum_ang[i]==3:
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                            elif sum_ang[i]==4:
                        
                                a.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])       
                    '''
