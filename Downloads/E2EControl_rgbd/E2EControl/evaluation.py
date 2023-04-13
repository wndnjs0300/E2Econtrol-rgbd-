import os
import sys
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import time

import utils
import data_loader
import models
from options import Options
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn_evaluation import plot
import scikitplot as skplt

import torch.nn as nn

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')

# DIM = 2   # revised by JIN2. 2022.07.29.
# BY_QUATERNION = False   # revised by JIN2. 2022.08.08.
opts = Options().parse()

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ----- Set params -----#
data_path = opts.data_path

# ********************** Add 2/16
cmd_mean_std_file_name = os.path.join(data_path, 'tr_cmd_mean_std.txt')
mean_cmd, std_cmd = np.float32(np.loadtxt(cmd_mean_std_file_name))
# ********************** Add 2/16

# data transform - 2022.05.02. revised
#img_transform = transforms.Resize((opts.cropsize,opts.cropsize))

# ********************** Add 2/16
img_transform = None
if not opts.model[:5] == 'Lidar':           # revised 2022.06.02
    tr_mean_std_file = os.path.join(data_path, 'tr_img_mean_std_aug.txt')
    tr_img_mean_std = np.float32(np.loadtxt(tr_mean_std_file))
    img_transform = transforms.Compose([
        transforms.Resize((opts.cropsize,opts.cropsize)),
        #transforms.CenterCrop(opts.cropsize),
        transforms.Normalize(mean=tr_img_mean_std[0], std=tr_img_mean_std[1])])
# ********************** Add 2/16


# model
if opts.model == 'Image_ViT_Pretrain':
    model = models.net_vit_pretrain()    
else:
    model = None
    print('You need to set the model type')



model = nn.DataParallel(model, device_ids = [0,1,2,3])
model.to(device)
model.eval()

filename_weights = os.path.join(opts.models_dir, opts.weights)

if os.path.isfile(filename_weights):
    checkpoint = torch.load(filename_weights, map_location=device)
    #print(checkpoint['model_state_dict'])
    utils.load_state_dict(model, checkpoint['model_state_dict'])
    print('Loaded weights from {:s}'.format(filename_weights))
else:
    print('Could not load weights from {:s}'.format(filename_weights))
    sys.exit(-1)


#v_criterion = lambda v_pred, v_gt: np.linalg.norm(v_pred - v_gt)
#w_criterion = lambda w_pred, w_gt: np.linalg.norm(w_pred - w_gt)



#w_criterion = utils.yaw_angular_error

# if BY_QUATERNION:
#     r_criterion = utils.quaternion_angular_error
# else:
#     r_criterion = utils.yaw_angular_error

for file_name, test_seq in zip(['Test_seq_', 'Train_seq_'], [opts.test_seq, opts.train_seq]):
    with open(os.path.join(opts.results_dir, file_name+opts.weights[:-8]+'.csv'), 'w', newline='') as csv_f:
        result_csv = csv.writer(csv_f)
        
        result_csv.writerow(['','0', '1' , '2', '3','4','5','6','7','8','9','10','11','12','13','14'
                            ]) 

        avg_result = []
        for seq_test in sorted(test_seq):
            print(seq_test)
            test_data = data_loader.E2EControlPF(data_path, flag_train=False, seqs=[seq_test])
            
            test_data.img_transform = img_transform
            #test_data.lidar_transform = lidar_transform
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=False)

        

            cmdvel_dim = 2
            cmdvels_prediction = np.zeros((1, cmdvel_dim))
            cmdvels_groundtruth = np.zeros((1, cmdvel_dim))

            with open(os.path.join(opts.models_dir, 'Test_'+seq_test+'_'+opts.weights[:-8]+'.txt'), 'w') as f:
                writer = SummaryWriter(log_dir=opts.runs_dir)
                for idx, (data_img, data_depth, data_ptcld, cmdvel_gt) in enumerate(test_loader):
                    print('data {:d} / {:d}'.format(idx, len(test_loader)))

                    data_img_var = torch.autograd.Variable(data_img, requires_grad=False)
                    #data_img_var = torch.autograd.Variable(data_img, requires_grad=False)
                    cmdvel_gt_var = torch.autograd.Variable(cmdvel_gt, requires_grad=False)

                    if opts.model[:5] == 'Image':
                        data_img_var = data_img_var.to(device)
                        with torch.set_grad_enabled(False):
                            output = model(data_img_var)
                            a=output.cpu()
                            print(len(a[0]))
                            acc=0
                            _acc=0
                            gtarr=[]
                            prediarr=[]
                            for i in range(len(output)):
                                gt=np.argmax(cmdvel_gt[i])
                                predi=np.argmax(a[i])
                                
                                gtarr.append(gt)
                                prediarr.append(predi)
                                #if abs(gt - predi)<=1:
                                if gt == predi:
                                    acc+=1
                                else:
                                    _acc+=1
                                    print('output: ',predi,)
                                    
                                    print('cmdvel:  ',gt,'\n')

                            #ret = multilabel_confusion_matrix(gtarr, prediarr, labels=[0, 1, 2])
                            #print(classification_report(gtarr, prediarr))
                            #tree_cm = plot.ConfusionMatrix.from_raw_data(gtarr, prediarr)  
                            skplt.metrics.plot_confusion_matrix(gtarr, prediarr)

                            plt.show()
                            #plt.close()
                            print("맟힌 개수", acc)
                            print("틀린 개수", _acc)
                            print('accuracy : ', acc/(acc+_acc)*100)
                            
                    print("----------------------------------------------------------------------------------------")
                    
            

        

                '''
                start_time = time.time()
                
                print(time.time() - start_time, 'in sec.')
                for v, w in zip(v_loss, w_loss):
                    writer.add_scalar('v_loss', v)
                    writer.add_scalar('w_loss', w)
                    f.write('v_loss {:.4f}\tw_loss {:.4f}\n' .format(v, w))
                writer.close()
                results = 'Error in linear velocity: median {:3.4f} m/s,  mean {:3.4f} m/s \n' \
                          'Error in angular velocity: median {:3.4f} degree/s, mean {:3.4f} degree/s'\
                    .format(np.median(v_loss), np.mean(v_loss), np.median(w_loss), np.mean(w_loss))
                print(results)

            # show and save results figures
            # fig = plt.figure()
            # real_pose = (poses_prediction[:, :DIM] - mean_pose_xy) / std_pose_xy
            # gt_pose = (poses_groundtruth[:, :DIM] - mean_pose_xy) / std_pose_xy
            # plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
            # plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
            # plt.xlabel('x [m]')
            # plt.ylabel('y [m]')
            # plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
            # plt.text(np.min(gt_pose[:, 1])+0.2, np.max(gt_pose[:, 0])+0.2, results)
            # plt.show()  # block=True)
            # image_filename = os.path.join(os.path.expanduser(opts.results_dir), '{:s}.png'.format(opts.model+'_'+seq_test))
            # fig.savefig(image_filename)

            # save results into excel file
            result_csv.writerow([seq_test, np.median(v_loss), np.mean(v_loss), np.median(w_loss), np.mean(w_loss)])
            avg_result.append([np.median(v_loss), np.mean(v_loss), np.median(w_loss), np.mean(w_loss)])
        avg_result = np.mean(np.array(avg_result), axis=0)
        result_csv.writerow(['average', avg_result[0], avg_result[1], avg_result[2], avg_result[3]])
'''