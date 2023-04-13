import os
import time
import numpy as np
import torch
#import torchsummary
import torchvision.transforms as transforms
#from tensorboardX import SummaryWriter

import utils
import data_loader
import models
from options import Options

# CUDA_LAUNCH_BLOCKING=1

# # to continue training - 2022.05.16. revised
# def load_checkpoint(_model, _criterion, _optimizer, _filename='epoch_100.pth.tar'):
#     # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
#     _start_epoch = 0
#     if os.path.isfile(_filename):
#         print("=> loading checkpoint '{}'".format(_filename))
#         checkpoint = torch.load(_filename)
#         _start_epoch = checkpoint['epoch']
#         _model.load_state_dict(checkpoint['model_state_dict'])
#         _criterion.load_state_dict(checkpoint['criterion_state_dict'])
#         _optimizer.load_state_dict(checkpoint['optim_state_dict'])
#         print("=> loaded checkpoint '{}' (epoch {})".format(_filename, checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(_filename))

#     return _model, _criterion, _optimizer, _start_epoch


opts = Options().parse()
torch.cuda.empty_cache()
device = torch.device('cuda:'+str(opts.gpus[0]) if torch.cuda.is_available() else 'cpu')

# ----- Load dataset & Set params -----#
data_path = opts.data_path
seqs_training = opts.train_seq
# data loading & mean-std calculation - 2022.06.28. revised by JIN2
training_data = data_loader.E2EControlPF(data_path, flag_train=True, seqs=seqs_training)

#training_data = data_loader.EreonDataset(data_path, flag_train=True, seqs=seqs_training, num_points=opts.num_points)


# ---- data transform - 2022.05.02. revised
if not opts.model[:5] == 'Lidar':           # revised 2022.06.02
    if opts.augmentation == 'Aug':
        # tr_mean_std_file = os.path.join(data_path, 'tr_img_mean_std_aug.txt')
        # tr_img_mean_std = np.float32(np.loadtxt(tr_mean_std_file))
        #tf = [transforms.Resize(opts.cropsize), transforms.RandomCrop(opts.cropsize)]
        tf = [transforms.Resize((opts.cropsize,opts.cropsize))]
        if opts.color_jitter > 0:
            assert opts.color_jitter <= 1.0
            print('Using ColorJitter data augmentation')
            tf.append(transforms.ColorJitter(brightness=opts.color_jitter, contrast=opts.color_jitter,
                                             saturation=opts.color_jitter, hue=0.5))
        else:
            print('Not Using ColorJitter')
        #tf.append(transforms.Normalize(mean=tr_img_mean_std[0], std=tr_img_mean_std[1]))   # remove np.sqrt 2022.06.28. revised
        training_data.img_transform = transforms.Compose(tf)

    elif opts.augmentation == 'Normal':
        tr_mean_std_file = os.path.join(data_path, 'tr_img_mean_std.txt')
        tr_img_mean_std = np.float32(np.loadtxt(tr_mean_std_file))
        training_data.img_transform = transforms.Compose(
            [transforms.Normalize(mean=tr_img_mean_std[0], std=tr_img_mean_std[1])])   # remove np.sqrt 2022.06.28. revised
# if not opts.model[:6] == 'Camera':          # revised 2022.06.02
#     if opts.model[:5] == 'Lidar':
#         opts.models_dir = opts.models_dir + '_' + str(opts.num_points)
#         utils.mkdirs([opts.models_dir])
#     if opts.augmentation == 'Aug' or opts.augmentation == 'Normal':
#         tr_mean_std_file = os.path.join(data_path, 'tr_lidar_xyz_mean_std_'+str(opts.num_points)+'.txt')
#         tr_lidar_mean_std = np.float32(np.loadtxt(tr_mean_std_file))
#         training_data.lidar_transform = [np.expand_dims(tr_lidar_mean_std[0], axis=1), np.expand_dims(tr_lidar_mean_std[1], axis=1)]

training_loader = torch.utils.data.DataLoader(training_data, batch_size=opts.batch_size, shuffle=True)

# model - 2022.06.16. revised
if opts.model == 'Image_ViT_Pretrain':
    model = models.net_vit_pretrain()
else:    
    model = None
    print('You need to set the model type')
model.to(device)
#if opts.model[:6] == 'Camera' or opts.model[:5] == 'Lidar':
#    torchsummary.summary(model, sz, device='cuda')

#criterion = utils.AtLocCriterion(saq=opts.beta, learn_beta=True)
criterion = utils.E2EControlCriterion(saq=opts.beta, learn_beta=True)
criterion.to(device)

param_list = [{'params': model.parameters()}]


if hasattr(criterion, 'sax') and hasattr(criterion, 'saq'):    
    param_list.append({'params': [criterion.sax, criterion.saq]})
optimizer = torch.optim.Adam(param_list, lr=opts.learning_rate, weight_decay=opts.weight_decay)

# Load pre-trained checkpoint
# model, criterion, optimizer, start_epoch = \
#     load_checkpoint(model, criterion, optimizer, _filename=os.path.join(opts.models_dir, opts.weights))

experiment_name = opts.exp_name
model.train()
train_data_time = utils.AverageMeter()
train_batch_time = utils.AverageMeter()
epoch_time = utils.AverageMeter()
tot_time = utils.AverageMeter()

start_epoch = 0
with open(os.path.join(opts.models_dir, 'Train'+str(start_epoch)+'.txt'), 'w') as f:
    tot_start_time = time.time()
    end_epoch = time.time()
    end = time.time()

    for epoch in range(start_epoch, opts.epoch):
        for i, batch_data in enumerate(training_loader, 0):
            train_data_time.update(time.time() - end)

            optimizer.zero_grad()

            if opts.model[:5] == 'Image':
                training_samps_img, training_samps_depth, training_samps_scan, cmdvels_target = batch_data                
                training_samps_img = training_samps_img.to(device)
                output = model(training_samps_img)
            # elif opts.model[:5] == 'Lidar':          # revised by Jiyong Oh 2022.06.03
            #     training_samps_pd, _, poses_target, _ = batch_data
            #     training_samps_pd = training_samps_pd.to(device)
            #     output = model(training_samps_pd)
            # elif opts.model[:6] == 'Fusion':          # revised by Jiyong Oh 2022.06.02
            #     training_samps_pd, training_samps_img, poses_target, _ = batch_data                
            #     # print('training_samps_pd = ', training_samps_pd.shape)                
            #     # training_samps_pd = torch.concat((training_samps_pd, torch.zeros(opts.batch_size,1,opts.num_points)), dim=1)
            #     training_samps_pd = training_samps_pd.to(device)
            #     if opts.model == 'Fusion_Transformer':                      # added by Jiyong Oh 2023.01.16
            #         training_samps_pd = training_samps_pd.permute(0,2,1).contiguous()                    
            #     training_samps_img = training_samps_img.to(device)
            #     output = model(training_samps_pd, training_samps_img)

            #poses_target = poses_target.to(device)
            cmdvels_target = cmdvels_target.to(device)

            loss = criterion(output, cmdvels_target)
            loss.backward()
            optimizer.step()

            train_batch_time.update(time.time() - end)

            print('Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\t'
                 'Batch time {:.4f} ({:.4f})\tLoss {:f}'
                 .format(experiment_name, epoch+1, i+1, len(training_loader),
                         train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg,
                         loss.item()))

            f.write('Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}\n'
                    .format(epoch + 1, i + 1, len(training_loader),
                            train_data_time.val, train_data_time.avg, train_batch_time.val, train_batch_time.avg,
                            loss.item()))
            end = time.time()


        epoch_time.update(time.time() - end_epoch)
        print('Epoch {:d} checkpoint saved for {:s} (Batch time {:.4f})'.format(epoch + 1, experiment_name, epoch_time.val))
        f.write('Epoch {:d} checkpoint saved for {:s} (Batch time {:.4f})\n'.format(epoch + 1, experiment_name, epoch_time.val))
        end_epoch = time.time()

        if (epoch+1) % opts.save_freq == 0:
            print('\n----------------Save------------\n')
            filename = os.path.join(opts.models_dir, 'epoch_{:03d}.pth.tar'.format(epoch+1))
            checkpoint_dict = {'epoch': epoch+1,
                               'model_state_dict': model.state_dict(),
                               'optim_state_dict': optimizer.state_dict(),
                               'criterion_state_dict': criterion.state_dict()}
            torch.save(checkpoint_dict, filename)

    tot_time.update(time.time() - tot_start_time)
    f.write('{:s} Total training time {:.4f}\n'.format(experiment_name, tot_time.val))    
    print('Finished Training')
