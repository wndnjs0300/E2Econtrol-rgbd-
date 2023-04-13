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

import torch.optim as optim

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F


writer = SummaryWriter('runs/rgbdexperiment1')

opts = Options().parse()
torch.cuda.empty_cache()
device = torch.device('cuda:'+str(opts.gpus[0]) if torch.cuda.is_available() else 'cpu')

# ----- Load dataset & Set params -----#
data_path = opts.data_path
seqs_training = opts.train_seq
# data loading & mean-std calculation - 2022.06.28. revised by JIN2
training_data = data_loader.E2EControlPF(data_path, flag_train=True, seqs=seqs_training)


# ---- data transform - 2022.05.02. revised
if not opts.model[:5] == 'Lidar':           # revised 2022.06.02
    #-------------------------------------------- Add 2/16
    tr_mean_std_file = os.path.join(data_path, 'tr_img_mean_std_aug.txt')
    tr_img_mean_std = np.float32(np.loadtxt(tr_mean_std_file))
    #-------------------------------------------- Add 2/16
    #tf = [transforms.Resize(opts.cropsize), transforms.RandomCrop(opts.cropsize)]
    tf = [transforms.Resize((opts.cropsize,opts.cropsize))]
    if opts.color_jitter > 0:
        assert opts.color_jitter <= 1.0
        print('Using ColorJitter data augmentation')
        tf.append(transforms.ColorJitter(brightness=opts.color_jitter, contrast=opts.color_jitter,
                                            saturation=opts.color_jitter, hue=0.5))
    else:
        print('Not Using ColorJitter')
    #-------------------------------------------- Add 2/16
    tf.append(transforms.Normalize(mean=tr_img_mean_std[0], std=tr_img_mean_std[1]))   # remove np.sqrt 2022.06.28. revised
    training_data.img_transform = transforms.Compose(tf)
    #-------------------------------------------- Add 2/16



training_loader = torch.utils.data.DataLoader(training_data, batch_size=opts.batch_size, shuffle=True)

# model - 2022.06.16. revised
if opts.model == 'Image_ViT_Pretrain':
    model = models.net_vit_pretrain()
elif opts.model == 'Image_ViT':
    model = models.net_vit_from_scratch()
else:    
    model = None
    print('You need to set the model type')

model = nn.DataParallel(model, device_ids = [0,1,2,3])
model.to(device)

#criterion = utils.E2EControlCriterion(saq=opts.beta, learn_beta=True)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

param_list = [{'params': model.parameters()}]




if hasattr(criterion, 'sax') and hasattr(criterion, 'saq'):    
    param_list.append({'params': [criterion.sax, criterion.saq]})
optimizer = torch.optim.Adam(param_list, lr=opts.learning_rate, weight_decay=opts.weight_decay)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
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
with open(os.path.join(opts.models_dir, 'Train'+str(opts.epoch)+'.txt'), 'w') as f:
    tot_start_time = time.time()
    end_epoch = time.time()
    end = time.time()

    for epoch in range(start_epoch, opts.epoch):
        for i, batch_data in enumerate(training_loader, 0):
            train_data_time.update(time.time() - end)

            optimizer.zero_grad()

            if opts.model[:5] == 'Image':
                training_samps_img, training_samps_depth, training_samps_scan, cmdvels_target = batch_data                
                
                #print(111111111111111111111111111111111111111)
                #print(training_samps_img.type())
                #print(training_samps_depth.size())
                #a=training_samps_img.numpy()
                #b=training_samps_depth.numpy()
                #a=np.stack(training_samps_img.numpy(),training_samps_depth.numpy(),axis=1)
                #c=np.stack(np.vectorize(a),np.vectorize(b))
                rgbd=torch.cat((training_samps_img,training_samps_depth),dim=1)
                #print(rgbd.size())
                rgbd = rgbd.to(device)
                output = model(rgbd)
           

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
        writer.add_scalar("Loss/train", loss, epoch)


        scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])

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
    writer.close()
    print('Finished Training')
