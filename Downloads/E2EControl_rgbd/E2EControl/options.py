import argparse
import os
import utils as utils
import torch


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        # base options
        self.parser.add_argument('--data_dir', type=str, default='data')
        self.parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
        # self.parser.add_argument('--num_points', type=int, default=850, help='number of points in an input point cloud')
        self.parser.add_argument('--cropsize', type=int, default=224)
        #self.parser.add_argument('--cropsize', type=int, default=256)
        # self.parser.add_argument('--print_freq', type=int, default=20)
        self.parser.add_argument('--gpus', type=str, default='0', help='specify gpu device')
        # self.parser.add_argument('--nThreads', default=8, type=int, help='threads for loading data')
        self.parser.add_argument('--dataset', type=str, default='E2EControlFP')
        #self.parser.add_argument('--scene', type=str, default='full_0708')
        # self.parser.add_argument('--scene', type=str, default='part4')
        self.parser.add_argument('--model', type=str, default='Image_ViT')
        # self.parser.add_argument('--seed', type=int, default=7)
        # self.parser.add_argument('--lstm', type=bool, default=False)
        self.parser.add_argument('--logdir', type=str, default='./logs')
        # self.parser.add_argument('--exp_name', type=str, default='name')
        # self.parser.add_argument('--skip', type=int, default=10)
        # self.parser.add_argument('--variable_skip', type=bool, default=False)
        # self.parser.add_argument('--real', type=bool, default=False)
        # self.parser.add_argument('--steps', type=int, default=3)
        # self.parser.add_argument('--val', type=bool, default=False)
        self.parser.add_argument('--full_seq', type=str,  default={ 'seq-01', 'seq-02', 'seq-03','seq-04', 'seq-05','seq-06', 'seq-07', 'seq-08', 'seq-09', 'seq-10',
                                                'seq-11', 'seq-12',  'seq-13',  })
                                 #default={'seq-11', 'seq-12', 'seq-14', 'seq-15', 'seq-01', 'seq-02', 'seq-04', 'seq-05', 'seq-07', 'seq-08', 'seq-10',
                                 #         'seq-13', 'seq-03', 'seq-06', 'seq-09'})

        # train options
        self.parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
        self.parser.add_argument('--beta', type=float, default=0.0)
        # self.parser.add_argument('--beta', type=float, default=-3.0)
        # self.parser.add_argument('--gamma', type=float, default=None, help='only for AtLoc+ (-3.0)')
        self.parser.add_argument('--augmentation', type=str, default='Aug')
        self.parser.add_argument('--color_jitter', type=float, default=0.7,
                                  help='0.7 is only for RobotCar, 0.0 for 7Scenes')
        # self.parser.add_argument('--train_dropout', type=float, default=0.5)
        # self.parser.add_argument('--val_freq', type=int, default=5)
        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--runs_dir', type=str, default='runs')
        # self.parser.add_argument('--lr', type=float, default=5e-5)
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001)
        
        #self.parser.add_argument('--train_seq', type=str, default={'seq-01','seq-02'})
        self.parser.add_argument('--train_seq', type=str, default={'seq-01', 'seq-02', 'seq-03', 'seq-04', 'seq-05','seq-06',  'seq-07','seq-08', 'seq-09', 'seq-10',
                                                'seq-11', 'seq-12'})
                                 #default={'seq-01', 'seq-02', 'seq-04'})
        #self.parser.add_argument('--train_seq', type=str, default={'seq-01', 'seq-02'})
                                 #default={'seq-01', 'seq-02'}
                                 #default={'seq-01', 'seq-02', 'seq-04', 'seq-05', 'seq-07', 'seq-08', 'seq-10'}
        # test options
        #self.parser.add_argument('--test_dropout', type=float, default=0.0)
        self.parser.add_argument('--weights', type=str, default='epoch_%03d.pth.tar' % 300)
        self.parser.add_argument('--save_freq', type=int, default=300)
        self.parser.add_argument('--test_seq', type=str, #default={'seq-03'})
                                 #default={'seq-04', 'seq-07'})
                                 default={'seq-13'})

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        str_ids = self.opt.gpus.split(',')
        self.opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpus.append(id)

        # set gpu ids
        if len(self.opt.gpus) > 0:
            torch.cuda.set_device(self.opt.gpus[0])

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')

        # save to the disk
        self.opt.data_path = os.path.join(self.opt.data_dir, self.opt.dataset)
        self.opt.exp_name = '{:s}_{:s}'.format(self.opt.dataset, self.opt.model)
        #expr_dir = os.path.join(self.opt.scene+'_3DoF', 'Fusion_Transformer', self.opt.exp_name)   # 2022.07.12. revised by JIN2
        #expr_dir = os.path.join(self.opt.scene+'_3DoF', 'ResNet+Transformer', self.opt.exp_name)   # 2022.07.12. revised by JIN2
        #expr_dir = os.path.join(self.opt.scene+'_3DoF', 'Img_Drop', self.opt.exp_name)
        #expr_dir = os.path.join('Image_ViT_Pretrain', self.opt.exp_name)
        expr_dir = os.path.join('Image_ViT', self.opt.exp_name)
        results_path = [expr_dir, self.opt.results_dir, self.opt.augmentation, 'lr_'+str(self.opt.learning_rate)]
        self.opt.results_dir = os.path.join(*results_path)
        models_path = [expr_dir, self.opt.models_dir, self.opt.augmentation, 'lr_'+str(self.opt.learning_rate)]
        self.opt.models_dir = os.path.join(*models_path)
        runs_path = [expr_dir, self.opt.runs_dir, self.opt.augmentation, 'lr_' + str(self.opt.learning_rate)]
        self.opt.runs_dir = os.path.join(*runs_path)
        # self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        # self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        #utils.mkdirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])
        utils.mkdirs([expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])
        return self.opt
