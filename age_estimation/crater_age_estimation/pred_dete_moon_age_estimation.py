
import argparse
import os
import shutil
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import moon_net
import pdb
import bisect

from loader_moon_two import MoonData
import math
from math import ceil
import torch.nn.functional as F
from methods_two import train_sup, train_mt, validate


parser = argparse.ArgumentParser(description='PyTorch Moon Age Estimation Model Predicting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: '+ ' (default: resnet50)')
parser.add_argument('--model', '-m', metavar='MODEL', default='mt',
                    help='model: '+' (default: baseline)', choices=['baseline', 'mt'])
parser.add_argument('--ntrial', default=5, type=int, help='number of trial')
parser.add_argument('--optim', '-o', metavar='OPTIM', default='adam',
                    help='optimizer: '+' (default: adam)', choices=['adam'])
parser.add_argument('--dataset', '-d', metavar='DATASET', default='moon_two',
                    help='dataset: '+' (default: moon_two)')
parser.add_argument('--aug', action='store_true', default=False, help='control data aug or not')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 225)')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weight_l1', '--l1', default=1e-3, type=float,
                    metavar='W1', help='l1 regularization (default: 1e-3)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',default=5, type=int, help='number of classes in the model')
parser.add_argument('--attr_filters',default=(78, 1024, 4096))
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
parser.add_argument('--gpu',default=0, type=str, help='cuda_visible_devices')
parser.add_argument('--is_inception', help='is or not inception struction',
                    default=False, type=bool)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


def main(trial=1):
    global args, best_prec1, best_test_prec1
    global acc1_tr, losses_tr 
    global losses_cl_tr
    global acc1_val, losses_val, losses_et_val
    global acc1_test, losses_test, losses_et_test
    global weights_cl
    args = parser.parse_args()
    print(args)
    
    attr_range = (0,78)
    # Data loading code
    # test data
    df_test = pd.read_csv('input/imgs_gray_{}_dete/filelist_{}.csv'.format(args.datatype, 256))

    test_img_path = 'input/imgs_gray_{}_dete/train_{}/'.format(args.datatype, 256)
    test_attr_path = 'input/attr_{}_dete/'.format(args.datatype, 256)
    if args.aug:
        pred_img_path = 'result_CV_{}_dete/pred_error_test_img_by_{}_{}_trial{}_aug/'.format(args.datatype, args.arch, args.model, trial)
    else:
        pred_img_path = 'result_CV_CE1/pred_error_test_img_by_{}_{}_trial{}/'.format(args.arch, args.model, trial)
    if not os.path.exists(pred_img_path):
        os.makedirs(pred_img_path)
    
    img_path_test = []
    attr_path_test = []
    y_test = []

    for f, tags in tqdm(df_test.values, miniters=100):
        img_path = test_img_path + '{}.jpg'.format(f)
        img_path_test.append(img_path)
        attr_path = test_attr_path + '{}.csv'.format(f)
        attr_path_test.append(attr_path)
        y_test.append(tags)

    #img_path_test, attr_path_test, y_test = shuffle(img_path_test, attr_path_test, y_test, random_state = 24)
    
    print('Testing on {} samples\n'.format(len(img_path_test)))

    testset = MoonData(img_path_test, attr_path_test, labels=y_test, attr_range=attr_range)
    test_loader = data.DataLoader(testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)
        
    
    # create model
    if args.arch == 'resnet50':
        print("Model: %s" %args.arch)
        model = moon_net.resnet_50_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'resnet101':
        print("Model: %s" %args.arch)
        model = moon_net.resnet_101_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'resnet152':
        print("Model: %s" %args.arch)
        model = moon_net.resnet_152_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'densenet201':
        print("Model: %s" %args.arch)
        model = moon_net.densenet_201_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'senet':
        print("Model: %s" %args.arch)
        model = moon_net.senet_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'se_resnet152':
        print("Model: %s" %args.arch)
        model = moon_net.se_resnet152_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'se_resnet101':
        print("Model: %s" %args.arch)
        model = moon_net.se_resnet101_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'se_resnet50':
        print("Model: %s" %args.arch)
        model = moon_net.se_resnet50_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'se_resnext101':
        print("Model: %s" %args.arch)
        model = moon_net.se_resnext101_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'polynet':
        print("Model: %s" %args.arch)
        model = moon_net.polynet_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'inceptionv3':
        print("Model: %s" %args.arch)
        args.is_inception = False
        model = moon_net.inceptionv3_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    elif args.arch == 'dpn68b':
        print("Model: %s" %args.arch)
        model = moon_net.dpn68b_two(num_classes=args.num_classes, attr_filters=args.attr_filters)
    else:
        assert(False)
    
    if args.model == 'mt':
        import copy  
        model_teacher = copy.deepcopy(model)
        model_teacher = torch.nn.DataParallel(model_teacher).cuda()

    model = torch.nn.DataParallel(model).cuda()
    #print(model)
    
    if args.aug:
        ckpt_dir = args.ckpt+'_'+args.dataset+'_'+args.arch+'_'+args.model+'_aug'+'_'+args.optim
    else:
        ckpt_dir = args.ckpt+'_'+args.dataset+'_'+args.arch+'_'+args.model+'_'+args.optim
    ckpt_dir = ckpt_dir + '_e%d'%(args.epochs)
    cudnn.benchmark = True
    
    # deifine loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()    
    criterion_l1 = nn.L1Loss(size_average=False).cuda()
   
    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    '''pred test set one by one and save error pred img'''
    if args.aug:
        checkpoint = torch.load(os.path.join(ckpt_dir, args.arch.lower() + '_aug_trial{}'.format(trial) + '_best.pth.tar'))
        prob_error_file = 'prob_error_test_img_{}_aug_trial{}.csv'.format(args.arch.lower(), trial)
        prob_test_file = 'prob_test_img_{}_aug_trial{}.csv'.format(args.arch.lower(), trial)
    else:
        checkpoint = torch.load(os.path.join(ckpt_dir, args.arch.lower() + '_trial{}'.format(trial) + '_best.pth.tar'))
        prob_error_file = 'prob_error_test_img_{}_trial{}.csv'.format(args.arch.lower(), trial)
        prob_test_file = 'prob_test_img_{}_trial{}.csv'.format(args.arch.lower(), trial)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    prec1_t_test, loss_t_test, pred_test, prob_test = validate(test_loader, model, criterions, args, mode='test')

    prob_test = np.array(prob_test)
    prob_dict_test = {'p'+str(ii+1):prob_test.T[ii] for ii in range(args.num_classes)}
    #img_name_dict_test = {'img_name':prob_img_name_test}
    img_name_dict_test = {'img_name':img_path_test}
    prob_dict_test.update(img_name_dict_test)
    prob_df_test = pd.DataFrame(prob_dict_test)
    columns=['img_name']
    prob_df_test.to_csv(pred_img_path+prob_test_file, columns=columns.extend('p'+str(ii+1) for ii in range(args.num_classes)), index=False)


if __name__ == '__main__':
    for trial in range(args.ntrial):
        time1 = time.time()
        best_prec1 = 0
        best_test_prec1 = 0
        acc1_tr, losses_tr = [], []
        losses_cl_tr = []
        acc1_val, losses_val, losses_et_val = [], [], []
        acc1_test, losses_test, losses_et_test = [], [], []
        acc1_t_tr, acc1_t_val, acc1_t_test = [], [], []
        learning_rate, weights_cl = [], []
        main(trial+1)
        time2 = time.time() - time1
        print(time2)
