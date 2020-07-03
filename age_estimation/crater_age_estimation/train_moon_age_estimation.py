
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


parser = argparse.ArgumentParser(description='PyTorch Moon Age Estimation Model Training')
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
    global args, best_prec1, best_loss1, best_test_prec1
    global acc1_tr, losses_tr 
    global losses_cl_tr
    global acc1_val, losses_val, losses_et_val
    global acc1_test, losses_test, losses_et_test
    global weights_cl
    args = parser.parse_args()
    print(args)

    attr_range = (0,78)
    # Data loading
    # train data
    if args.aug:
        df_train = pd.read_csv('input/train_imgs_gray_CE1_single_size/train_labels_{}_aug_trial{}.csv'.format(256, trial))
        num_classes = int(max(df_train['label']))
        train_img_path = 'input/train_imgs_gray_CE1_single_size/train_{}_aug/'.format(256)
        train_attr_path = 'input/train_attr_CE1_aug/'
    else:
        df_train = pd.read_csv('input/train_imgs_gray_CE1_single_size/train_labels_{}_trial{}.csv'.format(256, trial))
        num_classes = int(max(df_train['label']))
        train_img_path = 'input/train_imgs_gray_CE1_single_size/train_{}/'.format(256)
        train_attr_path = 'input/train_attr_CE1/'

    img_path_train = []
    attr_path_train = []
    y_train = []

    for f, tags in tqdm(df_train.values, miniters=100):
        img_path = train_img_path + '{}.jpg'.format(f.split('.')[0])
        img_path_train.append(img_path)
        attr_path = train_attr_path + '{}.csv'.format(f)
        attr_path_train.append(attr_path)
        y_train.append(tags-1)

    img_path_train, attr_path_train, y_train = shuffle(img_path_train, attr_path_train, y_train, random_state = 24)

    # valid data
    df_valid = pd.read_csv('input/train_imgs_gray_CE1_single_size/valid_labels_{}_trial{}.csv'.format(256, trial))

    valid_img_path = 'input/train_imgs_gray_CE1_single_size/train_{}/'.format(256)
    valid_attr_path = 'input/train_attr_CE1/'

    img_path_valid = []
    attr_path_valid = []
    y_valid = []

    for f, tags in tqdm(df_valid.values, miniters=100):
        img_path = valid_img_path + '{}.jpg'.format(f.split('.')[0])
        img_path_valid.append(img_path)
        attr_path = valid_attr_path + '{}.csv'.format(f)
        attr_path_valid.append(attr_path)
        y_valid.append(tags-1)

    img_path_valid, attr_path_valid, y_valid = shuffle(img_path_valid, attr_path_valid, y_valid, random_state = 24)
    
    # test data
    df_test = pd.read_csv('input/train_imgs_gray_CE1_single_size/test_labels_{}_trial{}.csv'.format(256, trial))

    test_img_path = 'input/train_imgs_gray_CE1_single_size/train_{}/'.format(256)
    test_attr_path = 'input/train_attr_CE1/'

    img_path_test = []
    attr_path_test = []
    y_test = []

    for f, tags in tqdm(df_test.values, miniters=100):
        img_path = test_img_path + '{}.jpg'.format(f.split('.')[0])
        img_path_test.append(img_path)
        attr_path = test_attr_path + '{}.csv'.format(f)
        attr_path_test.append(attr_path)
        y_test.append(tags-1)

    img_path_test, attr_path_test, y_test = shuffle(img_path_test, attr_path_test, y_test, random_state = 24)

    
    # Unlabel Data loading code
    df_unlabel = pd.read_csv('input/imgs_gray_CE1_unlabel_single_size/train_filelist_CE1_unlabel.csv')
    
    unlabel_img_path = 'input/imgs_gray_CE1_unlabel_single_size/train_256/'
    unlabel_attr_path = 'input/attr_CE1_unlabel/'
    
    img_path_unlabel = []
    attr_path_unlabel = []
    y_unlabel = []
    
    for f, tags in tqdm(df_unlabel.values, miniters=100):
        img_path = unlabel_img_path + '{}.jpg'.format(f)
        img_path_unlabel.append(img_path)
        attr_path = unlabel_attr_path + '{}.csv'.format(f)
        attr_path_unlabel.append(attr_path)
        y_unlabel.append(tags-1)
    
    img_path_unlabel, attr_path_unlabel, y_unlabel = shuffle(img_path_unlabel, attr_path_unlabel, y_unlabel, random_state = 24)
    
    print('\nTraining on {} samples'.format(len(img_path_train)))
    print('Unlabeling on {} samples'.format(len(img_path_unlabel)))
    print('Validating on {} samples'.format(len(img_path_valid)))
    print('Testing on {} samples\n'.format(len(img_path_test)))

   
    labelset = MoonData(img_path_train, attr_path_train, labels=y_train, attr_range=attr_range)
    unlabelset = MoonData(img_path_unlabel, attr_path_unlabel, labels=y_unlabel, attr_range=attr_range)
    batch_size_label = args.batch_size//2
    batch_size_unlabel = args.batch_size//2
    if args.model == 'baseline': batch_size_label=args.batch_size

    label_loader = data.DataLoader(labelset, 
        batch_size=batch_size_label, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True)
    label_iter = iter(label_loader) 

    unlabel_loader = data.DataLoader(unlabelset, 
        batch_size=batch_size_unlabel, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True)
    unlabel_iter = iter(unlabel_loader) 

    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)


    validset = MoonData(img_path_valid, attr_path_valid, labels=y_valid, attr_range=attr_range)
    val_loader = data.DataLoader(validset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True)

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
    
    if args.resume:
        checkpoint_org = torch.load(os.path.join('checkpoints', 'resnet50_best_'+str(args.attr_filters[0])+'.pth.tar'))
        model.load_state_dict(checkpoint_org['state_dict'])
        if args.model == 'mt':
            model_teacher.load_state_dict(checkpoint_org['state_dict'])
    
    if args.optim == 'sgd' or args.optim == 'adam':
        pass
    else:
        print('Not Implemented Optimizer')
        assert(False)
 
        
    if args.aug:
        ckpt_dir = args.ckpt+'_'+args.dataset+'_'+args.arch+'_'+args.model+'_aug'+'_'+args.optim
    else:
        ckpt_dir = args.ckpt+'_'+args.dataset+'_'+args.arch+'_'+args.model+'_'+args.optim
    ckpt_dir = ckpt_dir + '_e%d'%(args.epochs)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)
    cudnn.benchmark = True

    
    # deifine loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()    
    criterion_l1 = nn.L1Loss(size_average=False).cuda()
   
    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    if args.optim == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    betas=(0.9,0.999),
                                    weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        print('Using SGD optimizer')
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        print('Learning rate schedule for Adam')
        lr = adjust_learning_rate_adam(optimizer, epoch)
        
        # train for one epoch
        if args.model == 'baseline':
            print('Supervised Training')
            for i in range(10): #baseline repeat 10 times since small number of training set 
                prec1_tr, loss_tr = train_sup(label_loader, model, criterions, optimizer, epoch, args)
                weight_cl = 0.0
        elif args.model == 'mt':
            print('Mean Teacher model')
            prec1_tr, loss_tr, loss_cl_tr, prec1_t_tr, weight_cl = train_mt(label_loader, unlabel_loader, model, model_teacher, criterions, optimizer, epoch, args)
        else:
            print("Not Implemented ", args.model)
            assert(False)
        
        # evaluate on validation set        
        prec1_val, loss_val, _, _ = validate(val_loader, model, criterions, args, 'valid')
        prec1_test, loss_test, _, _ = validate(test_loader, model, criterions, args, 'test')
        if args.model=='mt':
            prec1_t_val, loss_t_val, _, _ = validate(val_loader, model_teacher, criterions, args, 'valid')
            prec1_t_test, loss_t_test, _, _ = validate(test_loader, model_teacher, criterions, args, 'test')

        # append values
        acc1_tr.append(prec1_tr)
        losses_tr.append(loss_tr)
        acc1_val.append(prec1_val)
        losses_val.append(loss_val)
        acc1_test.append(prec1_test)
        losses_test.append(loss_test)
        if args.model != 'baseline': 
            losses_cl_tr.append(loss_cl_tr)
        if args.model=='mt':
            acc1_t_tr.append(prec1_t_tr)
            acc1_t_val.append(prec1_t_val)
            acc1_t_test.append(prec1_t_test)
        weights_cl.append(weight_cl)
        learning_rate.append(lr)

        # remember best prec@1 and save checkpoint
        if args.model == 'mt': 
            print("---- Loss of Epoch [{}]: {:.3f}".format(epoch, loss_t_val))
            is_best = loss_t_val < best_loss1
            if is_best:
                best_test_prec1_t = prec1_t_test
                best_test_prec1 = prec1_test
            print("---- Best test precision: {:.3f}".format(best_test_prec1_t))
            best_loss1 = min(loss_t_val, best_loss1)
            best_prec1 = max(prec1_t_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_test_prec1' : best_test_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'acc1_test' : acc1_test,
                'losses_test' : losses_test,
                'acc1_t_tr': acc1_t_tr,
                'acc1_t_val': acc1_t_val,
                'acc1_t_test': acc1_t_test,
                'state_dict_teacher': model_teacher.state_dict(),
                'best_test_prec1_t' : best_test_prec1_t,
                'weights_cl' : weights_cl,
                'learning_rate' : learning_rate,
            }
       
        else:
            print("---- Loss of Epoch [{}]: {:.3f}".format(epoch, loss_val))
            is_best = loss_val < best_loss1
            if is_best:
                best_test_prec1 = prec1_test
            print("---- Best test precision: {:.3f}".format(best_test_prec1))
            best_loss1 = min(loss_val, best_loss1)
            best_prec1 = max(prec1_val, best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'best_test_prec1' : best_test_prec1,
                'acc1_tr': acc1_tr,
                'losses_tr': losses_tr,
                'losses_cl_tr': losses_cl_tr,
                'acc1_val': acc1_val,
                'losses_val': losses_val,
                'acc1_test' : acc1_test,
                'losses_test' : losses_test,
                'weights_cl' : weights_cl,
                'learning_rate' : learning_rate,
            }

        if args.aug:
            save_checkpoint(dict_checkpoint, is_best, args.arch.lower()+'_aug_trial{}'.format(trial), dirname=ckpt_dir)
        else:
            save_checkpoint(dict_checkpoint, is_best, args.arch.lower()+'_trial{}'.format(trial), dirname=ckpt_dir)
        
        
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirname='.'):
    fpath = os.path.join(dirname, filename + '_latest.pth.tar')
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dirname, filename + '_best.pth.tar')
        shutil.copyfile(fpath, bpath)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""
    
    boundary = [args.epochs//2,args.epochs//4*3,args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    boundary = [args.epochs//5*4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    print('Learning rate: %f'%lr)
    #print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

  

if __name__ == '__main__':
    
    for trial in range(args.ntrial):
        best_prec1 = 0
        best_loss1 = 1e10
        best_test_prec1 = 0
        acc1_tr, losses_tr = [], []
        losses_cl_tr = []
        acc1_val, losses_val, losses_et_val = [], [], []
        acc1_test, losses_test, losses_et_test = [], [], []
        acc1_t_tr, acc1_t_val, acc1_t_test = [], [], []
        learning_rate, weights_cl = [], []
        main(trial+1)
