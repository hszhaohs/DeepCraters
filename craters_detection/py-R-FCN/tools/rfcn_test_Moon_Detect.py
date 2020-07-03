#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
import matplotlib
matplotlib.use('Agg')
import csv
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import cPickle
import uuid
from datasets.voc_eval_moon import voc_eval
import datetime


CLASSES = ('__background__',
           'impact_crater')

NETS = {'ResNet-101': ('ResNet-101',
                  'moon_resnet101_rfcn_iter_100000.caffemodel')}


def vis_detections(im, items):
    """Draw detected bounding boxes."""
    rects = []
    rects_out = []
    cas = []
    for item in items:
        class_name = item[0]
        dets = item[1]
        thresh = item[2]

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue

        for i in inds:
    	    bbox = dets[i, :4]
    	    score = dets[i, -1]
            
	    rect = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            rects.append(rect)
            rects_out.append(bbox)
            cs = [class_name, score]
            cas.append(cs)
    return rects, cas, rects_out

        
def demo(net, image_name, image_path, CONF_THRESH, NMS_THRESH, boxes_savepath, labels_savepath, images_savepath):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class

    cand = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        one = [cls, dets, CONF_THRESH]
        cand.append(one)
    rects, cas, rects_out = vis_detections(im, cand)

    header_rects = ['xmin','ymin','xmax','ymax']
    header_cas = ['label','accuracy']
    
    csvfileName_rects = boxes_savepath + '/' + image_name.split('.')[0] + '_boxes.csv'
    csvfileName_rects2 = boxes_savepath + '/' + image_name.split('.')[0] + '_boxes2.csv'
    csvfileName_cas = labels_savepath  + '/' + image_name.split('.')[0] + '_label.csv'
    
    List2CSV(csvfileName_rects, rects_out, header_rects)
    List2CSV(csvfileName_rects2, rects, header_rects)
    List2CSV(csvfileName_cas, cas, header_cas)

    fig_save_name = images_savepath + '/new-' + image_name  # path + fig_name of output

    fig, ax = plt.subplots(figsize=(12,12))
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    for ii in range(len(rects)):
        r = rects[ii]
        ax.add_patch(
            plt.Rectangle((r[0], r[1]), r[2], r[3] ,
                fill=False, edgecolor='red', linewidth=3.5))
        c = cas[ii]
        ax.text(r[0], r[1] - 2,
                '{:s} {:.3f}'.format(c[0], c[1]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=16, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    plt.savefig(fig_save_name)  # save and output the labeled figure
    plt.close()
    
    return scores, boxes, timer

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--conf', dest='CONF_THRESH', help='CONF_THRESH to use',
                        default=0.8, type=float)
    parser.add_argument('--nms', dest='NMS_THRESH', help='NMS_THRESH to use',
                        default=0.3, type=float)
    parser.add_argument('--dset', dest='Dataset', help='Dataset to use',
                        default='CE1', type=str)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        choices=NETS.keys(), default='ResNet-101')

    args = parser.parse_args()

    return args

def List2CSV(fileName = '',dataList = [], headerList = []):
    with open(fileName, 'wb') as csvFile:
        csvWriter = csv.writer(csvFile)

        csvWriter.writerow(headerList)
        for data in dataList:
            csvWriter.writerow(data)
        csvFile.close


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'rfcn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    
    Result_SavePath = '/home/hai/py-R-FCN/dete_output_{}'.format(args.Dataset)
    file_path = '/home/hai/Moon_All_Patch'
    test_file = os.path.join(file_path, 'filelist_{}.txt'.format(args.Dataset))
    file_path_img = os.path.join(file_path, 'Image_Patch_{}'.format(args.Dataset))
    
    if not os.path.exists(Result_SavePath):
        os.makedirs(Result_SavePath)
        
    boxes_savepath = os.path.join(Result_SavePath, 'test_boxes_output')
    if not os.path.exists(boxes_savepath):
        os.makedirs(boxes_savepath)

    labels_savepath = os.path.join(Result_SavePath, 'test_labels_output')
    if not os.path.exists(labels_savepath):
        os.makedirs(labels_savepath)

    images_savepath = os.path.join(Result_SavePath, 'test_images_output')
    if not os.path.exists(images_savepath):
        os.makedirs(images_savepath)


    CONF_THRESH = args.CONF_THRESH
    NMS_THRESH = args.NMS_THRESH
    max_per_image = 100
    num_classes = 2
    
    
    with open(test_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    
    num_images = len(image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    
    for i in xrange(num_images):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'dete for {}'.format(image_index[i])
        im_name = image_index[i] + '.jpg'
        image_path = os.path.join(file_path_img, im_name)
        im = cv2.imread(image_path)
    
        
        scores, boxes, time_pre = demo(net, im_name, image_path, CONF_THRESH, NMS_THRESH, boxes_savepath, labels_savepath, images_savepath)
        _t['im_detect'] = time_pre
    
        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > CONF_THRESH)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, NMS_THRESH)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets
    
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()
    
        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
    
