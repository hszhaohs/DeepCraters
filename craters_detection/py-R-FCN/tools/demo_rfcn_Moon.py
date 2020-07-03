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

CLASSES = ('__background__',
           'impact_crater')

NETS = {'ResNet-101': ('ResNet-101',
                  'moon_resnet101_rfcn_iter_100000.caffemodel')}


def vis_detections(im, items):
    """Draw detected bounding boxes."""
    rects = []
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
            cs = [class_name, score]
            cas.append(cs)
    return rects, cas

        
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
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
    rects, cas = vis_detections(im, cand)

    header_rects = ['xmin','ymin','xmax','ymax']
    header_cas = ['label','accuracy']
    
    csvfileName_rects = '/home/hai/py-R-FCN/test_output/test_boxes_output/' + image_name.split('.')[0] + '_boxes.csv'
    csvfileName_cas = '/home/hai/py-R-FCN/test_output/test_labels_output/' + image_name.split('.')[0] + '_label.csv'
    
    List2CSV(csvfileName_rects, rects, header_rects)
    List2CSV(csvfileName_cas, cas, header_cas)
    # List2CSV(csvfileName_rects, boxes, header_rects)
    # List2CSV(csvfileName_cas, cas, header_cas)

    fig_save_name = '/home/hai/py-R-FCN/test_output/test_images_output/new-' + image_name

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
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    plt.savefig(fig_save_name)  # save and output the labeled figure

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
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

    im_names = ['0037_0110_CE1.jpg', '0057_0087_CE1.jpg', '0004_0151_CE2.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

