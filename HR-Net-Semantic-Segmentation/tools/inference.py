# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import torch.nn.functional as F

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')        
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    start = timeit.default_timer()

    name = 'snow'

    image = cv2.imread('/remote-home/zhangjingdong/HRNet-Semantic-Segmentation/'+name+'.jpg',cv2.IMREAD_COLOR)

    image = input_transform(image)

    image = F.interpolate(image, (750, 1562), mode='bilinear')

    size = image.size()

    pred, pred_aux = model(image)

    pred = F.interpolate(
        input=pred, size=size[-2:],
        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    )

    pred_aux = F.interpolate(
        input=pred_aux, size=size[-2:],
        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
    )

    pred_aux_norm = (pred_aux - torch.min(pred_aux))/(torch.max(pred_aux) - torch.min(pred_aux))
    pred_aux_max = torch.max(pred_aux_norm, dim=1, keepdim=False).values
    ones = torch.ones_like(pred_aux_max) * 200
    zeros = torch.zeros_like(pred_aux_max)
    unconf = torch.where(pred_aux_max < 0.6, ones, zeros)

    pred = torch.argmax(nn.Softmax2d()(pred),dim=1)
    pred_aux = torch.argmax(nn.Softmax2d()(pred_aux), dim=1)

    print(pred.size())
    print(pred_aux.size())
    print(unconf.size())

    #save_image(pred, '/remote-home/zhangjingdong/HRNet-Semantic-Segmentation/pred1.png')
    save_image(pred_aux, '/remote-home/zhangjingdong/HRNet-Semantic-Segmentation/'+name+'_pred.png')
    save_image(unconf, '/remote-home/zhangjingdong/HRNet-Semantic-Segmentation/'+name+'_unconf.png', color_required=False)

    print(pred)
    print(pred_aux)


    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = torch.Tensor(image.copy()).cuda()
    image = image / 255.0
    image -= torch.Tensor([0.485, 0.456, 0.406]).cuda()
    image /= torch.Tensor([0.229, 0.224, 0.225]).cuda()
    return image.permute(2, 0, 1).unsqueeze(0)

def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 19
    label_colors = get_cityscapes_colors()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        b[label_mask == ll] = label_colors[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.uint8(rgb)


def decode_segmap_sequence(label_masks, dataset='cityscapes'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def get_cityscapes_colors():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def save_image(input_tensor, filename, color_required=True):
    input_tensor = input_tensor.permute(1, 2, 0).type(torch.uint8).detach().cpu().numpy()
    if color_required == True:
        input_tensor = decode_segmap(input_tensor.squeeze(2))
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


if __name__ == '__main__':
    main()
