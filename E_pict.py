#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg
from models.factory import build_net
from models.enhancement import EnhanceNet as PreProcessor
from torchvision.utils import make_grid

from tqdm import tqdm

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benckmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr

def load_models():
    print('build network')
    net, _ = build_net('test', cfg.NUM_CLASSES, 'vgg')
    net.eval()
    net.load_state_dict(torch.load('../final_weights/Face-Detector.pth'))

    pre_processor = PreProcessor()
    pre_processor.eval()
    pre_processor.load_state_dict(torch.load('../final_weights/Illumination-Enhancer.pth'))

    if use_cuda:
        net = net.cuda()
        pre_processor = pre_processor.cuda()

    return net, pre_processor

if __name__ == '__main__':

    ''' Parameters '''

    ## DSFD mAP = 16.1
    USE_MULTI_SCALE = True
    MY_SHRINK = 1

    ## DSFD mAP = 15.3
    # USE_MULTI_SCALE = False
    # MY_SHRINK = 2

    save_path = './test_face_pict_H'
    img_dir = './test_face_pict'

    ''' Main Test '''

    net, pre_processor = load_models()

    def load_images(file_pathname):

        result_paths = []
        for filename in tqdm(os.listdir(file_pathname)):
            for picture in tqdm(os.listdir(file_pathname + '/' + filename)):
                image = cv2.imread(file_pathname + '/' + filename + '/' + picture)
                if image is not None and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    print("no_ture")
                if image is not None:
                    cv2.imwrite(file_pathname + '/' + filename + '/' + picture, image)
                    result_paths.append(file_pathname + '/' + filename + '/' + picture)
                else:
                    pass
        return result_paths

    img_list = load_images(img_dir)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_path in tqdm(img_list):
        # Load images       
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = np.array(image)

        # Low light enhancement
        image_to_enhance = torch.from_numpy(image / 255.0).float()
        image_to_enhance = image_to_enhance.permute(2,0,1).unsqueeze(0)

        if use_cuda:
            image_to_enhance = image_to_enhance.cuda()

        with torch.no_grad():
            image_to_enhance = pre_processor(image_to_enhance)


        # #E_test
        # img1 = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(tensor_to_image(image_to_enhance), cv2.COLOR_RGB2BGR)
        # images = cv2.hconcat([img1,img2])
        # cv2.imshow('image/E(image)',images)
        # cv2.imwrite(img_dir + '/' + img_path.split('/')[-1],images)
        # cv2.waitKey(0)


        image = tensor_to_image(image_to_enhance)

        img_path_split = img_path.split('/')

        if not os.path.exists(save_path + '/' + img_path_split[-2]):
            os.makedirs(save_path + '/' + img_path_split[-2])

        cv2.imwrite(save_path + '/' + img_path_split[-2] + '/' + img_path_split[-1], image)


