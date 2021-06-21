#!/usr/bin/env python
from __future__ import print_function

import copy

import click
import cv2
import numpy as np
import torch
import os, os.path
import datetime
import PIL
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image, ImageDraw

from src.utils.models import Darknet
from src.utils.grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

def grad_save(filename, flag_data):
    flag_data -= flag_data.min()
    flag_data /= flag_data.max()
    flag_data *= 255.0

    cv2.imwrite(filename, np.uint8(flag_data))


def grad_cam_save(filename, flag_data, raw_img):
    height, width, _ = raw_img.shape
    flag_data = cv2.resize(flag_data, (width, height))
    flag_data = cv2.applyColorMap(np.uint8(flag_data * 255.0), cv2.COLORMAP_JET)
    flag_data = flag_data.astype(np.float) + raw_img.astype(np.float)
    flag_data = flag_data / flag_data.max() * 255.0
    cv2.imwrite(filename, np.uint8(flag_data))

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append('yolov4')

model_names.append('yolov4-tiny')

# model_names.append('yolov4_medium')


@click.command()
@click.option('-i', '--image-path', help="Path to image", type=str, required=True)
# @click.option('-a', '--arch', help="a model name from ```torchvision.models```, e.g., 'resnet152' (default: yolov4)", type=click.Choice(model_names), default='yolov4')
@click.option('-c', '--config', help="config path", required=True)
@click.option('-w', '--weights', help="weights path", required=True)
@click.option('-n', '--num_class', help="number of classes to generate (default: 2)", type=int, default=2)
@click.option('--cuda/--no-cuda', help="GPU mode or CPU mode (default: cuda mode)", default=True)
def main(image_path, config, weights, num_class, cuda):

    arch = 'yolov4-tiny' if 'tiny' in config else 'yolov4'

    startTime = datetime.datetime.now()

    CONFIG = {
        'yolov4': {
            'target_layer': 'module_list.5.conv_5',
            'input_size': 512
        },
        'yolov4-tiny':{
            'target_layer': 'module_list.5',
            'input_size':512
        }
    }.get(arch)

    flag_img_name= (image_path.split("/"))[-1]
    img_name = flag_img_name.split(".")[0]

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # v4_model_cfg = "./cfg/yolov4.cfg"

    # tiny_model_cfg = "./cfg/yolov4-tiny.cfg"
    # v4_model_cfg = config
    # tiny_model_cfg = config
    # v4_weights_path = "../weights/yolov4_training_4000.weights"

    # tiny_weights_path = "../weights/yolov4-tiny_training_3000.weights"
    
    if arch == 'yolov4-tiny':
            # os.system('./src/utils/darknet/darknet detect ' + tiny_model_cfg + ' ' + tiny_weights_path + ' ' + image_path + ' -thresh 0.95 -nms 0.1')
            # os.rename("./predictions.jpg", "../results/yolov4-tiny/" + img_name)

            with open('./src/config/yolov4-tiny_intersting_layers.txt', 'r') as f:
                interesting_layer = f.read().split('\n')
            print('Visualization target layers: '+str(interesting_layer))

    elif arch == 'yolov4': 
        # os.system('./src/utils/darknet/darknet detect ' + v4_model_cfg + ' ' + v4_weights_path + ' ' +image_path + ' -thresh 0.999')
        # os.rename("./sample_data/Bottle caps_210510_58_0.jpg", "../results/yolov4/" + img_name)

        # os.system('python3 ./src/utils/yolov4/detect.py --image-folder ' + image_path + ' --output-folder ./results/yolov3_output/ --cfg ' + model_cfg + ' --weights ' + weights_path + ' --conf-thres 0.9999 --nms-thres 0.33 --img-size ' + str(CONFIG['input_size']))
        # print("Finished Yolov3 object detection check!")
        # intersting_layer = 161
        with open('./src/config/yolov4_intersting_layers.txt', 'r') as f:
            interesting_layer = f.read().split('\n')
        print('Visualization target layers: '+str(interesting_layer))


    # if arch == 'yolov4-csp': 
    #     os.system('./src/utils/darknet/darknet detect ' + v4_csp_cfg + ' ' + csp_weights_path + ' ' +image_path + ' -thresh 0.999')
    #     # os.rename("./predictions.jpg", "../results/yolov4-csp/" + img_name)

    #     # os.system('python3 ./src/utils/yolov4/detect.py --image-folder ' + image_path + ' --output-folder ./results/yolov3_output/ --cfg ' + model_cfg + ' --weights ' + weights_path + ' --conf-thres 0.9999 --nms-thres 0.33 --img-size ' + str(CONFIG['input_size']))
    #     # print("Finished Yolov3 object detection check!")

    #     with open('./src/config/yolov4-csp_interesting_layers.txt', 'r') as f:
    #         intersting_layer = f.read().split('\n')
    #     print('Visualization target layers: '+str(intersting_layer))        


    # dictionary = list()
    # with open('src/config/synset_words.txt') as lines:
    #     for line in lines:
    #         line = line.strip().split(' ', 1)[1]
    #         line = line.split(', ', 1)[0].replace(' ', '_')
    #         dictionary.append(line)

    model = Darknet(config, CONFIG['input_size'])
    print('-------------------------------------------------------------------')
    # if weights.endswith('.weights'):
    model.load_weights(weights)
    # else:
    #     model = models.__dict__[arch](pretrained=True)

    #print out all layers' names
    # print(*list(model.named_modules()), sep='\n')

    model.to(device)
    model.eval()

    # img_path = image_directory_path
    # ttl_img = []
    # valid_images = [".jpg"]
    # for f in os.listdir(img_path):
    #     ext = os.path.splitext(f)[1]
    #     if ext.lower() not in valid_images:
    #         continue
    #     ttl_img.append((os.path.join(img_path, f)))
    # print(ttl_img)
    # print(str(len(ttl_img)) + " images in total")     


    print('loading '+img_name+' ......')

    # if arch == 'yolov4' or arch == 'yolov4-tiny':

    raw_img = Image.open(image_path).convert('RGB')
    # w, h = CONFIG['input_size'],CONFIG['input_size']
    w,h = raw_img.width,raw_img.height
    # print(raw_img.size)

    max_dimension = max(h, w)
    pad_w = int((max_dimension - w) / 2)
    pad_h = int((max_dimension - h) / 2)
    # ratio = float(CONFIG['input_size']) / float(max_dimension)

    raw_img = transforms.functional.pad(raw_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    raw_img = transforms.functional.resize(raw_img, (CONFIG['input_size'], CONFIG['input_size']))
    # raw_img = transforms.functional.resize(raw_img, (w, w))
    flag_image = np.array(raw_img)   
    # print(flag_image.shape)

    raw_img = transforms.functional.to_tensor(raw_img)

    image = raw_img.unsqueeze(0)
    print(image.shape)
    raw_img = raw_img.permute(2, 1, 0)
    # print(raw_img.shape)

    print('loading complete!')

    # =========================================================================

    print('Grad-CAM Visualization on '+ img_name)

    grad_cam = GradCAM(model=model)
    probs, idx = grad_cam.forward(image.to(device))

    print(len(interesting_layer))
    for i in range(0,len(interesting_layer)):
        startTime = datetime.datetime.now()

        grad_cam.backward(idx=idx[0])
        output = grad_cam.generate(target_layer=interesting_layer[i])

        if output is None:
            continue

        # grad_cam_save('results/{}_grad_cam_{}.png'.format(i, arch), output, raw_img)
        grad_cam_image_path = './results/raw/{}_grad_cam_{}.png'.format(img_name, interesting_layer[i])
        grad_cam_save(grad_cam_image_path, output, flag_image)

        endTime = datetime.datetime.now()
        time_spent =  endTime - startTime
        # yolov4_output_path = './results/yolov3_output/' + img_name
        yolov4_output_path = './results/yolov4/' + img_name
        tiny_output_path = "./results/yolov4-tiny/" + img_name

        grad_img = Image.open(grad_cam_image_path)
        if arch == 'yolov4':
            pred_img = Image.open(image_path).convert('RGB')
        if arch == 'yolov4-tiny':
            pred_img = Image.open(image_path).convert('RGB')
        w,h = pred_img.width,pred_img.height
        max_dimension = max(h, w)
        pad_w = int((max_dimension - w) / 2)
        pad_h = int((max_dimension - h) / 2)

        pred_img = transforms.functional.pad(pred_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        pred_img = transforms.functional.resize(pred_img, (CONFIG['input_size'], CONFIG['input_size']))

        imgs    = [grad_img, pred_img]
        # # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

        # # save that beautiful picture
        imgs_comb = PIL.Image.fromarray( imgs_comb )
        imgs_comb.save( './results/compare/{}_{}_compare.png'.format(img_name, interesting_layer[i]))

        print('Time elapsed on layer ['+ interesting_layer[i] + ']: ' + str(time_spent))
        # os.system('open ./results/compare/')

    #     for i in range(0, num_class):
    #         grad_cam.backward(idx=idx[i])
    #         output = grad_cam.generate(target_layer=CONFIG['target_layer'])
    #         grad_cam_save('../results/{}_gcam_{}.png'.format(dictionary[idx[i]], arch), output, raw_image)
    #         print('[{:.5f}] {}'.format(probs[i], dictionary[idx[i]]))

if __name__ == '__main__':
    main()

