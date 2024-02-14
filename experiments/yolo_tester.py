import argparse
import time
from pathlib import Path
import sys
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from numpy import random
from torch.utils.data import DataLoader

sys.path.append('../')



from models.experimental import attempt_load
from yolo_utils.general import check_img_size, non_max_suppression
from yolo_utils.plots import plot_one_box
from yolo_utils.torch_utils import select_device, TracedModel

'''
thresholds : list ([iou_thres, conf_thres])
img0 : image loaded with cv2 (BGR)
img : img0 to tensor
prompt : prompt text
model_dict : returned by load_yolov7()
returns : 1) predictions of the model of the format (a list of tensors) ->
          [bb_xx, bb_xy, bb_yx, bb_yy, conf, obj_class] x num of predicted bbs,
          2) image visualizing the bbs.
saves : a visualization to ./outputs_vis if save_vis == True.
'''

def infer_yolov7(thresholds, img = None, img0 = None, save_vis = True, model_dict = None, prompt = None, args = None):
    
    # suppose that we have the img ass a BGR image, 
    # which is processed by cv2.
    
    model = model_dict['model']
    names = model_dict['names']
    colors = model_dict['colors']
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    ax[0].imshow(img0)
    ax[1].imshow(T.ToPILImage()(img[0]))
    fig.savefig('./saved_fig.png')
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # to a valid torch tensor.
        
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
        
    # default : conf_thres = 0.65, iou_thres = 0.35 (hyper-parameters)
    pred = non_max_suppression(pred, thresholds[0], thresholds[1])
    print(pred)
    
    # if visualization is enabled, save the visualization.
    if save_vis:
        s = '' # empty string
        img_arr = []
        # process detections
        for i, det in enumerate(pred):  # detections per image
            save_path = str(f'{args.yolo_output_dir}/{prompt}_{i}.png')
            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
        
                cv2.imwrite(save_path, img0)
                img_arr.append(img0)
                print(f" The image with the result is saved in: {save_path}")
        
            # Print time (inference + NMS)
            print(f'{s}Done.')

    return {
        'pred' : pred,
        'img_arr' : img_arr
    }




# args given from exp_seg.py

'''
returns : model and its specifications (module names and colors)
'''

def load_yolov7(args=None, dataset = None, traced = False):

    '''s
    args.yolo_weights == './ckpt/yolov7.pt'
    args.yolo_output_dir == './outputs'
    args.yolo_imgsz == 512
    '''
    
    # assume that we use cuda:0
    device = select_device('0') # 'cuda:0'
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load('./yolo_ckpt/yolov7.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(512, s=stride)  # check img_size (adjust)

    if traced:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16
        
    # toy
    if device.type != 'cpu':
        out = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #print((1, 3, imgsz, imgsz))
    #print(out)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    return {
        'model' : model,
        'names' : names,
        'colors' : colors,
    }
    
    '''
    # inference (for simple vis on a single image- not related to our own experiment)
    
    img0 = cv2.imread('../example/2008_001997.jpg')
    img0 = cv2.resize(src=img0s, dsize=(512,512),interpolation=cv2.INTER_NEAREST_EXACT)
    img = torch.from_numpy(img0.transpose(2,0,1)).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
    print(pred, '\n', pred[0].shape)
    # conf_thres = 0.65, iou_thres = 0.35 (hyper-parameters)
    pred = non_max_suppression(pred, 0.65, 0.35)
    print(pred, '\n', pred[0].shape)
    

    s = ''
    # process detections
    for i, det in enumerate(pred):  # detections per image
        save_path = str(f'./outputs/{i}.png')
        #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
    
            cv2.imwrite(save_path, img0)
            print(f" The image with the result is saved in: {save_path}")
    
        # Print time (inference + NMS)
        print(f'{s}Done.')
    '''

'''
if __name__ == '__main__':
    load_yolov7(args = None)
'''


    