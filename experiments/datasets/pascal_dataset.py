import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets.pascal_utils import PASCAL_CLASSES, is_plural
from experiments.exp_utils import ImageScribble, ImageAnnotation
from torch.nn import functional as F
from utils.vis_utils import resize_and_pad
from copy import deepcopy
import cv2


# need to replace those.

def prepare_scribble_from_img(img_path, width, height, device):
    '''
    Change an image of black scribble with a white background 
    to a white scribble with black background. 
    '''
    img_path_list = []
    if type(img_path) is str:
        img_path_list.append(img_path)
    elif type(img_path) is list:
        img_path_list = img_path
    
    scribble_list = [None] * len(img_path_list)

    for i in range(len(img_path_list)):
        scribble_list[i] = Image.open(img_path_list[i]).convert('L')
        scribble_list[i] = np.array(scribble_list[i])
        scribble_list[i] = 255 - scribble_list[i]
        scribble_list[i] = np.where(scribble_list[i] < 128, 0, 255).astype(np.float32)
        scribble_list[i] /= 255.0
        scribble_list[i] = torch.from_numpy(scribble_list[i])
        scribble_list[i] = F.interpolate(scribble_list[i].unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        scribble_list[i] = scribble_list[i].bool().float().to(device)

    return scribble_list


def preprocess_scribble(img_path, width=64, height=64, device=torch.device("cpu")):
    img_path_list = []
    if type(img_path) is str:
        img_path_list.append(img_path)
    elif type(img_path) is list:
        img_path_list = img_path
    else:
        raise TypeError("img_path should be a string or a list of strings")

    scribble_list = prepare_scribble_from_img(img_path_list, width, height, device)

    return scribble_list



# data_split_idx : choose the split (naively split in 4 parts) - 0, 1, 2, 3 or -1 (use the entire dataset)

class PascalDataset(Dataset):
    def __init__(self, 
                 root_dir,
                 split='train',
                 dataset_type='VOC2012', 
                 transform=None, 
                 require_seg=False,
                 to_tensor=False,
                 output_dir='../experiments/outputs',
                 width=512,
                 height=512,
                 data_split_idx=-1,
                 n_splits = 3
                 ):
        assert os.path.exists(root_dir), f"Directory {root_dir} does not exist."
        
        #assert os.path.exists(output_dir), f"Directory {output_dir} does not exist."
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.require_seg = require_seg
        self.to_tensor = to_tensor

        self.class_to_idx = PASCAL_CLASSES
        self.output_dir = output_dir

        self.width = width
        self.height = height

        # Define the path to the dataset
        self.path = os.path.join(self.root_dir, dataset_type)

        # Define the path to the image and annotation directories
        self.img_dir = os.path.join(self.path, 'JPEGImages')
        self.seg_dir = os.path.join(self.path, 'SegmentationClass')
        self.scr_dir = os.path.join(self.path, 'Scribbles')
        self.bb_dir = os.path.join(self.path, 'Annotations')

        # Define the path to the split file
        split_file = os.path.join(self.path, 'ImageSets', 'Main', split + '.txt')

        # Read the split file and store the image names
        with open(split_file, 'r') as f:
            if self.require_seg:
                self.img_names = [line.strip() for line in f.readlines() if os.path.exists(os.path.join(self.seg_dir, line.strip() + '.png'))]
            else:
                self.img_names = [line.strip() for line in f.readlines()]

        # choose the split and use it. (3 splits)
        dataset_size = len(self.img_names)
        
        assert n_splits <= dataset_size
        
        chunks = [self.img_names[i:i + dataset_size//n_splits] for i in range(0, dataset_size, dataset_size//n_splits)]
        if data_split_idx != -1:
            self.img_names = deepcopy(chunks[data_split_idx])
        print(f'dataset size : {len(self.img_names)}')

        '''
        # from the scr_dir, exclude the paths that are not working.
        self.entire_scr = []
        for im in self.img_names:
            self.entire_scr.append(os.path.join(self.scr_dir, im + '.xml'))
        
        for idx, scr_path in enumerate(self.entire_scr):
            if ImageScribble(scr_path, test_mode=True)._get_scribble() == -9999:
                # exclude the path
                #print(os.path.split(scr_path)[1][:-4])
                self.img_names.remove(os.path.split(scr_path)[1][:-4]) 
        '''

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        # Load the image and annotation
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        seg_path = os.path.join(self.seg_dir, img_name + '.png')
        scr_path = os.path.join(self.scr_dir, img_name + '.xml')
        bb_path = os.path.join(self.bb_dir, img_name + '.xml')

        item_output_dir = os.path.join(self.output_dir, img_name)
        if not os.path.exists(item_output_dir):
            os.makedirs(item_output_dir)

        img = Image.open(img_path).convert('RGB')
        original_seg = Image.open(seg_path)

        img = resize_and_pad(img, self.width)
        seg, coords = resize_and_pad(original_seg, self.width, return_coords=True)
        
        img_scribble = ImageScribble(scr_path)
        img_bbox = ImageAnnotation(bb_path)

        # Draw bounding boxes for the 'person' object
        image_with_bbs = img_bbox.draw_annotations(img_path)
        
        scribble_masks = {}

        object_names = [name for name in img_scribble.scribble_coordinates.keys()]
        
        print(f'img_name : {img_name}')
        print(f'object_names : {object_names}')

        for object_name in object_names:
            if object_name != 'background':
                scribble_masks[object_name] = img_scribble.draw_scribble(object_name)
                
        # Apply the transform if specified
        if self.transform:
            img = self.transform(img)

        # Convert the annotation to a tensor
        # Convert the scribble masks also to tensors
        if self.to_tensor:
            seg = torch.from_numpy(np.array(seg)).long()
            original_seg = torch.from_numpy(np.array(original_seg)).long()
            for object_name in object_names:
                if object_name != 'background':
                    scribble_masks[object_name] = torch.from_numpy(np.array(scribble_masks[object_name])).long()

        prompt, classnames = self.get_prompt(scribble_masks)

        # Do not save masks
        '''
        for classname in classnames:
            mask_path = os.path.join(item_output_dir, classname + '.png')
            scribble_masks[classname].save(mask_path)
        '''

        #print(f'img:{img}\n\n img_name:{img_name}\n\n prompt:{prompt}\n\n seg:{seg}\n\n scribble_masks:{scribble_masks}\n\n coords:{coords}')

        #return img, img_name, prompt, seg, scribble_masks, coords
        return {'img': img, 'img_name': img_name, 'prompt': prompt, 'seg': seg, 'scribble_masks': scribble_masks, 'coords': coords, 'original_seg': original_seg, 'bb_vis': np.array(image_with_bbs)[0] }


    def get_prompt(self, scribble_masks):
        prompt = 'a photography of'

        valid_classnames = []
        scribble_classnames = [name for name in scribble_masks.keys()]

        for classname in scribble_classnames:
            if classname == 'background':
                continue

            if is_plural(classname):
                valid_classnames.append(classname)
            elif classname[0] in ['a', 'e', 'i', 'o', 'u']:
                valid_classnames.append('an ' + classname)
            else:
                valid_classnames.append('a ' + classname)

        if len(valid_classnames) == 1:
            prompt += ' ' + valid_classnames[0]
        else:
            prompt += ' ' + ', '.join(valid_classnames[:-1])
            prompt += ' and ' + valid_classnames[-1]

        '''
        if 'background' in scribble_classnames:
            prompt += ' on a background.'
        '''
        return prompt, valid_classnames
