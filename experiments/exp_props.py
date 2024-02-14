import numpy as np   
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

    
def peek_dataset(dataset):    
    # taking a peek at the dataset

    img, img_name, prompt, seg, scribble_masks, coords, gt, bb_vis = dataset[0].values()

    print(f'size of dataset : {len(dataset)}, '
        f'\nimg name : {img_name}, '
        f'\nimg shape : {np.array(img).shape}, '
        f'\nscribble masks : {scribble_masks.keys()} '
        f'\nscribble shape : {len(scribble_masks.values())} '
        f'\nprompt : {prompt} '
        f'\nGT mask shape : {np.array(seg).shape} '
        f'\ncoords : {coords}')
    
    
def axis_off(ax):
    # Remove figure axis for all figures
    for row in ax:
        for col in row:
            col.axis('off')
    
        
# will have to replace below code.

def prompt_to_tokens(prompt, tokenizer):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode

    prompt_tokens = [decoder(token) for token in tokens]
    return prompt_tokens



def find_bbox(mask):
    y_indices, x_indices = np.where(mask.detach().cpu().numpy())

    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return x_min, y_min, x_max - x_min, y_max - y_min



def compute_bbox(scribble, padding=3):
    '''
    Compute the bounding box of a mask.
    '''
    scribble_list = []
    bbox_list = []

    if type(scribble) is list:
        scribble_list = scribble
    elif type(scribble) is torch.Tensor:
        if len(scribble.shape) == 3:
            scribble_list = scribble
        else:
            scribble_list.append(scribble)

    for scribble in scribble_list:
        x_min, y_min, w, h = find_bbox(scribble)

        bboxes = []
        x_max = x_min + w
        y_max = y_min + h

        x_min = max(0, x_min - padding) / scribble.shape[1]
        y_min = max(0, y_min - padding) / scribble.shape[0]
        x_max = min(scribble.shape[1], x_max + padding) / scribble.shape[1]
        y_max = min(scribble.shape[0], y_max + padding) / scribble.shape[0]

        bboxes.append([x_min, y_min, x_max, y_max])
        bbox_list.append(bboxes)

    return bbox_list
        

def compute_mask(bbox_list, size, device=torch.device("cpu")):
    bbox_dist_list = []
    mask_dist_list = []

    for bboxes in bbox_list:
        mask_dist = torch.zeros(size, size).to(device)
        bbox_dist = torch.zeros(size, size).to(device)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox

            mask_dist[int(y_min * size):min(int(y_max * size) + 1, size), 
                      int(x_min * size):min(int(x_max * size) + 1, size)] = 1
        
        # bbox_dist_list.append(bbox_dist)
        mask_dist_list.append(mask_dist)

    # masks = compute_unique_mask(mask_dist_list, bbox_dist_list)
    # return masks
    return mask_dist_list       
             
        
        
        
def preprocess_arguments(models, arguments):
    # preprocess arguments for the run() method.

    # text to be input.
    phrases = [key for key in arguments['scribble_masks'].keys()]
    phrases = sorted(phrases, key=lambda x: arguments['prompt'].index(x) if x in arguments['prompt'] else float('inf'))
    print(phrases)

    # scribbles to be input.
    scribble_list = []
    for key in phrases:
        mask = arguments['scribble_masks'][key]
        #print(mask.shape)
        #print(mask.detach().numpy().astype(np.uint8).shape)
        mask = Image.fromarray(mask.detach().numpy().astype(np.uint8)[0])
        mask = mask.convert('L')
        mask = np.array(mask)
        mask = np.where(mask < 128, 0, 255).astype(np.float32)
        mask /= 255.
        scribble_list.append(torch.from_numpy(mask).float())

    return {
        'phrases' : [phrases], 
        'scribble_list' : [scribble_list]
    }
   
'''
**OBSOLETE**
sum up different scribbles, but note that they have different pixel values
(this means we merge all the scribbles into one image)
we do not discriminate the classes. Only to tell scribbles apart.
'''   
    
def addup_scribbles(scr_list = None):
    # give them unique idx for different scr
    new = [scr*idx for idx, scr in enumerate(scr_list)]    
    new_scr = torch.zeros_like(new[0])
    for n in new:
        new_scr += n
    return new_scr
        

def plot_scribbles(scr_list = None, ax = None, fig = None):
    
    for idx, scr in enumerate(scr_list):
        row_incr = idx//2; col_incr = idx%2
        ax[2+row_incr,col_incr].set_title(f'scribble {idx}')
        ax[2+row_incr,col_incr].imshow(scr)
    

def plot_results(args = None,
                 cmap = None,
                 images = None,
                 save_path = None):
    
    num_scribbles_half = len(args['scribble_list'][0])//2
    nrows = 3 + num_scribbles_half
    fig, ax = plt.subplots(nrows=nrows, ncols=2)
    ax[0,0].set_title('original image')
    ax[0,1].set_title('GT')
    ax[1,0].set_title('generated image')
    ax[1,1].set_title('prediction')
    ax[0,0].imshow(images['img'])
    ax[0,1].imshow(images['msk_displayed'], cmap = cmap)
    ax[1,0].imshow(images['sample'])
    ax[1,1].imshow(images['pred'], cmap = cmap)
    plot_scribbles(args['scribble_list'][0], ax = ax, fig = fig)
    axis_off(ax = ax)
    plt.savefig(os.path.join(save_path, f"{images['img_name'][0]}.png"))
    print('figure saved.')
    plt.close()
