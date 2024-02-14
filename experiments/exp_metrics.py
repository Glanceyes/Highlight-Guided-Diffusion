from transformers import CLIPModel, CLIPProcessor
import numpy as np  
from PIL import Image
from torchvision import transforms
from datasets.pascal_utils import *
from copy import deepcopy
import cv2
import torch


'''
builds confusion matrix (each elem for #(matched pred and GT pixels))
for single image (batch size = 1)
'''
def build_confusion_mat(GT, pred, class_num = 21):

    print(GT.shape, pred.shape)

    # flatten GT and pred array
    GT = GT.flatten()
    pred = pred.flatten()

    # PASCAL mask includes white outline (IGNORE), so bincount array has 255 in it.
    GT_freq = np.bincount(GT, minlength=class_num)[:class_num]
    pred_freq = np.bincount(pred, minlength=class_num)[:class_num]
    
    # save slots of the matched and unmatched
    # ex) 43 = 21 * 2 + 1, so GT = class 2, pred = class 1 in that case.
    #match_slots = class_num * GT + pred
    


    mask = (GT >= 0) & (GT < class_num)
    conf_mat = np.bincount(
        class_num * GT[mask].astype(int) + pred[mask],
        minlength = class_num ** 2,
    ).reshape(class_num, class_num)
    
    #conf_mat = np.bincount(match_slots, minlength=class_num ** 2).reshape(class_num, class_num)   

    return conf_mat

'''
returns overall class_IOUs and mIOU. (as dict)
'''
def iou(conf_mat, class_num = 21):
    
    IOU = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
    mIOU = np.nanmean(IOU)
    class_IOUs = dict(zip(range(class_num), IOU))

    return {
        'mIOU' : mIOU,
        'class_IOUs' : class_IOUs
    }
    
    
'''
return cosine similarity (scalar) between 
embeddings of text and image according to pretrained CLIP.
args : config, phrases(list), x0(generated image batch)

'''
def T2I_sim(opt, phrases, x0):
    
    clip_ver = opt.clip_vit_ver
    clip = CLIPModel.from_pretrained(clip_ver).to('cpu')
    clip_processor = CLIPProcessor.from_pretrained(clip_ver)
    
    # exclude 'on a background'.
    inputs = clip_processor(text=[phrases[:-4]], images=x0, return_tensors='pt')
    outputs = clip(**inputs)
    
    #The values are cosine similarities between the corresponding image and text features, times 100.
    return outputs.logits_per_image.item() / 100

'''
get the number of scribble pixels in predicted mask of x0. 

Exception if the prediction does not contain any classes.
'''
def get_scr_in_msk_ratio(print_log = True, pred = None, scr_list = None, classdict = None, phrase = None):
    
    # pred : (512,512), scr msk : tensor([64,64]), so resize msk
    for idx, scr in enumerate(scr_list):
        scr_list[idx] = transforms.ToPILImage()(scr_list[idx])
        scr_list[idx] = np.array(scr_list[idx].resize((512, 512), resample=Image.NEAREST))/255 # [0, 1]
    
    # now pred : (512,512), scr msk : (512,512)
    
    print(pred.shape, scr_list[0].shape)
    print(np.unique(pred), np.unique(scr_list))
    
    #pred_classes = np.unique(pred)
    ratio_dict = {}
    scr_dict = {}
    for p_idx, p in enumerate(phrase):
        matched_idx = None
        matched_scr = None
        
        for name, class_idx in classdict.items():
            #print(name, class_idx)
            #print('name:',name,'p:', p)
            # name might be plural.
            if name == p or get_classname_plural(name) == p \
                or name == 'tvmonitor' and (p == 'monitor' or p == 'tv'): # tv monitor not shown as monitor or tv in phrase
                matched_idx = class_idx
                matched_scr = scr_list[p_idx]
                matched_name = name
                scr_dict[class_idx] = scr_list[p_idx] # {0: scr(numpy) of background ..}
                
        
        try:
            matched_idx != None # must match (for exceptions in the dataset, we can exclude it)
        except:
            print('no matched class for', p)
            continue
        
        # for relevant pred, calculate overlap (scr is an array of 0 and 1s)
        
        if matched_scr is not None:
        
            relevant_pred = np.zeros_like(pred)
            relevant_pred[pred == matched_idx] = 1
            intersection = np.sum(relevant_pred * matched_scr)
            scr_pixels = np.sum(matched_scr)
            ratio_dict[matched_name] = intersection / (scr_pixels + 1e-6)
            
            if print_log:
                print(f'for {matched_name} (class {matched_idx}), the scribble pixels inside the mask is {intersection} and the total scribble pixels are {scr_pixels}.')
                
    scr_ratio = 0.0
    for name, ratio in ratio_dict.items():
        if print_log:
            print(f'scr_in_mask_ratio for {name} : {round(ratio_dict[name], 3)}')
            scr_ratio += ratio
    
    if len(ratio_dict.keys()) == 0:
        return 0.0
    return round(scr_ratio / len(ratio_dict.keys()), 3)
    
'''
Give unpropagated scribbles as input. (original_scribbles) -> list
(A['scribble_list'][0])
Also give propagated scribbles as input. (propped_scribbles) -> list
'''    
    
def evaluate_propagation_mIOU(original_scribbles = None,
                              propped_scribbles = None,
                              image = None,
                              seg_model = None,
                              classdict = None,
                              phrase = None,
                              device = None):
    
    epsilon = 1e-6
    propped_list = []
    for idx, (orig_scr, prop_scr) in enumerate(zip(original_scribbles, propped_scribbles)):
        original_scribbles[idx] = transforms.ToPILImage()((original_scribbles[idx]*255).astype(np.uint8))
        original_scribbles[idx] = np.array(original_scribbles[idx].resize((128, 128), resample=Image.NEAREST))/255*2-1 # [-1, 1]
        original_scribbles[idx] = torch.from_numpy(original_scribbles[idx])
        propped = transforms.ToPILImage()(np.array(propped_scribbles[idx].cpu()*255).astype(np.uint8))
        propped_list.append(np.array(propped.resize((512, 512), resample=Image.NEAREST))/255) # [0, 1]
    
    image = np.array(image)
    im_tensor = torch.from_numpy(cv2.resize(image,(128, 128), cv2.INTER_AREA)).permute(2,0,1)
    im_tensor = im_tensor/255.0*2.0 - 1.0 
     
    # keep original dict and propped dict by their classes.
    propped_dict = {}
    orig_dict = {}
    iou_dict = {}

    for p_idx, p in enumerate(phrase):
        matched_idx = None
        matched_scr = None
        
        for name, class_idx in classdict.items():
            # name might be plural.
            if name == p or get_classname_plural(name) == p \
                or name == 'tvmonitor' and (p == 'monitor' or p == 'tv'): # tv monitor not shown as monitor or tv in phrase
                matched_idx = class_idx
                matched_scr = original_scribbles[p_idx]
                prop_scr = propped_list[p_idx]
                matched_name = name
                orig_dict[class_idx] = deepcopy(matched_scr) # {0: scr(numpy) of background ..}
                propped_dict[class_idx] = deepcopy(prop_scr)
            
                if matched_idx is not None and matched_scr is not None \
                    and matched_scr is not None:
                
                    input_batch = torch.zeros((1, 4, 128, 128)).to(device)
                    
                    print(torch.unique(matched_scr))
                    # append scribble to the 4th channel of the batch
                    input_batch[0] = torch.cat((im_tensor, matched_scr[None,...]), dim = 0).to(device)
                    
                    with torch.no_grad():
                        pred = seg_model(input_batch)
                        pred = torch.clamp(pred.squeeze(1),min=0, max=1).permute(1,2,0)
                        # resize predicted mask to original image size
                        masks = cv2.resize(pred.cpu().numpy(),(512, 512))
                        # prediction thresholded (if over 0.5, 1 / otherwise 0)
                        masks[masks >= 0.5] = 1
                        masks[masks != 1] = 0
                        labels = (masks[:,:]*255).astype('uint8')
                        #inference = Image.fromarray(labels)
                        #inference.save(f'pred_{name}.png')
                        propped = (prop_scr[:,:]*255).astype('uint8')
                        propped = Image.fromarray(propped)
                        #propped.save(f'propped_{name}.png')
                        
                        intersection = np.sum(prop_scr * masks)
                        union = np.sum(prop_scr + masks) - intersection
                        iou_dict[name] = intersection / (union + epsilon)
                        print(f'IOU for {name} : {iou_dict[name]}')
                        
    return iou_dict
                    
            

                
                
