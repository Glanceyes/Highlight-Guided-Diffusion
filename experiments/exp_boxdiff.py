import os
import torch
import numpy as np
import sys

import warnings
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy
import io
import time
import traceback
import cv2
from functools import partial
from types import SimpleNamespace
from model_comparisons.boxdiff import run_boxdiff

from omegaconf import OmegaConf
from exp_config import ExpConfig
from exp_props import *
from exp_metrics import *
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_model.utils import set_bn_momentum
from ldm.models.segmentation import AbstractUNet
from configs.scribble_seg import get_args

# add more dataset
from datasets.pascal_dataset import PascalDataset

from torch.utils.tensorboard import SummaryWriter
from data import PromptInput
from utils.utils import load_ckpt, set_alpha_scale, alpha_generator
from pytorch_lightning import seed_everything
from segmentation_model import modeling
from run import run

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from losses.loss_config import ScribbleLossConfig
from losses.loss_scheduler import ScribbleLossScheduler
from scribble_propagation import ScribblePropagator, SelfAttnAggregator

from yolo_tester import load_yolov7, infer_yolov7
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split_idx", type=int, default=-1, help="-1 = use all dataset, 0,1,2: choose from three splits")
parser.add_argument("--exp_name", type=str, default='exp1', help='exp name')
args = parser.parse_args()


def preprocess_prompt_input_exp(
        prompt_inputs: PromptInput, 
        text_encoder,
        clip_model,
        clip_processor,
        grounding_tokenizer_input,
        prompt,
        scribble_masks,
    ):
    
    prompt_inputs.valid_check(exp_mode=True)
    
    prompt_inputs.update_scribbles_and_phrases(prompt, scribble_masks, text_encoder)
    
    prompt_inputs.get_tensors_from_lists(text_encoder)
    prompt_inputs.save_scribbles()
    prompt_inputs.save_scribbles(save_individual=True)
    
    prompt_inputs.save_masks()
    prompt_inputs.save_masks(save_individual=True)
    
    prompt_inputs.get_grounding_input(
        clip_model=clip_model,
        clip_processor=clip_processor,
        grounding_tokenizer_input=grounding_tokenizer_input
    )


# supports multiple experiments based on exp_config.py

def run_boxdiff_exp_pipeline():
    # bring experiment configs
    
    opt = dict()
    assert os.path.exists('./exp_opt.json'), "Please specify the path to the config file."
    
    config_file = json.load(open('./exp_opt.json', "r"))
    
    opt = SimpleNamespace(**config_file["config"])
    
    
    #args = ExpConfig()

    seed_everything(opt.seed)
    assert opt.device == "cuda" and torch.cuda.is_available(), "CUDA is not available."
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    
    totensor = transforms.Compose([transforms.ToTensor()])

    # dataset
    if opt.dataset_type == 'PASCAL':    
        dataset = PascalDataset(root_dir='./datasets', 
                                split='trainval', 
                                transform= totensor,
                                to_tensor=True, # mask to tensor
                                require_seg=True,
                                data_split_idx=args.split_idx)
        classdict = dataset.class_to_idx

    elif opt.dataset_type == 'COCO':
        # To be implemented
        # we have to include the scribble annotations in the new COCO dataset.
        pass
    
    # take a peek at the dataset
    #peek_dataset(dataset)

    #dataloader = DataLoader(dataset, batch_size = args.batch_size, collate_fn=collate_fn)
    dataloader = DataLoader(dataset, batch_size = 1)

    
    ###################################################################
    # load checkpoint (diffusion model weights)
    models, config = load_ckpt(opt.ckpt, opt.device)
    config = OmegaConf.create(config)
    text_encoder = models["text_encoder"]

    grounding_tokenizer_input = instantiate_from_config(config["grounding_tokenizer_input"])
    models["model"].grounding_tokenizer_input = grounding_tokenizer_input
    
    clip_ver = text_encoder.tokenizer.name_or_path
    clip_model = CLIPModel.from_pretrained(clip_ver).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_ver)

    # load seg model checkpoint (DeeplabV3+)
    seg_model = modeling.__dict__[opt.seg_model_name](num_classes=21, output_stride=16)
    set_bn_momentum(seg_model.backbone, momentum=0.01)
    checkpoint = torch.load(opt.deeplab_ckpt, map_location=torch.device(opt.device))
    seg_model.load_state_dict(checkpoint["model_state"])
    seg_model.eval()
    seg_model.to(opt.device)
    
    # load interactive model checkpoint (AbstractUNet)
    interactive_seg_model = AbstractUNet(get_args(name='amix')).to(opt.device)

    # load the pretrained weights (cuda)
    seg_ckpt = torch.load(opt.interactive_seg_ckpt, map_location=lambda storage, loc: storage)
    interactive_seg_model.load_state_dict(seg_ckpt['net'])
    
    # load YOLOv7 model checkpoint
    yolo_dict = load_yolov7(dataset)

    # define colormap (pyplot)
    num_classes = 21
    colors = plt.cm.get_cmap('tab20', num_classes) 
    background_color = [0, 0, 0, 1]  # RGBA values for black
    custom_colors = np.vstack([background_color, colors(np.arange(num_classes))])
    # custom ListedColormap. (with the class 0 as background [black])
    cmap = ListedColormap(custom_colors)


    ##############################################################
    # TODO :replace this part with DDP.
    ##############################################################

    print('Generating dataset..')
    # focus on img and mask. (img : image from the dataset, mask : GT)
    # observe how generated images' segmentation pred matches the GT mask.
    conf_mat_init = False

    transform_x0 = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    ############ METRICS #########################################

    # first metric (avg T2I-similarity)
    t2i_sim_sum = 0.0
      
    # TODO : add YOLO metric (precision)
    
    # scr in mask ratio
    scr_ratio = 0.0
    ref_datasize = len(dataset) # when exceptions, exclude such.
    
    # prop iou
    prop_miou = 0.0
    prop_ref_datasize = len(dataset)
    
    os.makedirs(opt.seg_output, exist_ok=True)
    
    exp_log_path = './boxdiff_exp_results_log'
    # save the experiment logs (exceptions caught and the stats of the exp)
    if not os.path.exists(exp_log_path):
        os.mkdir('./boxdiff_exp_results_log')
    # yolo dir for saving the vis
    if not os.path.exists(opt.yolo_output_dir):
        os.mkdir(opt.yolo_output_dir)
    
    # open a file for logging caught exceptions (might cause memory leak.)
    exception_fp = open(os.path.join(exp_log_path, 'exception_log.txt'), 'w')
            
    tensorboard_writer = SummaryWriter(log_dir=opt.log_dir)
    
    num_repeat = opt.n_repeat # 1

    for batch_idx, batch in tqdm(enumerate(dataloader)):
        # gaussian noise X_T for the reverse diffusion process.
        x_T = torch.randn(num_repeat, 4, 64, 64).to(opt.device)
        
        # batch as a dict
        img, img_name, prompt, seg, scribble_masks, coords, original_seg, bb_vis = batch.values()
        
        print('prompt : ', prompt[0])
        
        
        opt.prompt = prompt[0] 
        
        output_dir = os.path.join(opt.output_dir, f'{img_name[0]}')
        save_scribble_dir = os.path.join(opt.save_scribble_dir, f'{img_name[0]}')
        save_mask_dir = os.path.join(opt.save_mask_dir, f'{img_name[0]}')
        vis_dir = os.path.join(opt.vis_dir, f'{img_name[0]}')
        
        prompt_input = PromptInput(
            batch_size=num_repeat,
            prompts=[opt.prompt] * num_repeat,
            output_dirs=[output_dir] * num_repeat,
            save_scribble_dirs=[save_scribble_dir] * num_repeat,
            save_mask_dirs=[save_mask_dir] * num_repeat,
            vis_dirs=[vis_dir] * num_repeat,
            scribble_res=opt.scribble_res,
            device=device
        )
        
        
        preprocess_prompt_input_exp(
            prompt_input, 
            text_encoder, 
            clip_model, 
            clip_processor,
            grounding_tokenizer_input,
            prompt[0],
            scribble_masks
        )
        
        
        # generate x(batch_size) image for each images in PASCAL.
        # generated images will be in ./generated.
        # this will take a long time.
        # with verbose, to display tqdm
        
        # redirect stdout to capture the output
        # also, ignore warnings.
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x0 = torch.from_numpy(np.array(run_boxdiff(prompt_input = prompt_input,
                        seeds = [1],
                    )))# generated batch
        finally:
            # Restore the original stdout
            sys.stdout = original_stdout

        
        print(f'x0 shape : {x0.shape}')
        # need to augment x0 to follow the pretrained model config
        x0 = torch.clamp(x0, -1, 1) * 0.5 + 0.5 # clip x0 to [0, 1]
        
        sample = x0.cpu().numpy()
        sample = Image.fromarray(sample.astype(np.uint8))
        sample_path = os.path.join(output_dir, "{}.png".format(img_name[0]))
        sample.save(sample_path)
        
        x0_aug = transform_x0(x0)
        x0 = x0.detach().cpu()

        
        with torch.no_grad():
            # segmentation model prediction of the generated batch
            pred = seg_model(x0_aug).detach().max(1)[1].cpu().squeeze(0).numpy()
            msk = original_seg.cpu()

            # for visualization (single batch)
            
            x0_sample = (x0.numpy()*255).astype(np.uint8)
            sample = transforms.ToPILImage()(x0_sample)
            img = transforms.ToPILImage()(img[0].cpu())
            msk = msk.squeeze(0).numpy() 
            msk = Image.fromarray(msk.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
            msk = np.array(msk)
            msk_displayed = deepcopy(msk)
            msk_displayed[msk_displayed == 255] = 0
            
            # for YOLO
            img0_yolo = np.array(deepcopy(sample))
            img0_yolo = cv2.cvtColor(img0_yolo, cv2.COLOR_RGB2BGR)
            img_yolo = torch.from_numpy(img0_yolo.transpose(2,0,1)).to(opt.device)
            img_yolo = img_yolo.half()
            img_yolo /= 255.0
            
            # save image args
            image_args = {
                'img' : img,
                'sample' : sample,
                'msk_displayed' : msk_displayed,
                'pred' : pred,
                'img_name' : img_name
            }
            
            # comment this out, we are running out of storage.
            '''
            if opt.seg_vis:
                
                try:
                    plot_results(args = A, 
                             cmap = cmap, 
                             images = image_args,
                             save_path = opt.seg_output)
                except:
                    exception_fp.write(f'Exception from plot_results for {batch_idx} iteration: \n')
                    exception_fp.write(traceback.format_exc())
                    exception_fp.write(f'scribble args : {A}\n')
            '''     

            # compute confusion matrix for a batch (IOU)
            if conf_mat_init == False:
                conf_mat_init = True
                conf_mat = build_confusion_mat(msk, pred)
            else:
                conf_mat += build_confusion_mat(msk, pred) # sum them up
            
            # compute T2I-sim for a batch
            # FIX : Do not load CLIP every time we call the function.
            t2i_sim = T2I_sim(opt, prompt[0], x0)
            t2i_sim_sum += t2i_sim
            print(f'LOG t2i_sim : {round(t2i_sim, 3)}')
            tensorboard_writer.add_scalar(f'T2I_sim_iteration_{batch_idx}', t2i_sim, batch_idx)
            
        
            # YOLO 지금 안하니까 제외했습니다.
            '''
            # YOLOv4 to predict the BBs for x0s (generated images). 
            bb_preds, yolo_vis = infer_yolov7(
                thresholds = opt.yolo_thresholds,
                img = img_yolo,
                img0 = img0_yolo,
                save_vis = True,
                model_dict = yolo_dict,
                prompt = prompt[0],
                args = opt
            ).values()
            
            # if bounding box is detected in x0, save the vis.
            if len(yolo_vis):
                fig, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].set_title('bounding boxes (original image)')
                ax[1].set_title('bounding_boxes (generated image)')
                ax[0].imshow(bb_vis)
                ax[1].imshow(yolo_vis[0])
                ax[0].axis('off')
                ax[1].axis('off')
                plt.savefig(os.path.join(opt.yolo_output_dir, f'{prompt[0]}.png'))
                plt.close()
            '''    
            
            '''
            yolo_stats = None
            
            # if the dataset type is COCO dataset, we can evaluate the YOLO score. (for the ground truth)
            if args.dataset_type == 'COCO':
                anno_json = './coco/annotations/instances_val2017.json'
                coco_pred = np.array(bb_preds[0])
                coco_gt = COCO(anno_json)
                coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                stats = deepcopy(coco_eval)
            ''' 
            
            # average ratio of scribbles' pixels in the predicted mask of x0.
            
            try:
                cur_scr_ratio = get_scr_in_msk_ratio(print_log = True,
                                                pred = pred, 
                                                scr_list = prompt_input.scribble_list[0],
                                                classdict = classdict,
                                                phrase = prompt_input.phrases[0])
            except:
                cur_scr_ratio = 0.0
                ref_datasize -= 1 # exclude this case.
            
            scr_ratio += cur_scr_ratio
            print(f'average scribble in mask ratio : {round(cur_scr_ratio, 3)}')
            
            # evaluate mIOU for scribble masks compared to the pred of interactive seg model and propagated scribbles.
            
            try:
                iou_dict = evaluate_propagation_mIOU(original_scribbles=prompt_input.scribble_list[0],
                                      propped_scribbles=prompt_input.individual_scribbles[0],
                                      seg_model=interactive_seg_model,
                                      classdict=classdict,
                                      phrase=prompt_input.phrases[0],
                                      image=img,
                                      device=device)
                cur_prop_miou = 0.0
                for v in iou_dict.values():
                    cur_prop_miou += v
                if len(iou_dict.values()):
                    cur_prop_miou /= len(iou_dict.values())
                print(f'cur prop_miou : {cur_prop_miou}')
                prop_miou += cur_prop_miou
            
            except:
                cur_prop_miou = 0.0
                prop_ref_datasize -= 1
            
               
    # log results
    # FIX num_repeat to batchsize later.
    # further modification will be output as a file (for each exp configs.)
    
    
    with open(f'./exp_results_log/exp_{args.exp_name}.txt', 'w') as file:
        file.write(f"overall mIOU for PASCAL 2012 Dataset: {round(iou(conf_mat)['mIOU'], 3)}\n")
        file.write(f"class mIOU for PASCAL 2012 Dataset: {iou(conf_mat)['class_IOUs']}\n")
        file.write(f'T2I_sim for PASCAL 2012 Dataset: {round(t2i_sim_sum / (len(dataset) * num_repeat), 3)}\n')
        if ref_datasize:
            file.write(f'scr_ratio for PASCAL 2012 Dataset: {round(scr_ratio / (ref_datasize * num_repeat), 3)}\n')
        else: # division by zero : scr_ratio could not be computed.
            file.write('Failed to compute scr_ratio : ref_datasize = 0.')
        if prop_ref_datasize:
            file.write(f'prop mIOU : {round(prop_miou / (prop_ref_datasize * num_repeat), 3)}\n')
        else:
            file.write('Failed to compute prop_mIOU : prop_ref_datasize = 0.')
            
    # add after YOLO
    '''
    if yolo_stats != None:
        print(stats)
    '''


if __name__ == '__main__':
    run_boxdiff_exp_pipeline()