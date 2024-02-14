class ExpConfig:
    def __init__(self):
        
        # name of the experiment (pass as arg in .sh)
        self.exp_name = "0" 
        
        # Model parameters
        self.ckpt = "../checkpoints/gligen/text-box/diffusion_pytorch_model.bin"
        self.clip_vit_ver = "openai/clip-vit-large-patch14" # same as BoxDiff
        self.sd_weights_path = "../checkpoints/stable_diffusion/SD_input_conv_weight_bias.pth"
        self.device = "cuda"
        
        # Generation parameters (add more detailed hyper-parameters for tuning)
        self.random_seed = 42
        self.prompt = None
        self.batch_size = 1
        self.sampler_type = "PLMSSampler"
        self.step_size = 50
        self.guidance_scale = 7.5
        self.negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        self.loss_type = ['all']
        self.alpha_type = [0.3, 0.0, 0.7]
        
        # Output parameters
        self.output_dir = "outputs"
        self.save_scribble = 0 # do not save scribbles in the generated directory
        
        # Visualization parameters
        self.save_vis = 0
        self.vis_dir = './exp_run_res'
        self.tensorboard = 0
        self.log_dir = "../logs"
        self.vis_cross_res = [8, 16, 32, 64]
        self.vis_self_res = [16]
        
        
        # Scribble parameters
        self.scribble_size = 64
        self.scribble_res = 64
        
        # Interactive segmentation parameters
        self.interactive_seg_ckpt = "../checkpoints/segmentation/amix_150000.pt"
        self.seg_mask_thres = 0.5

        # DeeplabV3+ seg configs for the experiment
        self.seg_model_name = 'deeplabv3plus_resnet101'
        self.deeplab_ckpt = "./segmentation_model/ckpt/best_deeplabv3plus_resnet101_voc_os16.pth"
        self.seg_vis = True
        self.seg_output = './seg_result'
        
        # YOLO configs for the experiments
        self.yolo_weights = './ckpt/yolov7.pt'
        self.yolo_output_dir = './yolo_outputs'
        self.yolo_imgsz = 512
        self.traced = False
        self.yolo_thresholds = [0.65, 0.35]
        
        # Dataset Type
        self.dataset_type = 'PASCAL'
