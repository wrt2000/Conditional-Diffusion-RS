import os
import torch
import argparse
import logging
import utils.logger as Logger
import utils.metrics as Metrics
from utils.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import random
import numpy as np
from diffusion import DDPM
from data.RS_dataset import RSDataset, create_dataloader
from utils.metrics import calculate_IS
from utils.fid_eval import calculate_fid_given_dataset

seed = 6666
print('Random seed :{}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True   # improve performace instead of reproduce 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='infer_config.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-infer', '-i', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-steps', '--steps', type=int, default=50)
    parser.add_argument('-eta', '--eta', type=float, default=0.0)
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    val_set = RSDataset(dataset_opt = opt['datasets']['val'])
    val_loader = create_dataloader(val_set,phase='val')
    logger.info('Initial Dataset Finished')

    # model
    diffusion = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(diffusion.__class__.__name__))
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    steps = args.steps
    eta = args.eta
    print(steps,eta)
    result_path = '{}'.format(opt['path']['results'])
    fake_path = os.path.join(result_path,'fake_save')
    mask_path = os.path.join(result_path,'mask_save')
    ori_path = os.path.join(result_path,'ori_save')
    print(result_path)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True,condition_ddim = True,steps = steps,eta = eta)
        visuals = diffusion.get_current_visuals(need_LR=False)

        ori_img = Metrics.tensor2img(visuals['REAL'])  # uint8
        fake_img = Metrics.tensor2img(visuals['FAKE'])  # uint8
        mask_img = Metrics.tensor2img(visuals['MASK'])  # uint8

        # # grid img
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}_fake_process.png'.format(fake_path, current_step, idx))
        # Metrics.save_img(
        #     Metrics.tensor2img(visuals['FAKE'][-1]), '{}/{}_{}_fake.png'.format(fake_path, current_step, idx))  # only save the last
        # Metrics.save_img(
        #     ori_img, '{}/{}_{}_ori.png'.format(ori_path, current_step, idx))


        visuals['FAKE'] = visuals['FAKE'][-val_data['Image'].shape[0]:]
        for i, img_tensor in enumerate(visuals['FAKE']):
            Metrics.save_img(
                Metrics.tensor2img(img_tensor),
                '{}/{}_{}_fake_{}.png'.format(fake_path, current_step, idx, i)
            )

        for i, img_tensor in enumerate(visuals['REAL']):
            Metrics.save_img(
                Metrics.tensor2img(img_tensor),
                '{}/{}_{}_ori_{}.png'.format(ori_path, current_step, idx, i)
            )
        Metrics.save_img(
            mask_img, '{}/{}_{}_mask.png'.format(mask_path, current_step, idx))

        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['FAKE'][-1]), ori_img)
    
    IS = calculate_IS(fake_path)
    paths = [ori_path,fake_path]
    Fid = calculate_fid_given_dataset(paths)
    print("infer: steps = ",steps,",eta = ",eta,";IS: ",IS,",Fid: ",Fid)
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('<steps:{:4d}, eta:{:.2e}> FID: {:.4e} IS: {:.4e}'.format(
        steps, eta, Fid, IS))

    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
        wandb_logger.log_metrics({'validation/IS': IS, 'validation/FID': Fid})
