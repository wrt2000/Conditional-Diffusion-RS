import torch
import data as Data
import argparse
import logging
import utils.logger as Logger
import utils.metrics as Metrics
from utils.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from diffusion import DDPM
from data.RS_dataset import RSDataset, create_dataloader
from utils.metrics import calculate_IS
from utils.fid_eval import calculate_fid_given_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='infer_config.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-infer', '-i', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-steps', '--steps', type=int, default=50)
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-eta', '--eta', type=float, default=0.5)

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
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')                                                                                             
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = RSDataset(dataset_opt = opt['datasets']['train'])
            train_loader = create_dataloader(
                train_set, phase='train')
        elif phase == 'val':
            val_set = RSDataset(dataset_opt = opt['datasets']['val'])
            val_loader = create_dataloader(val_set,phase='val')
    logger.info('Initial Dataset Finished')

    # model
    diffusion = DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(diffusion.__class__.__name__))
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            logger.info('Training epoch: {}'.format(current_epoch))
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation (using FID or IS ...)
                if current_step % opt['train']['val_freq'] == 0:
                    idx = 0
                    steps = args.steps
                    eta = args.eta
                    print(steps,eta)
                    result_path = '{}_{}'.format(opt['path']['results'], current_step)
                    fake_path = os.path.join(result_path,'fake_save')
                    mask_path = os.path.join(result_path,'mask_save')
                    ori_path = os.path.join(result_path,'ori_save')
                    print(result_path)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(mask_path, exist_ok=True)
                    os.makedirs(ori_path, exist_ok=True)
                    os.makedirs(fake_path, exist_ok=True)
                    diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['train'], schedule_phase='train')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=True,condition_ddim = True,steps = steps,eta = eta)
                        visuals = diffusion.get_current_visuals(need_LR=False)

                        ori_img = Metrics.tensor2img(visuals['REAL'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['FAKE'])  # uint8
                        mask_img = Metrics.tensor2img(visuals['MASK'])  # uint8
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

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((ori_img, mask_img, fake_img), axis=1)
                            )
                    
                    IS = calculate_IS(fake_path)
                    paths = [ori_path,fake_path]
                    Fid = calculate_fid_given_dataset(paths)
                    logger.info('FID: {:.4e} IS: {:.4e}'.format(Fid, IS))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<steps:{:4d}, eta:{:.2e}> FID: {:.4e} IS: {:.4e}'.format(
                        steps, eta, Fid, IS))
                    
                    tb_logger.add_scalar('validation/IS', IS, current_step)
                    tb_logger.add_scalar('validation/FID', Fid, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({'validation/IS': IS, 'validation/FID': Fid})
                        val_step += 1

                    diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['train'], schedule_phase='train')
                
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:      
        logger.info('Begin Model Inference.')
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

