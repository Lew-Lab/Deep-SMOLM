import comet_ml
import argparse
import collections
#import sys
#import requests
#import socket
import torch
#import mlflow
#import mlflow.pytorch
from data_loader.MicroscopyDataloader_est_experiment import MicroscopyDataLoader_est_experiment
from torch.utils.data import DataLoader
import model.loss as module_loss
import model.metric as module_metric
#import model.model as module_arch
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer_main import *
from trainer.est_main import *
from collections import OrderedDict
import random
import numpy as np
#import pixiedust


def main(config: ConfigParser):
   
    # parameters for the training and testing set

    list_data_batch = np.int_(np.arange(0,config['est_dataset_experiment']['number_dataSet'])+config['est_dataset_experiment']['starting_dataSet'])
    list_data_FoV = np.int_(np.arange(1,config['est_dataset_experiment']['number_FoV']+1))


    for data_batch_cur in list_data_batch:
        for data_FoV_cur in list_data_FoV:
            data_FoV_cur = data_FoV_cur+18
            params_est = {'batch_size':config['est_dataset_experiment']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}
            
            est_file_names = {'noise_image_name':config['est_dataset_experiment']['noise_image_name'],
        'file_folder':config['est_dataset_experiment']['file_folder'],                                   
        'batch_size':config['est_dataset_experiment']['batch_size'],
        'upsampling':config['est_dataset_experiment']['upsampling'],
        'offset_name':config['est_dataset_experiment']['offset_name'],
        'background_name':config['est_dataset_experiment']['background_name'],
        'tophoton':config['est_dataset_experiment']['tophoton'],
        'data_batch_cur':data_batch_cur,
        'data_FoV_cur':data_FoV_cur}




            list_ID_est = np.int_(np.arange(1,config['est_dataset_experiment']['number_images_per_dataset']+1))
            est_set = MicroscopyDataLoader_est_experiment(list_ID_est, **est_file_names)
            est_generator = DataLoader(est_set, **params_est)


            

            # build model architecture, then print to console
            model = getattr(module_arch, config["arch"]["type"])()

            # get function handles of loss and metrics
            
            

            # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())

            optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

            lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)


            trainer = Est(model, optimizer,
                            config=config,
                            valid_data_loader=None,
                            est_data_loader=est_generator)
                                                                                

            #trainer.train()
            trainer.est_experiment(data_batch_cur,data_FoV_cur)
    



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training parameters')
    args.add_argument('-c', '--config', default="config_orientations.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default="/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_89/0622_202722/model_best.pth", type=str,
                      help='path to latest checkpoint (default: None)')
# 0622_202722  train with background
# 0601_231555 train with background subtracted
                    #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_89/0530_235407     #trained with pixOL com using 523/610+unform [-150,150] z distribition;background in two channel don't fixed ratio, intesity is linear distribution
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0217_172852    #trained with pixOL com using 523/610+unform [-100,100] z distribition
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0215_231941    #trained with beads using 523/610 filter
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0211_004116/   #trained with beads using 593/45 filter
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0210_093220
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/train_with_retrieved_pmask_for_pixOL_com_sym_90/0126_012702
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_close2unifrm_sample_M_v2_sym_90/0206_130818
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/intensity_weighted_moments_training_sym_90/0112_220013
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_M_close2uniform_sampled_sym_90/0128_001127
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_corrected_angle_uniform_sampling_sym_90/0128_235439
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_corrected_angle_uniform_sampling_sym_90/0204_110752
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--lamb', '--lamb'], type=float, target=('train_loss', 'args', 'lambda')),
        CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')),
        CustomArgs(['--percent', '--percent'], type=float, target=('trainer', 'percent')),
        CustomArgs(['--asym', '--asym'], type=bool, target=('trainer', 'asym')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',))
    ]
    
    config = ConfigParser.get_instance(args, options)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache() 
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    main(config)

# %%
