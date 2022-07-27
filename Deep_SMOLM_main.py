import comet_ml
import argparse
import collections
#import sys
#import requests
#import socket
import torch
#import mlflow
#import mlflow.pytorch
from data_loader.MicroscopyDataloader import MicroscopyDataLoader
#from data_loader.MicroscopyDataloader_singleSM import MicroscopyDataLoader_singleSM as MicroscopyDataLoader
from torch.utils.data import DataLoader
import model.loss as module_loss
import model.metric as module_metric
#import model.model as module_arch
import model.model as module_arch
#import model.resnet as module_arch
from parse_config import ConfigParser
from trainer.trainer_main import *
from trainer.est_main import *
from collections import OrderedDict
import random
import numpy as np
#import pixiedust
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main(config: ConfigParser):
   
    # parameters for the training and testing set
    params_train = {'batch_size':config['data_loader']['args']['batch_size'], 'shuffle':config['data_loader']['args']['shuffle'],
'num_workers':config['data_loader']['args']['num_workers']}    

    params_test = {'batch_size':config['data_loader']['args']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}

    # infor for training/testing set
    train_test_file_names = {'noise_image_name':config['training_dataset']['noise_image_name'],
'GT_image_name':config['training_dataset']['GT_image_name'],         'GT_list_name':config['training_dataset']['GT_list_name'], 
'file_folder':config['training_dataset']['file_folder'],                                   
'batch_size':config['data_loader']['args']['batch_size'],
'setup_params':config['microscopy_params']['setup_params'], 'background_name':config['training_dataset']['background_name']}
       
    
   
    number_images = config['training_dataset']['number_images']  
    percentage = config['trainer']['percent']                                                                   
    numb_training = np.floor(number_images*percentage) 
    numb_testing = np.floor(number_images*(1-percentage))                                       
    # instantiate the data class and create a datalaoder for training
    list_ID_train = np.int_(np.arange(1,numb_training+1))
    training_set = MicroscopyDataLoader(list_ID_train, **train_test_file_names)
    training_generator = DataLoader(training_set, **params_train)
    

    
    list_ID_test = np.int_(np.arange(numb_training+1+1,numb_training+numb_testing+1))
    test_set = MicroscopyDataLoader(list_ID_test, **train_test_file_names)
    test_generator = DataLoader(test_set, **params_test)
    batch_size = config['data_loader']['args']['batch_size']
    print(len(training_generator)*batch_size, len(test_generator)*batch_size)



    # build model architecture, then print to console
    model = getattr(module_arch, config["arch"]["type"])()

    # get function handles of loss and metrics
    #

    train_loss_metri = getattr(module_loss, config['train_loss'])
    train_loss_change_metri = getattr(module_loss, config['train_loss_change'])
    test_loss_metri = getattr(module_loss, config['test_loss'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)


    trainer = Trainer(model, optimizer,
                      config=config,
                      data_loader=training_generator,
                      test_data_loader=test_generator,
                      lr_scheduler=lr_scheduler)
                                                                           

    trainer.train()




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training parameters')
    args.add_argument('-c', '--config', default="config_orientations.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

                      #0515_203657
#"/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0527_185236/model_best.pth"
                      #/home/wut/Documents/Deep-SMOLM/data/save/models/training_with_retrieve_pixOL_com_sym_90/0210_011251

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
    #time.sleep(4000)
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
