import comet_ml
import argparse
import collections
import torch
import data_loader as dataLoaderMethod
from torch.utils.data import DataLoader
import model.loss as module_loss
import model.postprocessing_main as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer_main import *
from trainer.est_main import *
from collections import OrderedDict
import random
import numpy as np


def main(config: ConfigParser):
   
    # parameters for the estimation data

    params_est = {'batch_size':config['est_dataset']['batch_size'],'shuffle':False, 'num_workers':config['data_loader']['args']['num_workers']}

    est_file_names = {'noise_image_name':config['est_dataset']['noise_image_name'],
    'GT_list_name':config['est_dataset']['GT_list_name'], 
'file_folder':config['est_dataset']['file_folder'],                                   
'batch_size':config['est_dataset']['batch_size'],
'setup_params':config['microscopy_params']['setup_params']}


    # read the dataloading method
    MicroscopyDataLoader_method =  getattr(dataLoaderMethod, config['est_dataset']['dataloader_method'])    
    if "repeat_frame" in  config['est_dataset']:
        repeatFrame = config['est_dataset']['repeat_frame']
    else:
        repeatFrame=1

    list_ID_est = np.int_(np.arange(1,(config['est_dataset']['number_images']*repeatFrame)+1))
    est_set = MicroscopyDataLoader_method(list_ID_est, **est_file_names,repeat_frame=repeatFrame)

    est_generator = DataLoader(est_set, **params_est)


    # build model architecture, then print to console
    model = getattr(module_arch, config["arch"]["type"])()
    # use the trained Deep-SMOLM that saved in location below 
    model_location = config['Deep-SMOLM_model_trained']

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    

    estimator = Est(model,model_location, optimizer,
                    config=config,
                    est_data_loader=est_generator)
                                                                           
    estimator.est()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='training parameters')
    args.add_argument('-c', '--config', default="config_orientations.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

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
