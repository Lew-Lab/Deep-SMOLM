import numpy as np
import torch
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
from utils import inf_loop
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio
from trainer.train_test_epoches import *

class Trainer:
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model,  optimizer, config, data_loader,
                 test_data_loader=None, lr_scheduler=None, len_epoch=None):
        # pass all variables to self
        #prepare the training device
        # setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self,config['n_gpu'])
        self.model = model.to(self.device)
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.config = config


        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)


        self.optimizer = optimizer
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = self.config.save_dir
        if self.config.resume is not None:
            resume_checkpoint(self,self.config.resume)

        #training metric

        self.test_data_loader = test_data_loader

        self.do_test = self.test_data_loader is not None

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_loss_list: List[float] = []
        self.test_loss_list: List[float] = []


        

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        ifSaveData = self.config["comet"]["savedata"]
        if ifSaveData == True:
            savedata2comet(self,epoch=0)

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):

            result= train_epoch(self,epoch)
            
            log = {'epoch': epoch}
            for key, value in result.items():
                log.update({key:value})
            
            
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                
                if not_improved_count > 2:
                    self.optimizer.param_groups[0]['lr'] = 0.1*self.optimizer.param_groups[0]['lr']

                if self.optimizer.param_groups[0]['lr']<1e-5:
                    break
                    

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            if ifSaveData == True:               
                savedata2comet(self,epoch,best=best)
            plt.close('all')





    


   



    