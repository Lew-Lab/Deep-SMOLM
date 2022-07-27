import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from typing import List
from torchvision.utils import make_grid
#from base import BaseTrainer
from utils import inf_loop
import matplotlib.pyplot as plt
import sys
from numpy import inf
from trainer.trainer_utils import *
from logger import CometWriter
import scipy.io as sio
#from train_test_epoches import *
from trainer.est_epoches import *

class Est:
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, optimizer,config,valid_data_loader=None, est_data_loader=None):
        # pass all variables to self
        #prepare the training device
        # setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self,config['n_gpu'])
        self.model = model.to(self.device)
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        #self.val_result = [] # Stack all validation result later

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.optimizer = optimizer
        self.config = config


        # configuration to monitor model performance and save best

        self.checkpoint_dir = self.config.save_dir
        if self.config.resume is not None:
            resume_checkpoint(self,self.config.resume)

        else:
            NotImplementedError('trained model is required for esting/validation')

        #training metric
        
        self.valid_data_loader = valid_data_loader
        self.est_data_loader= est_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_estimation = self.est_data_loader is not None




        

    def est(self):
        """
        Full training logic
        """
        not_improved_count = 0
        #ifSaveData = self.config["comet"]["savedata"]


        if self.do_validation:
            orien_est_all,orient_GT_all, M_est_all,bias_con_all,data,label,output,I_GT_all,I_est_all = valid_epoch(self, 1)

            sio.savemat(str(self.checkpoint_dir / self.config["validation_dataset"]["save_name"]),{'orien_est_all':orien_est_all,
            'orient_GT_all':orient_GT_all,'M_est_all':M_est_all,'bias_con_all':bias_con_all,'data':data,'label':label,'output':output,
            'I_GT_all':I_GT_all,'I_est_all':I_est_all}) 

        if self.do_estimation:
            est,GT_list_all = est_withou_GT(self)
            sio.savemat(str(self.checkpoint_dir / self.config["est_dataset"]["save_name"]),{'est':est, 'GT_list_all':GT_list_all}) 



    def est_experiment(self,data_batch_cur,data_FoV_cur):
        """
        Full training logic
        """
        not_improved_count = 0
        #ifSaveData = self.config["comet"]["savedata"]


        if self.do_validation:
            orien_est_all,orient_GT_all, M_est_all,bias_con_all,data,label,output,I_GT_all,I_est_all = valid_epoch(self, 1)

            sio.savemat(str(self.checkpoint_dir / self.config["validation_dataset"]["save_name"]),{'orien_est_all':orien_est_all,
            'orient_GT_all':orient_GT_all,'M_est_all':M_est_all,'bias_con_all':bias_con_all,'data':data,'label':label,'output':output,
            'I_GT_all':I_GT_all,'I_est_all':I_est_all}) 

        if self.do_estimation:
            est,GT_list_all = est_withou_GT(self)
            sio.savemat(str(self.checkpoint_dir)+ '/'+self.config["est_dataset_experiment"]["save_name"]+'_'+str(data_batch_cur)+'_'+str(data_FoV_cur)+'th_FoV.mat',{'est':est, 'GT_list_all':GT_list_all}) 

            






    


   



    