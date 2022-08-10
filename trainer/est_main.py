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
from trainer.est_epoches import *

class Est:
    """
    Est class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, model_loc, optimizer,config, est_data_loader=None):
        # pass all variables to self
        #prepare the training device
        # setup GPU device if available, move model into configured device
        self.device, device_ids = prepare_device(self,config['n_gpu'])
        self.model = model.to(self.device)
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.optimizer = optimizer
        self.config = config


        # configuration to save the estimated results
        self.checkpoint_dir = self.config.save_dir
        #load trained model (Deep-SMOLM) from saved location
        resume_trained_model(self,model_loc)

        #training metric
        self.est_data_loader= est_data_loader
        self.do_estimation = self.est_data_loader is not None




        

    def est(self):
        """
        estimation
        """

        if self.do_estimation:
            est,GT_list_all = est_epoches(self)
            sio.savemat(str(self.checkpoint_dir / self.config["est_dataset"]["save_name"]),{'est':est, 'GT_list_all':GT_list_all}) 



    def est_experiment(self,data_batch_cur,data_FoV_cur):
        """
        estimation
        """

        if self.do_estimation:
            est,GT_list_all = est_epoches(self)
            sio.savemat(str(self.checkpoint_dir)+ '/'+self.config["est_dataset_experiment"]["save_name"]+'_'+str(data_batch_cur)+'_'+str(data_FoV_cur)+'th_FoV.mat',{'est':est, 'GT_list_all':GT_list_all}) 

            






    


   



    