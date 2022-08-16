import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np
import torch
from logger import CometWriter

def prepare_device(self, n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        self.logger.warning("Warning: There\'s no GPU available on this machine,"
                            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def save_checkpoint(self, epoch, save_best=False):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__

    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'monitor_best': self.mnt_best
    }
    # filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    # torch.save(state, filename)
    # self.logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth at: {} ...".format(best_path))


def resume_checkpoint(self, resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    self.logger.info("Loading checkpoint: {} ...".format(resume_path))
    if self.device.type=='cpu':
        checkpoint = torch.load(resume_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1
    self.mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    if checkpoint['arch'] != self.config['arch']:
        self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                            "checkpoint. This may yield an exception while state_dict is being loaded.")
    self.model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
    #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
    #                         "Optimizer parameters not being resumed.")
    # else:
    self.optimizer.load_state_dict(checkpoint['optimizer'])

    self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


def resume_trained_model(self, resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    self.logger.info("Loading checkpoint: {} ...".format(resume_path))
    if self.device.type=='cpu':
        checkpoint = torch.load(resume_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1
    self.mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    if checkpoint['arch'] != self.config['arch']:
        self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                            "checkpoint. This may yield an exception while state_dict is being loaded.")
    self.model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
    #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
    #                         "Optimizer parameters not being resumed.")
    # else:
    self.optimizer.load_state_dict(checkpoint['optimizer'])

    self.logger.info("Checkpoint loaded. Estimation using model saved in {}".format(resume_path))


def savedata2comet(self,epoch,fig_1SM=None, fig_angles_1SM=None, fig_2SMs=None, best=None):
    if epoch==0:
        self.writer = CometWriter(
        self.logger,
        project_name = "deep-smolm",
        experiment_name = self.config['exper_name'],
        api_key = self.config['comet']['api'],
        log_dir = self.config.log_dir,
        offline = self.config['comet']['offline'])

        self.writer.log_hyperparams(self.config.config) 
    else:

        if epoch % self.save_period == 0:
                save_checkpoint(self,epoch, save_best=best)
        # if (epoch - 1) % 1 == 0:
        #     figure_name = f"Epoch {epoch} validation result 1SM"
        #     self.writer.add_plot(figure_name, fig_1SM)
        #     figure_name = f"Epoch {epoch} validation result 1SM angles"
        #     self.writer.add_plot(figure_name, fig_angles_1SM)
        # if (epoch - 1) % 1 == 0:
        #     figure_name = f"Epoch {epoch} validation result 2SMs"
        #     self.writer.add_plot(figure_name, fig_2SMs)


def progress_bar(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.data_loader, 'n_samples'):
        current = batch_idx * self.data_loader.batch_size
        total = self.data_loader.n_samples
    else:
        current = batch_idx
        total = self.len_epoch
    return base.format(current, total, 100.0 * current / total)
        
