import numpy as np
import torch
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from numpy import inf
from trainer.trainer_utils import *
from model.metric import postprocessingv2
import model.loss as module_loss




def train_epoch(self, epoch):
    print(epoch)
           
    self.model.train()

    total_loss = 0
    train_criterion = getattr(module_loss, self.config['train_loss'])
    train_criterion_change = getattr(module_loss, self.config['train_loss_change'])
    for batch_idx, (data, label,indx) in enumerate(self.data_loader):
        
        data, label = data.to(self.device), label.to(self.device)       
        output = self.model(data)

       
        if self.config['change_traing_loss_function']:
            if epoch<self.config['epoch_change']:
                loss,loss_track = train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization
            else:
                loss,loss_track = train_criterion_change(output, label, self.config["scaling_factor"]) # Chaged for only localization
        else:    
                loss,loss_track = train_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        #est = postprocessingv2(self.config, output, 1000*label[:,6:12,:,:], indx)

        ifSaveData = self.config["comet"]["savedata"]
        if ifSaveData == True:
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, epoch=epoch)
            self.writer.add_scalar({'loss': loss.item()})
            self.writer.add_scalar({'loss_MSE': loss_track[0]})
            self.writer.add_scalar({'loss_I': loss_track[1]})
            self.writer.add_scalar({'loss_XX': loss_track[2]})
            self.writer.add_scalar({'loss_YY': loss_track[3]})
            self.writer.add_scalar({'loss_ZZ': loss_track[4]})
            self.writer.add_scalar({'loss_XY': loss_track[5]})
            self.writer.add_scalar({'loss_XZ': loss_track[6]})
            self.writer.add_scalar({'loss_YZ': loss_track[7]})

        
        self.train_loss_list.append(loss.item())
        total_loss += loss.item()

        # if batch_idx == self.len_epoch:
        #     break


    log = {
        'loss': total_loss / self.len_epoch,
        'learning rate': self.lr_scheduler.get_last_lr()
    }
    
    if self.do_test:
        test_log = test_epoch(self, epoch)
        log.update(test_log)
    else:
        test_log = None


    if self.lr_scheduler is not None:
        self.lr_scheduler.step()

    
    return log



def test_epoch(self, epoch):
    print(epoch)

    self.model.eval()
    total_test_loss = 0

    test_criterion = getattr(module_loss, self.config['test_loss'])
    

    with torch.no_grad():
        
        for batch_idx, (data, label,idx) in enumerate(self.test_data_loader):
            
            data, label = data.to(self.device), label.to(self.device)
            output = self.model(data)
            
            #test_criterion = getattr(module_loss, self.config['test_loss'])
            loss,loss_track = test_criterion(output, label, self.config["scaling_factor"]) # Chaged for only localization

            ifSaveData = self.config["comet"]["savedata"]

            self.test_loss_list.append(loss.item())
            total_test_loss += loss.item()
            
    loss = total_test_loss / len(self.test_data_loader)
    save_output = [loss]

    if ifSaveData == True:
        self.writer.set_step(epoch, epoch=epoch, mode = 'test')
        self.writer.add_scalar({'loss': loss})

    return {
        'test_loss': loss
    }


# def warmup_epoch(self, epoch):
#     total_loss = 0
#     self.model.train()

#     data_loader = self.data_loader#self.loader.run('warmup')


#     with tqdm(data_loader) as progress:
#         for batch_idx, (data, label, _, indexs , _) in enumerate(progress):
#             progress.set_description_str(f'Warm up epoch {epoch}')

#             data, label = data.to(self.device), label.long().to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             out_prob = torch.nn.functional.softmax(output).data.detach()

#             self.train_criterion.update_hist(indexs.cpu().detach().numpy().tolist(), out_prob)

#             loss = torch.nn.functional.cross_entropy(output, label)

#             loss.backward() 
#             self.optimizer.step()

#             #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             #self.writer.add_scalar('loss', loss.item())
#             self.train_loss_list.append(loss.item())
#             total_loss += loss.item()


#             if batch_idx % self.log_step == 0:
#                 progress.set_postfix_str(' {} Loss: {:.6f}'.format(
#                     progress_bar(self,batch_idx),
#                     loss.item()))
#                 #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#     if hasattr(self.data_loader, 'run'):
#         self.data_loader.run()
#     log = {
#         'loss': total_loss / self.len_epoch,
#         'noise detection rate' : 0.0,
#         'learning rate': self.lr_scheduler.get_lr()
#     }

#     if self.do_validation:
#         val_log = valid_epoch(self,epoch)
#         log.update(val_log)
#     if self.do_test:
#         test_log, test_meta = test_epoch(self,epoch)
#         log.update(test_log)
#     else: 
#         test_meta = [0,0]

#     return log



