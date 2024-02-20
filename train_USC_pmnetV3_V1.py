from __future__ import print_function, division
import os
import time
from tkinter import W
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled

from network.pmnet_v3 import PMNet as Model
from config.config_USC_pmnetV3_V2 import config_USC_pmnetV3_V2 as Config

from data_loader.loader_USC import PMnet_usc

from tqdm import tqdm
from datetime import datetime
import sys
#import pytorch_model_summary
from torchsummary import summary as summary_

import cv2
import matplotlib.pyplot as plt


# RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = 'C:/Users/ABMAN23_ML/Desktop/Joohan/PMNet_Extension_Result'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'


def L1_loss(pred, target):
  loss = nn.L1Loss()(pred, target)
  return loss

def MSE(pred, target):
  loss = nn.MSELoss()(pred, target)
  return loss
  
def RMSE(pred, target, metrics=None):
  loss = (((pred-target)**2).mean())**0.5
#   metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
  return loss


def train(model, train_loader, test_loader, optimizer, scheduler, writer, cfg=None):  
    best_loss = 1e10
    best_val = 100
    count = 0

    # looping over given number of epochs
    for epoch in range(cfg.num_epochs):
        tic = time.time()
        
        model.train()

        for inputs, targets in tqdm(train_loader):
            count += 1

            inputs = inputs.cuda()
            targets = targets.cuda()
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = MSE(preds, targets)

            loss.backward()
            optimizer.step()

            # tensorboard logging
            writer.add_scalar('Train/Loss', loss.item(), count)

            if count % 100 == 0:
                print(f'Epoch:{epoch}, Step:{count}, Loss:{loss.item():.6f}, BestVal:{best_val:.6f}, Time:{time.time()-tic}')
            tic = time.time()

        print(f"lr: {optimizer.param_groups[0]['lr']} at epoch {epoch}")
        scheduler.step()
        if epoch%cfg.val_freq==0:
          val_loss, best_val = eval_model(model, test_loader, error='MSE', best_val=best_val, cfg=cfg)
          writer.add_scalar('Val/Loss', val_loss, count)
        
    return best_val

def eval_model(model, test_loader, error="MSE", best_val=100, cfg=None, eval_mode=False, infer_img_path=''):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    pred_cnt=1 # start from 1
    for inputs, targets in tqdm(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()


        with torch.set_grad_enabled(False):
            if error == "MSE":
                criterion = MSE
            elif error == "RMSE":
                criterion = RMSE
            elif error == "L1_loss":
                criterion = L1_loss

            preds = model(inputs)  
            preds = torch.clip(preds, 0, 1)

            # inference image
            if infer_img_path!='':
                for i in range(len(preds)):
                    plt.imshow(cv2.cvtColor(preds[i][0].cpu().detach().numpy(), cv2.COLOR_BGR2RGB))

                    img_name=os.path.join(infer_img_path,'inference_images',f'{pred_cnt}.png')
                    plt.savefig(img_name)
                    pred_cnt+=1
                    if pred_cnt%100==0:
                        print(f'{img_name} saved')

            loss = criterion(preds, targets) 
            # NMSE

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)

    if avg_loss < best_val and eval_mode==False:
        best_val = avg_loss
        # save ckpt
        torch.save(model.state_dict(), f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        print(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        f_log.write(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        f_log.write('\n')

    model.train()
    return avg_loss, best_val

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def helper(cfg,writer,phase='train', data_root = 'C:/Users/ABMAN23_ML/Desktop/Joohan/UCLA/', model_to_eval=''):

    # data_root = 'C:/Users/ABMAN23_ML/Desktop/Joohan/USC/'
    
    if cfg.sampling == 'exclusive':
        ddf = pd.DataFrame(np.arange(1,19016))
        ddf.to_csv('C:/Users/ABMAN23_ML/Desktop/Joohan/USC/Data_coarse_train.csv',index=False)

        data_usc_train = PMnet_usc(csv_file = 'C:/Users/ABMAN23_ML/Desktop/Joohan/USC/Data_coarse_train.csv', dir_dataset=data_root)

        dataset_size = len(data_usc_train)

        train_size = int(dataset_size * cfg.train_ratio)
        # validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(data_usc_train, [train_size, test_size], generator=torch.Generator(device='cuda'))

        train_loader =  DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
    elif cfg.sampling == 'random':
        pass
    
    
    if phase=='train':
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,)

        model.cuda()

        print(summary_(model,  (2, 256, 256),batch_size=cfg.batch_size))

        # init optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

        best_val = train(model, train_loader, test_loader, optimizer, scheduler, writer, cfg=cfg)

        print('[*] train ends... ')
        print(f'[*] best val loss: {best_val}')


    '''
    Evaluation the model region
    '''
    if phase =='eval':
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,)
        model.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_to_eval))
        model.to(device)

        # create inference images directory if not exist
        createDirectory(os.path.join(os.path.split(model_to_eval)[0], 'inference_images'))

        result = eval_model(model, test_loader, error="RMSE", best_val=100, cfg=None, eval_mode=True, 
                            infer_img_path=os.path.split(model_to_eval)[0])
        print('Evaluation score(RMSE): ', result)
        '''End evaluation region'''

def evaluation(data_root):

    cfg = Config()
    helper(cfg,writer=None, data_root=data_root)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Insufficient arguments")
        sys.exit()

    phase=sys.argv[1] # 'train' or 'eval'
    data_root = sys.argv[2]
    model_to_eval = sys.argv[3]
    
    print('start')
    cfg = Config()
    cfg.now = datetime.today().strftime("%Y%m%d%H%M") # YYYYmmddHHMM

    
    cfg.param_str = f'{cfg.batch_size}_{cfg.lr}_{cfg.lr_decay}_{cfg.step}'
    os.makedirs(TENSORBOARD_PREFIX, exist_ok=True)
    os.makedirs(f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}', exist_ok=True)   

    print('cfg.exp_name: ', cfg.exp_name)
    print('cfg.now: ', cfg.now)
    for k, v in cfg.get_train_parameters().items():
      print(f'{k}: {v}')
    print('RESULT_FOLDER: ', RESULT_FOLDER)
    print('cfg.param_str: ', cfg.param_str)

    # write config on the log file
    f_log = open(f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/train.log', 'w')
    f_log.write(f'Train started at {cfg.now}.\n')
    for k, v in cfg.get_train_parameters().items():
      f_log.write(f'{k}: {v}\n')


    writer = SummaryWriter(log_dir=f'{TENSORBOARD_PREFIX}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}')

    helper(cfg,writer,phase, data_root, model_to_eval)

    f_log.write(f'Train finished at {datetime.today().strftime("%Y%m%d%H%M")}.\n')
    f_log.close()

