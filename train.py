from __future__ import print_function, division
import os
import time
#from tkinter import W
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

from tqdm import tqdm
from datetime import datetime
import sys
#import pytorch_model_summary
from torchsummary import summary as summary_

import argparse
import importlib
from utils import L1_loss, MSE, RMSE

import cv2
import matplotlib.pyplot as plt


# RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = 'C:/Users/ABMAN23_ML/Desktop/Joohan/PMNet_Extension_Result'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'


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

def eval_model(model, test_loader, error="MSE", best_val=100, cfg=None):

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

            loss = criterion(preds, targets)
            # NMSE

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)

    if avg_loss < best_val:
        best_val = avg_loss
        # save ckpt
        torch.save(model.state_dict(), f'{RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        print(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        f_log.write(f'[*] model saved to: {RESULT_FOLDER}/{cfg.exp_name}_epoch{cfg.num_epochs}/{cfg.param_str}/model_{best_val:.5f}.pt')
        f_log.write('\n')

    model.train()
    return avg_loss, best_val

def load_config_module(module_name, class_name):
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        return config_class()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, help='Directory where data located.')
    parser.add_argument('-n', '--network', type=str, help='Type of pmnet. pmnet_v1, pmnet_v3')
    parser.add_argument('-c', '--config', type=str, help='Class name in config file.')
    args = parser.parse_args()

    print('start')
    cfg = load_config_module(f'config.{args.config}', args.config)
    print(cfg.get_train_parameters())
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

    # Load dataset
    if cfg.sampling == 'exclusive':
        csv_file = os.path.join(args.data_root,'Data_coarse_train.csv')

        data_train = None
        if 'usc' in args.config.lower():
            from data_loader.loader_USC import PMnet_usc
            num_of_maps = 19016
            ddf = pd.DataFrame(np.arange(1,num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_usc(csv_file = csv_file, dir_dataset=args.data_root)
        elif 'ucla' in args.config.lower():
            from data_loader.loader_UCLA import PMnet_ucla
            num_of_maps = 3776
            ddf = pd.DataFrame(np.arange(1,num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_ucla(csv_file = csv_file, dir_dataset=args.data_root)
        elif 'boston' in args.config.lower():
            from data_loader.loader_Boston import PMnet_boston
            num_of_maps = 3143
            ddf = pd.DataFrame(np.arange(1,num_of_maps))
            ddf.to_csv(csv_file, index=False)
            data_train = PMnet_boston(csv_file = csv_file, dir_dataset=args.data_root)

        dataset_size = len(data_train)

        train_size = int(dataset_size * cfg.train_ratio)
        # validation_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(data_train, [train_size, test_size], generator=torch.Generator(device='cuda'))

        train_loader =  DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=8, generator=torch.Generator(device='cuda'))
    elif cfg.sampling == 'random':
        pass

    # Initialize PMNet and Load pre-trained weights if given.
    if 'pmnet_v1' == args.network:
        from network.pmnet_v1 import PMNet as Model
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,)

        model.cuda()
    elif 'pmnet_v3' == args.network:
        from network.pmnet_v3 import PMNet as Model
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,)

        model.cuda()
    
    # Load pre-trained weights if given
    if hasattr(cfg, 'pre_trained_model'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.pre_trained_model))
        model.to(device)

    # Train.
    # init optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step, gamma=cfg.lr_decay)

    best_val = train(model, train_loader, test_loader, optimizer, scheduler, writer, cfg=cfg)

    print('[*] train ends... ')
    print(f'[*] best val loss: {best_val}')

    f_log.write(f'Train finished at {datetime.today().strftime("%Y%m%d%H%M")}.\n')
    f_log.close()
