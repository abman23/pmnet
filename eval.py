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
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.enabled

from tqdm import tqdm
from datetime import datetime
import sys

import argparse
import importlib
import json
from utils import L1_loss, MSE, RMSE

import cv2
import matplotlib.pyplot as plt


# RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = 'results'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'


def eval_model(model, test_loader, error="MSE", cfg=None, infer_img_path=''):

    # Set model to evaluate mode
    model.eval()

    n_samples = 0
    avg_loss = 0

    # check dataset type
    pred_cnt=1 # start from 1
    for inputs, targets in tqdm(test_loader):
        image_buildings, tx_map = inputs[:, 0, :, :], inputs[:, 1, :, :]
        # BREAK
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
            output = np.zeros((256, 256, 3), dtype=np.uint8)

            # inference image
            if infer_img_path!='':
                for i in range(len(preds)):
                    output = np.stack([preds[i][0].cpu().detach().numpy()]*3, axis=2)
                    output[:, :, 0][tx_map[i] == 1] = 1
                    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

                    img_name=os.path.join(infer_img_path,f'{args.model_to_eval}_inference',f'{pred_cnt}.png')
                    plt.savefig(img_name)
                    pred_cnt+=1
                    if pred_cnt%100==0:
                        print(f'{img_name} saved')

            loss = criterion(preds, targets)
            # NMSE

            avg_loss += (loss.item() * inputs.shape[0])
            n_samples += inputs.shape[0]

    avg_loss = avg_loss / (n_samples + 1e-7)

    return avg_loss

def load_config_module(module_name, class_name):
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)
        return config_class()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, help='Directory where data located.')
    parser.add_argument('-n', '--network', type=str, help='Type of pmnet. pmnet_v1, pmnet_v3')
    parser.add_argument('-m', '--model_to_eval', type=str, help='Pretrained model to evaluate.')
    parser.add_argument('-c', '--config', type=str, help='Class name in config file.')

    args = parser.parse_args()

    print('start')
    cfg = load_config_module(f'config.{args.config}', args.config)
    print(cfg.get_train_parameters())
    cfg.now = datetime.today().strftime("%Y%m%d%H%M") # YYYYmmddHHMM

    # Load dataset
    if cfg.sampling == 'exclusive':
        # csv_file = os.path.join(args.data_root,'Data_coarse_train.csv')

        data_train = None
        if 'usc' in args.config.lower():
            from dataloader.loader_USC import PMnet_usc
            data_train = PMnet_usc(dir_dataset=args.data_root)
        elif 'ucla' in args.config.lower():
            from dataloader.loader_UCLA import PMnet_ucla
            data_train = PMnet_ucla(dir_dataset=args.data_root)
        elif 'boston' in args.config.lower():
            from dataloader.loader_Boston import PMnet_boston
            data_train = PMnet_boston(dir_dataset=args.data_root)

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
        from models.pmnet_v1 import PMNet as Model
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=16,)

        model.cuda()
    elif 'pmnet_v3' == args.network:
        from models.pmnet_v3 import PMNet as Model
        # init model 
        model = Model(
            n_blocks=[3, 3, 27, 3],
            atrous_rates=[6, 12, 18],
            multi_grids=[1, 2, 4],
            output_stride=8,)

        model.cuda()
    
    # Load pre-trained weights to evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_to_eval))
    model.to(device)

     # create inference images directory if not exist
    os.makedirs(os.path.join(os.path.split(args.model_to_eval)[-2], f'{args.model_to_eval}_inference'), exist_ok=True)

    result = eval_model(model, test_loader, error="RMSE", cfg=None, 
                        infer_img_path=os.path.split(args.model_to_eval)[-2])
    result_json_path = os.path.join(os.path.split(args.model_to_eval)[-2], 'result.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print('Evaluation score(RMSE): ', result)