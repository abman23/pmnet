
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")


class PMnet_usc(Dataset):
    def __init__(self, csv_file,
                 dir_dataset="USC/",               
                 transform= transforms.ToTensor()):
        
        self.ind_val = pd.read_csv(csv_file)
        self.dir_dataset = dir_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ind_val)
    
    def __getitem__(self, idx):

        #Load city map
        self.dir_buildings = self.dir_dataset+ "map_complimented/"
        img_name_buildings = os.path.join(self.dir_buildings, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        self.dir_Tx = self.dir_dataset+ "Tx/" 
        img_name_Tx = os.path.join(self.dir_Tx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Tx = np.asarray(io.imread(img_name_Tx))

        #Load Rx (reciever): (not used in our training)
        self.dir_Rx = self.dir_dataset+ "Rx/" 
        img_name_Rx = os.path.join(self.dir_Rx, str((self.ind_val.iloc[idx, 0]))) + ".png"
        image_Rx = np.asarray(io.imread(img_name_Rx))

        #Load Power:
        self.dir_power = self.dir_dataset+ "pmap/" 
        img_name_power = os.path.join(self.dir_power, str(self.ind_val.iloc[idx, 0])) + ".png"
        image_power = np.asarray(io.imread(img_name_power))        

        inputs=np.stack([image_buildings, image_Tx], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            power = self.transform(image_power).type(torch.float32)

        return [inputs , power]

