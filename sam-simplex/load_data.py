import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CropsDataset(Dataset):
    def __init__(self,images_dir,crops_dir):
        crops_path = {}        
        for file in os.listdir(images_dir):
            image_name = os.path.splitext(file)[0]
            crops_path[image_name]=[]
        
        for crop_name in os.listdir(crops_dir):
            crops_path[crop_name.split("-")[0]].append(crops_dir+"/"+crop_name)
        self.crops_path = list(crops_path.values())
        self.images_name = list(crops_path.keys())
    def __len__(self):
        return len(self.crops_path)
    
    def __getitem__(self, idx):
        crops = []
        crops_path = self.crops_path[idx]
        for crop_path in crops_path:
            crop = cv2.imread(crop_path)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        crops = np.stack(crops)
        crops = torch.from_numpy(crops)
        crops = crops.permute(0,3,1,2)
        return crops, self.images_name[idx]

        
        
        
        