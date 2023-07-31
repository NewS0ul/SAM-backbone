from args import args
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
class MaskDataset():
    def __init__(self,transform=[]):
        masks_names = os.listdir(args.masks_dir)
        self.masks_paths = [os.path.join(args.masks_dir,name) for name in masks_names]
        self.images_path = args.dataset_path + "/miniimagenetimages/images/"
    def __len__(self):
        return len(self.masks_paths)
    def __getitem__(self,idx):
        mask_path = self.masks_paths[idx]
        image_name = mask_path.split("/")[-1].split("_")[0]
        image_path = self.images_path + image_name + ".jpg"
        mask = torch.load(mask_path)
        image = cv2.imread(image_path)
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Get coordinates of mask
        coords = np.where(mask==1) 
        #Get bounding box
        bbox = [min(coords[1]),min(coords[0]),max(coords[1]),max(coords[0])] 
        #If bounding box is too small, expand it
        height,width,_ = image.shape
        crop_width,crop_height = bbox[2]-bbox[0],bbox[3]-bbox[1]
        if crop_width/width < args.crop_threshold:
            crop_width = width*args.crop_threshold
        if crop_height/height < args.crop_threshold:
            crop_height = height*args.crop_threshold
        #Crop image
        crop = image[round(bbox[1]):round(bbox[1]+crop_height),round(bbox[0]):round(bbox[0]+crop_width),:]
        return crop,image_name

def load_crops():
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((84,84)),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    ])
    dataset = MaskDataset(transforms=transforms)
    return torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
    
    
    
    

        
        
        