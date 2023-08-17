print("Importing libraries...",end=" ")
import sys
path = "/nasbrain/f21lin/venv/venvDuSchlag/lib/python3.10/site-packages/"
if path not in sys.path:
    sys.path.append(path)
from args import args
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import os
import json
print("Done")
class SAMcrop(object):
    def __init__(self):
        print("Loading SAM model...",end=" ")
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_path)
        sam.to(args.device)
        print("Done")
        self.mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=args.points_per_side,stability_score_thresh=0.1) #low stability score to segment more
        self.average_height = 500 #3rd quartile of the height of the images
        self.average_width = 500 #3rd quartile of the width of the images
    def __call__(self,img):
        torch.cuda.empty_cache()
        height, width, _ = img.shape
        aspect_ratio = width/height
        img = torch.from_numpy(img).permute(2,0,1)
        if height > width:
            img = F.resize(img,(self.average_height,int(self.average_height*aspect_ratio)),antialias=True) #Resize the image to avoid memory error
        else : 
            img = F.resize(img,(int(self.average_width/aspect_ratio),self.average_width),antialias=True) #Resize the image to avoid memory error
        img = img.permute(1,2,0).numpy()

        with torch.inference_mode():
            try:
                masks = self.mask_generator.generate(img)
                masks = [mask["segmentation"] for mask in masks]
            except RuntimeError:
                masks = []
        return masks

print("loading masks...")
if os.path.exists(args.masks_dir+"/masks_per_image.json"):
    with open(args.masks_dir+"/masks_per_image.json","r") as f:
        masks_per_image = json.load(f)
    print("Done")
else:
    masks_per_image = {}
    print("No masks found")
print("Number of images with masks:",len(masks_per_image))
all_filename = pd.concat([pd.read_csv(args.dataset_path+"/miniimagenetimages/"+subset)["filename"] for subset in ["train.csv","validation.csv","test.csv"]])
samCrop = SAMcrop()

print("Start generating masks...")
with torch.inference_mode():
    skipped = 0
    for filename in tqdm(all_filename):
        if filename in masks_per_image:
            continue
        masks_per_image[filename] = []
        path = args.dataset_path + "/miniimagenetimages/images/" + filename
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = samCrop(image) #list of masks
        print("Number of masks:",len(masks))
        if len(masks) == 0:
            skipped += 1
        print("Skipped:",skipped)
        masks.append(np.ones((image.shape[0],image.shape[1]))) #Add a mask for the whole image
        for i in range(len(masks)):
            crop_name = filename[:-4] + "_" + str(i)+".pt"
            masks_per_image[filename].append(crop_name)
            torch.save(torch.from_numpy(masks[i].astype(bool)),args.masks_dir+"/"+crop_name)
        with open(args.masks_dir+"/masks_per_image.json","w") as f:
            json.dump(masks_per_image,f)
print("Done")