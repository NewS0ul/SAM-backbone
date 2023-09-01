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
from PIL import Image
print("Done")

class SAMcrop(object):
    def __init__(self):
        print("Loading SAM model...",end=" ")
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_path) #load the model
        sam.to(args.device)
        print("Done")
        self.mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=args.points_per_side,stability_score_thresh=args.stability_score_thresh) #set lower stability score threshold to get more masks
        self.average_height = args.average_height
        self.average_width = args.average_width
    def __call__(self,img):
        torch.cuda.empty_cache() #empty the cache to avoid memory error
        img = Image.fromarray(img)
        width, height = img.size
        aspect_ratio = width/height #resize keeping the aspect ratio
        if height > width:
            img = F.resize(img,(self.average_height,int(self.average_height*aspect_ratio)),antialias=True)
        else : 
            img = F.resize(img,(int(self.average_width/aspect_ratio),self.average_width),antialias=True)
        with torch.inference_mode():
            img = np.array(img)
            masks = self.mask_generator.generate(img)
        return masks

try:
    with open(f"{args.masks_dir}/skipped.txt","r") as f:
        skipped_images = f.readlines()
        skipped_images = [image.strip() for image in skipped_images]
except FileNotFoundError:
    skipped_images = []
try:
    with open(f"{args.masks_dir}/masks_info.json","r") as f:
        masks_info = json.load(f)
except FileNotFoundError:
    masks_info = {}


print("loading masks...")
already_processed = set([filename.replace(".npz",".jpg") for filename in os.listdir(args.masks_dir) 
                         if filename.replace(".npz",".jpg") in masks_info.keys() or filename.replace(".npz",".jpg") in skipped_images]) #get the images already processed
print(f"loaded masks for {len(already_processed)} images")
all_filename = pd.concat([pd.read_csv(f"{args.dataset_path}{args.dataset}/{subset}")["filename"] for subset in ["train.csv","validation.csv","test.csv"]]) #get all the images
samCrop = SAMcrop()
skipped = 0
print("Start generating masks for",args.dataset,"...")
with torch.inference_mode():
    with tqdm(total=len(all_filename)) as pbar:
        for filename in all_filename:
            if filename in already_processed:
                pbar.update(1)
                continue
            path = f"{args.dataset_path}{args.dataset}/images/{filename}"
            image = cv2.imread(path) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                masks = samCrop(image)
            except RuntimeError: #may have memory error when generating masks
                masks = []
                skipped+=1
                with open(f"{args.masks_dir}/skipped.txt","a") as f: #write the images that c
                    f.write(f"{filename}\n")
            if len(masks)>0:
                masks_array = np.array([mask.pop("segmentation") for mask in masks]) #get only the segmentation array
                masks_info[filename] = masks #get other info
            else:
                masks_array = np.array([])
                masks_info[filename] = []
                with open(f"{args.masks_dir}/skipped.txt","a") as f:
                    f.write(f"{filename}\n")
            np.savez_compressed(f"{args.masks_dir}/{filename.replace('.jpg','.npz')}",masks=masks_array) #save the masks
            pbar.set_description(f"{filename} | Number of masks: {len(masks)} | Skipped: {skipped}")
            pbar.update(1)
            with open(f"{args.masks_dir}/masks_info.json","w") as f: #save the info
                json.dump(masks_info,f)
try:
    with open(f"{args.masks_dir}/skipped.txt","r") as f:
        skipped = f.read().splitlines()
        for filename in skipped:
            print(f"skipped {filename}")
        print(f"skipped {len(skipped)} images")
except FileNotFoundError:
    print("No skipped images")