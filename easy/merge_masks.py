from args import args
import json
import torch
from tqdm import tqdm
import numpy as np

k = 3
with open(args.masks_dir+"/masks_per_image.json") as f:
    masks_per_image = json.load(f)

for img,mask_names in tqdm(masks_per_image.items()):
    masks = [torch.load(args.masks_dir+"/"+mask_name) for mask_name in mask_names][:-1] #exclude the last mask, which is the full image
    masks = np.array(masks)
    centroids = [np.array(mask.nonzero().float().mean(axis=0)) for mask in masks]
    centroids = torch.tensor(centroids)
    distances = torch.cdist(centroids,centroids)
    sorted_indices = torch.argsort(distances,dim=1)
    merged_masks = [mask for mask in masks]
    seen_indices = []
    for i in range(2,k+1):
        top_indices =  sorted_indices[:,:i]
        for j in range(len(masks)):
            if list(top_indices[j].numpy()) not in seen_indices:
                merged_masks.append(masks[top_indices[j]].sum())
                seen_indices.append(top_indices[j])
    print(seen_indices)
    print(len(masks),len(merged_masks))
    

    
    