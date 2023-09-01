import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from args import args
import json
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import shutil
from scipy.ndimage import convolve

df = pd.concat([pd.read_csv(args.dataset_path+args.dataset+"/"+subset+".csv") for subset in ["train","validation","test"]])

try:
    with open(f"{args.masks_dir}/skipped.txt","r") as f:
        skipped_images = f.readlines()
        skipped_images = [image.strip() for image in skipped_images]
except FileNotFoundError:
    skipped_images = []
with open(args.masks_dir+"/masks_info.json","r") as f:
    masks_info = json.load(f)
filtered_masks_info = {}
already_filtered = set([filename.replace(".npz",".jpg") for filename in os.listdir(args.filtered_masks_dir) if filename.replace(".npz",".jpg") in masks_info.keys() or filename.replace(".npz",".jpg") in skipped_images])

print("Already filtered",len(already_filtered))
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') if args.dinov2 else torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
if args.dinov2:
    print("Using DINOv2 (careful, dinov2 has seen a lot of different images)")
else:
    print("Using DINOv1")
model.to(args.device)
model.eval()

with torch.no_grad():
    with tqdm(total=len(df)) as pbar:
        for i in range(len(df)):
            image_name = df.iloc[i]["filename"]
            if image_name in already_filtered:
                continue
            image = Image.open(args.dataset_path+args.dataset+"/images/"+image_name).convert('RGB')
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))(image)
            _,height,width = image.shape
            aspect_ratio = width/height #resize keeping the aspect ratio
            if height > width:
                image = F.resize(image,(args.average_height,int(args.average_height*aspect_ratio)),antialias=True)
            else : 
                image = F.resize(image,(int(args.average_width/aspect_ratio),args.average_width),antialias=True)
            _,height,width = image.shape
            patch_size = 14 if args.dinov2 else 16 # default is 16 for DINOv1
            h,w = height-height%patch_size,width-width%patch_size
            
            h_featmap,w_featmap = h//patch_size,w//patch_size
            image = transforms.Resize((h,w),antialias=True)(image)
            image = image.to(args.device).unsqueeze(0)
            torch.cuda.empty_cache()
            if args.dinov2: #calculate scores with dinov2
                output = model.get_intermediate_layers(x=image,
                                                    reshape=True,
                                                    n = 2,
                                                    return_class_token=True,
                                                    ) # get the output of the n-1 layer
                maps = output[0][0] # get the maps
                maps = maps.reshape((1,maps.shape[1],-1)).permute(0,2,1) # reshape to (1, N, C)
                class_token = output[0][1].reshape((1,-1,1)).permute(0,2,1) # get the class token
                maps = torch.cat((class_token, maps), dim=1) # concatenate the class token with the patches
                qkv = nn.Linear(maps.shape[-1], maps.shape[-1]*3, bias=True) # linear layer to get q,k,v
                qkv.weight.data = model.state_dict()["blocks.11.attn.qkv.weight"].data # initialize with the weights of the n-1 layer of the model
                qkv.bias.data = model.state_dict()["blocks.11.attn.qkv.bias"].data # initialize with the weights of the n-1 layer of the model
                qkv = qkv.to(args.device)
                B, N, C = maps.shape
                qkv_out = qkv(maps).reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)
                head_dim = C // model.num_heads 
                scale = head_dim**-0.5 # scale the q
                q, k = qkv_out[0] * scale, qkv_out[1] # get q and k 

                attn = q @ k.transpose(-2, -1) # Calculate attention
                
                nh = model.num_heads
                attentions = attn[:, :, 0, 1:].reshape(B, nh, -1) # remove the attention of the class token with the patches
                
                w_featmap = w // patch_size
                h_featmap = h // patch_size
                attentions = attentions.reshape(nh,w_featmap,h_featmap).cpu().numpy() # reshape to (num_heads, w_featmap, h_featmap)
                for i in range(len(attentions)): #remove the pixel with the highest attention as it is noisy
                    max_idx = np.unravel_index(attentions[i].argmax(), attentions[i].shape)
                    attentions[i][max_idx] = np.mean(attentions[i]) 
                mean_attention = np.mean(attentions,axis=0) # get the mean attention of the heads
                normalized_attention = (mean_attention - np.min(mean_attention))/(np.max(mean_attention)-np.min(mean_attention)) # normalize the attention map
                normalized_attention = torch.from_numpy(normalized_attention) > 0.75 # threshold the attention map
                
            else: #calculate scores with dinov1
                try:
                    attentions = model.get_last_selfattention(image).cpu()
                except:
                    print(image.shape)
                    exit()
                nh = attentions.shape[1] # number of head

                # we keep only the output patch attention
                attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                threshold = 0.9 #keep xx% of the mass.
                th_attn = cumval > (1 - threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, h_featmap, w_featmap).float()
                th_attn = th_attn.mean(axis=0)
                kernel = np.ones((3,3))
                convolved_attn = convolve(th_attn, kernel, mode='constant', cval=0.0) # denoise the attention map
                convolved_attn = torch.from_numpy(convolved_attn)
                convolved_attn = (convolved_attn-convolved_attn.min())/(convolved_attn.max()-convolved_attn.min())
                convolved_attn = nn.functional.interpolate(convolved_attn.unsqueeze(0).unsqueeze(0),size=(height,width),mode="bicubic")[0][0].cpu() # resize the attention map to the original size
                normalized_attention = convolved_attn > 0.75 # threshold the attention map to denoise it
            scores = []
            
            masks_name = image_name.replace(".jpg",".npz")
            masks = np.load(f"{args.masks_dir}/{masks_name}")
            masks = masks.get("masks")

            filtered_masks_info[image_name] = []
            
            for mask in masks:
                mask = torch.from_numpy(mask)
                area = mask.sum()
                height,width = mask.shape
                if area>0.8*height*width: #if the mask is too big, we skip it because it is probably a background mask
                    score = 0
                else:
                    score = (mask*normalized_attention).sum()/area**0.1 #normalize the score by the area to avoid only keeping the biggest masks (not too much to avoid keeping too small masks)
                scores.append(score)
            scores = np.array(scores)
            scores = (scores-np.min(scores))/(np.max(scores)-np.min(scores)) # normalize the scores to be between 0 and 1
            filtered_indices = np.where(scores>0.5)[0] # keep the masks with score > 0.5 (may need to be tuned)
            filtered_masks = []
            for i in filtered_indices:
                mask = masks[i]
                filtered_masks_info[image_name].append(masks_info[image_name][i])
                filtered_masks.append(mask)
                np.savez_compressed(f"{args.filtered_masks_dir}/{masks_name}",filtered_masks)
            pbar.set_description(f"Filtered {image_name} : {len(filtered_indices)}/{len(masks)}")
            pbar.update(1)
with open(os.path.join(os.path.dirname(args.masks_dir),f"{args.filtered_masks_dir}/masks_info.json"),"w") as f:
    json.dump(filtered_masks_info,f)
                

                
        

