import numpy as np
import cv2
import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from tqdm import tqdm
from config import config

def make_crops(runs_paths,crops_path,sam_path,n_runs,n_ways,n_queries,n_shot):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    with tqdm(total = n_runs*n_ways*(n_queries+n_shot)) as pbar:
        for run in tqdm(os.listdir(runs_paths)):
            try:
                os.mkdir(crops_path+"/"+run)
            except FileExistsError:
                pass
            for image_name in tqdm(os.listdir(runs_paths+"/"+run)):
                image = cv2.imread(runs_paths+"/"+run+"/"+image_name)
                height,width,_ = image.shape
                n_pixel = height*width
                with torch.inference_mode():
                    masks = mask_generator.generate(image)
                torch.cuda.empty_cache()
                min_area = round(0.01*n_pixel)
                min_size = max(0.05*np.array(image.shape))
                add_size = round(0.05*min(image.shape[0],image.shape[1]))
                for k in range(len(masks)):
                    mask=masks[k]
                    if mask["area"]>= min_area:
                        bbox = mask["bbox"]
                        if bbox[2]>min_size and bbox[3]>min_size:
                            ylim = round(min(bbox[1]+bbox[3]+add_size,image.shape[0]))
                            xlim = round(min(bbox[0]+bbox[2]+add_size,image.shape[1]))
                            crop = image[round(bbox[1]):ylim,round(bbox[0]):xlim,:]
                            cv2.imwrite(crops_path+"/"+run+"/"
                                        +os.path.splitext(image_name)[0]+"-"+str(k)+".jpg"
                                        ,cv2.resize(crop,(224,224)))
                pbar.update(1)
                

make_crops(config.runs_path,
           config.crops_path,
           config.sam_path,
           config.n_runs,
           config.n_ways,
           config.n_queries,
           config.n_shot)