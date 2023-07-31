print("Importing libraries...",end=" ")
from args import args
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import os
print("Done")
class SAMcrop(object):
    def __init__(self):
        print("Loading SAM model...",end=" ")
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_path)
        sam.to(args.device)
        print("Done")
        self.mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=args.points_per_side) 
        self.low_memory_mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=args.points_per_side//2)
        self.skipped =0 #number of images skipped because of memory error
    def __call__(self,img):
        torch.cuda.empty_cache()
        with torch.inference_mode():
            try:
                masks = self.mask_generator.generate(img)
            except RuntimeError:
                try:
                    masks = self.low_memory_mask_generator.generate(img)
                except RuntimeError:
                    masks = []
        masks = [mask["segmentation"] for mask in masks]
        return masks

samCrop = SAMcrop()

output_dir = os.listdir(args.masks_dir)
output_dir = set([mask_name.split("_")[0] for mask_name in output_dir])

print("Start cropping...")
with torch.inference_mode():
    with tqdm(total=60001) as pbar:
        with open(args.all_filenamecsv, "r") as f:
            start = 0
            skipped = 0
            for line in f:
                torch.cuda.empty_cache()
                if start<=len(output_dir): #skip images that have already been cropped
                    start+=1
                else:
                    filename = line.split(",")[0]
                    for mask_name in output_dir:
                        if filename[:-4] in mask_name:
                            pbar.update(1)
                            start = 0
                            break
                    if start < 1:
                        start = 1  
                        continue
                    path = args.dataset_path + "/miniimagenetimages/images/" + filename
                    image = cv2.imread(path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    masks = samCrop(image) #list of masks
                    print("Number of masks:",len(masks))
                    print("Skipped:",skipped)
                    if len(masks)==0:
                        skipped+=1
                        masks = [np.ones((image.shape[0],image.shape[1]))] #if no masks are found, we assume the whole image is the object
                        with open(args.skipped_images,"a") as f:
                            f.write(filename+"\n")
                    for i in range(len(masks)):
                        torch.save(torch.from_numpy(masks[i].astype(bool)),
                                   args.masks_dir+"/" + filename[:-4] + "_" + str(i)+".pt")
                pbar.update(1)
    print("Done")