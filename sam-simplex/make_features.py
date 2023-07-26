import os
import torch
from load_data import CropsDataset
from backbone import load_backbone
from config import config
from tqdm import tqdm


def make_features(images_dir,crops_dir,backbone_path,features_dir):
    dataset = CropsDataset(images_dir,crops_dir)
    model = load_backbone(backbone_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for sample in tqdm(dataset):
        crops, image_name = sample
        crops = crops.to(device).float()
        with torch.inference_mode():
            features = model(crops)
            torch.save(features,features_dir+"/"+image_name+"-FEATURES.pt")

for run in tqdm(os.listdir(config.runs_path)):
    try:
        os.mkdir(config.features_path+"/"+run)
    except FileExistsError:
        pass
    make_features(config.runs_path+"/"+run,
                config.crops_path+"/"+run,
                config.backbone_path,
                config.features_path+"/"+run)                
                    
            
    
    