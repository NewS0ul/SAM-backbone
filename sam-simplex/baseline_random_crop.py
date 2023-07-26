from config import config
import os
from backbone import load_backbone
import torch
import cv2
import numpy as np
import random
from tqdm import tqdm
import torchvision.transforms as transforms
import itertools

print("libraries imported")

random.seed(42)
num_crops = 10
model = load_backbone(config.backbone_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
similarity = torch.nn.CosineSimilarity(dim=1)
acc = 0
k = 1
print("model loaded")
def random_crop(image_tensor):
    _, height, width = image_tensor.shape
    crop_size = random.randint(20, min(height, width))
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    cropped_image = image_tensor[:, y:y + crop_size, x:x + crop_size]
    return cropped_image

with torch.inference_mode():
    with os.scandir(config.runs_path) as entries:
        for entry in tqdm(itertools.islice(entries,config.n_runs),desc= "Processing runs",unit="run"):
            if entry.is_dir():
                run_directory_path = entry.path
                torch.cuda.empty_cache()
                images_names = os.listdir(run_directory_path)
                seen_classes = []
                support = []
                query = []
                run_acc = 0
                for image_name in images_names:
                    label = image_name[:9]
                    if label not in seen_classes:
                        seen_classes.append(label)
                        support.append(image_name)
                    else:
                        query.append(image_name)

                support_features = []

                for image_name in support:
                    image = cv2.imread(run_directory_path+"/"+image_name)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = np.array(image)
                    image = torch.from_numpy(image).permute(2,0,1)
                    random_crops = []
                    for _ in range(num_crops):
                        crop = random_crop(image)
                        resized_crop = torch.nn.functional.interpolate(crop.unsqueeze(0),size=(84,84))
                        random_crops.append(resized_crop.squeeze(0))
                    image = torch.stack(random_crops).to(device).float()
                    features = model(image).squeeze(0)
                    support_features.append((features,image_name[:9]))


                for image_name in tqdm(query,desc="Processing query",unit="query"):
                    image = cv2.imread(run_directory_path+"/"+image_name)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = np.array(image)
                    image = torch.from_numpy(image).permute(2,0,1)
                    random_crops = []
                    for _ in range(num_crops):
                        crop = random_crop(image)
                        resized_crop = torch.nn.functional.interpolate(crop.unsqueeze(0),size=(84,84))
                        random_crops.append(resized_crop.squeeze(0))            
                    image = torch.stack(random_crops).to(device).float()
                    query_features = model(image)
                    distances = []
                    for support_feature,support_label in support_features:
                        distance = 1-similarity(support_feature,query_features)
                        distances = distances + [(d,support_label) for d in distance]
                    sorted_distances = sorted(distances)
                    class_weight = {}
                    for i in range(k):
                        distance,label = sorted_distances[i]
                        if label in class_weight:
                            class_weight[label]+=distance
                        else:
                            class_weight[label]=distance
                    min_distance = 10e5
                    for label in class_weight:
                        if class_weight[label]<min_distance:
                            min_distance = class_weight[label]
                            pred_class= label
                    run_acc+= image_name[:9]==pred_class
                run_acc = run_acc/len(query)
                acc+=run_acc
acc = acc/config.n_runs
print(acc)