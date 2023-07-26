
import sys
path = "/homes/f21lin/venv/fredVenv/lib/python3.10/site-packages"
if path not in sys.path: sys.path.append(path)
from config import config
import os
from backbone import load_backbone
import cv2
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

def split_support_query(run_path):
    seen_classes =[]
    support = []
    query = []
    for image_name in os.listdir(run_path):
        label = image_name[:9]
        if label not in seen_classes:
            seen_classes.append(label)
            support.append(os.path.join(run_path,image_name))
        else:
            query.append(os.path.join(run_path,image_name))
    return support,query,seen_classes

def calculate_features(image_path,model,device,num_crops=30):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((84,84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    features_list = []
    for _ in range(num_crops):
        tensor = transform(image)
        features = model(tensor.unsqueeze(0).to(device).float())
        features_list.append(features.squeeze(0))
    average_features = calculate_average_features(features_list)  
    return average_features

def calculate_average_features(features):
    stacked_features = torch.stack(features)
    average_features = torch.mean(stacked_features,dim=0)
    return average_features

def preprocess(query_features,average_features):
    centered_features = query_features - average_features
    return centered_features/torch.norm(centered_features)

def nearest_class_classifier(query_features,supports_features,average_features,classes):
    preprocessed_query = preprocess(query_features,average_features)
    preprocessed_support = [preprocess(support_features,average_features) for support_features in supports_features]
    preprocessed_support = torch.stack(preprocessed_support)
    distances = torch.cdist(preprocessed_query.unsqueeze(0).unsqueeze(0),preprocessed_support.unsqueeze(0),p=2).squeeze(0).squeeze(0)
    min_index = torch.argmin(distances)
    return classes[min_index]    

run_path = "/nasbrain/f21lin/runs/run_0"

model = load_backbone(config.backbone_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

acc=0
with torch.inference_mode():
    for run in tqdm(os.listdir(config.runs_path)[:100]):
        torch.cuda.empty_cache()
        run_path = os.path.join(config.runs_path,run)
        support,query,classes = split_support_query(run_path)

        supports_features = [calculate_features(image_path,model,device) for image_path in support]
        average_features = calculate_average_features(supports_features)

        run_acc=0
        for image_path in query:
            query_features = calculate_features(image_path,model,device)
            pred_class=nearest_class_classifier(query_features,supports_features,average_features,classes)
            truth = os.path.basename(image_path)[:9]
            run_acc+=1 if pred_class==truth else 0 
        print("run_acc:",run_acc/len(query))
        acc+=run_acc/len(query)
    print("acc:",acc/10)
