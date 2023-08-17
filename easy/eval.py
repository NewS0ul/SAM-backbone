from args import args
from resnet12 import ResNet12
from few_shot_eval import define_runs
import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

print("loading data")
df = pd.concat([pd.read_csv("/nasbrain/f21lin/testing/miniimagenetimages/"+subset+".csv") for subset in ["train","validation","test"]])
with open("/nasbrain/f21lin/masks/masks_per_image.json") as f:
    masks_per_image = json.load(f)
print("Done")
print("loading model")
model = ResNet12(64, (3, 84, 84), 64,True,False)
model.load_state_dict(torch.load("/homes/f21lin/stageFred2A/mini1.pt1", map_location=torch.device("cuda:0")))
model.to("cuda:0")
model.eval()
print("Done")

batch_size = 256


run_classes,run_indices = define_runs(5,1,15,10,10*[100])
all_acc = 0
for run_idx in tqdm(range(run_classes.shape[0]),desc="runs",unit="run"):
    run_img = []
    print("creating run") #creating run by selecting images from df
    for class_idx, class_ in enumerate(run_classes[run_idx]):
        for img_idx in run_indices[run_idx][class_idx]:
            run_img.append(df.iloc[(class_*100+img_idx).item()]["filename"])
    run_masks = []
    for img in run_img:
        run_masks.extend(masks_per_image[img])
    batched_masks = []
    print("Done")
    print("creating batches")
    for i in range(0,len(run_masks),batch_size):
        batched_masks.append(run_masks[i:i+batch_size])
    print("Done")
    print("extracting features") #extracting features from run
    all_features = {}
    with torch.no_grad():
        for batch in tqdm(batched_masks):
            batch_crops = []
            batch_names = []
            for mask_idx, mask in enumerate(batch): #creating crops from batch
                image_name = mask.split("_")[0]+".jpg"
                image_path = "/nasbrain/f21lin/testing/miniimagenetimages/images/"+image_name
                image = Image.open(image_path).convert("RGB")
                image = transforms.ToTensor()(image)
                mask = torch.load("/nasbrain/f21lin/masks/"+mask)
                coords = torch.nonzero(mask)
                bbox = torch.tensor([coords[:,1].min(),coords[:,0].min(),coords[:,1].max(),coords[:,0].max()]) #shape: [x1,y1,x2,y2]
                bbox = [int(x.item()) for x in bbox]
                crop = image[:,bbox[1]:bbox[3],bbox[0]:bbox[2]].float()
                crop = transforms.Resize((92,92))(crop)
                crop = transforms.CenterCrop((84,84))(crop)
                norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                crop = norm(crop)
                batch_crops.append(crop)
                batch_names.append(image_name)
            batch_crops = torch.stack(batch_crops,dim=0)
            batch_crops = batch_crops.to("cuda:0")
            _,batch_features = model(batch_crops)
            for i,name in enumerate(batch_names): #adding features to all_features
                if name not in all_features:
                    all_features[name] = []
                feature = batch_features[i]/torch.norm(batch_features[i],p=2)
                all_features[name].append(feature)
    print("Done")
    print("Creating support and query sets")
    support_features = {}
    query_features = {}
    seen_classes = []
    for img in run_img:
        class_name = img[:9]
        if class_name not in seen_classes:
            seen_classes.append(class_name)
            support_features[img] = all_features[img]
        else:
            query_features[img] = all_features[img]
    support_labels = []
    query_labels = []
    flattenedsupportfeatures = []
    flattenedqueryfeatures = []
    for img in support_features:
        for crop_idx,feature in enumerate(support_features[img]):
            flattenedsupportfeatures.append(feature)  #flattening features
            support_labels.append([img,crop_idx]) #creating labels with image name and crop index
    for img in query_features:
        for crop_idx,feature in enumerate(query_features[img]):
            flattenedqueryfeatures.append(feature) #flattening features
            query_labels.append([img,crop_idx]) #creating labels with image name and crop index
    support_features = torch.stack(flattenedsupportfeatures,dim=0)
    query_features = torch.stack(flattenedqueryfeatures,dim=0)
    support_features = support_features.to("cuda:0")
    query_features = query_features.to("cuda:0")
    print("Done")
    print("Calculating distances")
    distances = torch.cdist(query_features,support_features,p=2) #calculating pairwise distances
    queries_grouped = {}
    for i,[img,crop_idx] in enumerate(query_labels): #grouping queries by image
        if img not in queries_grouped:
            queries_grouped[img] = []
        for j,distance in enumerate(distances[i]):
            queries_grouped[img].append([distance.item(),support_labels[j],crop_idx])
    run_acc = 0
    for query_name,distance_info in queries_grouped.items():
        sorted_distances = sorted(distance_info,key=lambda x: x[0]) #sorting distances
        support_class = sorted_distances[0][1][0][:9]
        run_acc += int(query_name[:9] == support_class)
    run_acc /= len(queries_grouped)
    print(run_acc)
    print("Done")
    all_acc += run_acc
all_acc /= run_classes.shape[0]
print(all_acc)

        
            