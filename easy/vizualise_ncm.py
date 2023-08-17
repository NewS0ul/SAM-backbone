import torch
from few_shot_eval import define_runs, crops_ncm
features = torch.load("/nasbrain/f21lin/miniTestingSAMfeatures.pt1")
train_features = {}
val_features = {}
test_features = {}
num_classes = 20
val_classes = 10
for i,class_name in enumerate(features.keys()):
    if i < num_classes:
        train_features[class_name] = features[class_name]
    elif i < num_classes + val_classes:
        val_features[class_name] = features[class_name]
    else:
        test_features[class_name] = features[class_name]
        
val_classes,val_indices = define_runs(5,1,15,10,10*[100])
crops_ncm(train_features,val_features,val_classes,val_indices,1)
