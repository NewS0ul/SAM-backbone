import os
import torch
from tqdm import tqdm

def knn_on_features(query_features_path, support_features_paths,k,distance="euclidean"):
    if distance == "cosine":
        similarity = torch.nn.CosineSimilarity(dim=0)
    query = torch.load(query_features_path)
    distances = []
    for f in support_features_paths:
        support = torch.load(f)
        for i in range(support.shape[0]):
            for j in range(query.shape[0]):
                if distance =="cosine":
                    distance = 1-similarity(support[i],query[j])
                else:
                    distance = torch.norm(support[i]-query[j],p=2)
                distances.append((distance,os.path.basename(f)[:9]))
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
    return pred_class,min_distance

def accuracy(query_features_paths,support_features_paths,k):
    acc = 0
    for query_feature_path in tqdm(query_features_paths):
        predicted_class, _ = knn_on_features(query_feature_path,support_features_paths,k)
        acc += predicted_class== os.path.basename(query_feature_path)[:9]
    return acc/len(query_feature_path)
        