import pandas as pd
import os
import cv2
import torch
from tqdm import tqdm
from config import config


def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class, n_runs, device='cuda:0'):
    """
    Define runs either randomly or by specifying one sample to insert either as a query or as a support
    Args:
        n_ways: number of classes in the few-shot run
        n_shots: number of samples per class in the few-shot run
        n_queries: number of queries per class in the few-shot run
        num_classes: number of classes in the dataset
        elements_per_class: number of elements in each class
        n_runs: number of runs to generate
    Returns:
        run_classes: classes of the few-shot run
        run_indices: indices of the few-shot run
    """
    run_classes = torch.LongTensor(n_runs, n_ways).to(device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices



def create_runs(runs_path,miniImageNet_path,n_runs,n_ways,n_shot,n_queries,n_classes,elements_per_class):
    os.chdir(miniImageNet_path)
    miniImageNet = pd.concat((pd.read_csv(f) for f in ["train.csv","validation.csv","test.csv"]))

    images_path = miniImageNet_path+"/images"

    os.chdir(runs_path)
    run_classes,run_indices = define_runs(n_ways,
                                        n_shot,
                                        n_queries,
                                        n_classes,
                                        elements_per_class,
                                        n_runs)
    with tqdm(total = n_runs*n_ways) as pbar:
        for i in tqdm(range(n_runs)):
            try:
                os.mkdir("run_"+str(i))
            except FileExistsError:
                pass
            for j in tqdm(range(n_ways)):
                idx_class = run_classes[i,j].tolist()
                for idx in run_indices[i,j].tolist():
                    image_name = miniImageNet["filename"].iloc[600*idx_class+idx]
                    image = cv2.imread(images_path+"/"+image_name)
                    cv2.imwrite("./run_"+str(i)+"/"+image_name, image)
                pbar.update(1)
                
create_runs(config.runs_path,
            config.miniImageNet_path,
            config.n_runs,
            config.n_ways,
            config.n_shot,
            config.n_queries,
            config.n_classes,
            config.elements_per_class)
        
        
    