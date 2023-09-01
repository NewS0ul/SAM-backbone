import torch
import numpy as np
from args import args
from utils_0 import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
import json
from PIL import Image
from tqdm import tqdm
n_runs = args.n_runs
batch_few_shot_runs = args.batch_fs
#assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx): #get the right data and labels for the batch
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs] #(batch_few_shot_runs, n_ways)
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs] #(batch_few_shot_runs, n_ways, n_samples)
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2]).to(args.device) #(batch_few_shot_runs, n_ways, data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2]).to(args.device) #(batch_few_shot_runs, n_ways, n_samples, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1).to(args.device) #(batch_few_shot_runs, data.shape[0], data.shape[1], data.shape[2])
    cclasses = torch.gather(datas, 1, run_classes) #(batch_few_shot_runs, n_ways, data.shape[1], data.shape[2])
    res = torch.gather(cclasses, 2, run_indices) #(batch_few_shot_runs, n_ways, n_samples, data.shape[2])
    return res

def distances_list(runs, n_shots):
    return
        
def ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device) 
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        with tqdm(total=n_runs//batch_few_shot_runs) as pbar:
            for batch_idx in range(n_runs // batch_few_shot_runs):
                runs = generate_runs(features, run_classes, run_indices, batch_idx)
                means = torch.mean(runs[:,:,:n_shots], dim = 2)
                distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
                winners = torch.min(distances, dim = 2)[1]
                score = (winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy()
                scores += list(score)
                pbar.set_description(f"NCM accuracy: {np.mean(scores):.4f}")
                pbar.update(1)
        return stats(scores, "")

def crops_ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None,plot_ncm=False):

    with torch.no_grad():
        scores = []
        features = preprocess_crops(train_features, features, elements_train=elements_train)
        all_classes = list(features.keys())          
        n_runs = run_indices.shape[0]
        acc = 0
        with tqdm(total=n_runs) as pbar:
            for run_idx in range(n_runs):
                supports = []
                support_labels = []
                queries = []
                query_labels = []
                run_class = [all_classes[class_idx] for class_idx in run_classes[run_idx]] #list of classes in the run
                for class_idx,class_name in enumerate(run_class):
                    class_imgs = list(features[class_name].keys())
                    run_class_imgs = [class_imgs[img_idx] for img_idx in run_indices[run_idx][class_idx]] #list of image of the class in the run
                    for img_idx,image_name in enumerate(run_class_imgs):
                        if img_idx < n_shots: 
                            for crop_idx,crop_feature in enumerate(features[class_name][image_name]):
                                supports.append(crop_feature) #add the crop feature to the support set
                                support_labels.append([crop_idx,class_name,image_name]) #add the class name to the support set
                        else: #if it is a query image
                            for crop_idx,crop_feature in enumerate(features[class_name][image_name]): 
                                queries.append(crop_feature) #add the crop feature to the query set
                                query_labels.append([crop_idx,class_name,image_name]) #add the class name and the image name to the query set
                supports = torch.stack(supports).to(args.device) #shape: [n_supports, dim]
                queries = torch.stack(queries).to(args.device) #shape: [n_queries, dim]
                distances = torch.cdist(queries,supports,p=2) #shape: [n_queries, n_supports]
                queries_grouped = {}
                for i,[crop_idx,class_name,image_name] in enumerate(query_labels): #group the queries by image name
                    if image_name not in queries_grouped:
                        queries_grouped[image_name] = []
                    for j,distance in enumerate(distances[i]):
                        queries_grouped[image_name].append([distance,support_labels[j],[crop_idx,class_name]])
                run_acc = 0
                for query_name,distance_info in queries_grouped.items(): #for each query image
                    sorted_distance_info = sorted(distance_info,key=lambda x:x[0]) #sort the crops by distance
                    queries_grouped[query_name] = sorted_distance_info #save the sorted crops
                    min_distance,[support_crop_idx,support_class,support_name],[query_crop_idx,query_class] = sorted_distance_info[0] #get the closest crop
                    if support_class == query_class: #if the closest crop is from the same class as the query
                        run_acc += 1
                run_acc /= len(queries_grouped)
                
                acc += run_acc
                scores.append(run_acc)
                if plot_ncm:
                    for query_name,distance_info in queries_grouped.items(): #for each query image
                        sorted_distance_info = sorted(distance_info,key=lambda x:x[0]) #sort the crops by distance
                        queries_grouped[query_name] = sorted_distance_info #save the sorted crops
                        for i,(distance,[support_crop_idx,support_class,support_name],[query_crop_idx,query_class]) in enumerate(sorted_distance_info[:1]): #plot the 3 closest crops
                            query_mask = np.load(args.masks_dir+"/"+query_name.replace(".jpg",".npz")).get("arr_0")[query_crop_idx]
                            support_mask = np.load(args.masks_dir+"/"+support_name.replace(".jpg",".npz")).get("arr_0")[support_crop_idx]
                            query_image = Image.open(args.dataset_path + args.dataset+"/images/" + query_name)
                            support_image = Image.open(args.dataset_path +args.dataset+ "/images/" + support_name)
                            
                            support_width,support_height = support_image.size
                            support_aspect_ratio = support_width/support_height
                            if support_width > support_height:
                                support_image = transforms.Resize((int(args.average_width/support_aspect_ratio),args.average_width),antialias=True)(support_image) #resize keeping aspect ratio
                            else:
                                support_image = transforms.Resize((args.average_height,int(args.average_height*support_aspect_ratio)),antialias=True)(support_image) #resize keeping aspect ratio
                            query_width,query_height = query_image.size
                            query_aspect_ratio = query_width/query_height
                            if query_width > query_height:
                                query_image = transforms.Resize((int(args.average_width/query_aspect_ratio),args.average_width),antialias=True)(query_image) #resize keeping aspect ratio
                            else:
                                query_image = transforms.Resize((args.average_height,int(args.average_height*query_aspect_ratio)),antialias=True)(query_image) #resize keeping aspect ratio
                            query_height,query_width = query_mask.shape
                            support_height,support_width = support_mask.shape
                            query_coords = torch.where(query_mask==1)
                            support_coords = torch.where(support_mask==1)
                            query_bbox = [torch.min(query_coords[0]),torch.min(query_coords[1]),torch.max(query_coords[0]),torch.max(query_coords[1])]
                            support_bbox = [torch.min(support_coords[0]),torch.min(support_coords[1]),torch.max(support_coords[0]),torch.max(support_coords[1])]
                            query_crop_width,query_crop_height = query_bbox[2]-query_bbox[0],query_bbox[3]-query_bbox[1]
                            support_crop_width,support_crop_height = support_bbox[2]-support_bbox[0],support_bbox[3]-support_bbox[1]
                            if query_crop_width/query_width < args.crop_threshold:
                                query_crop_width = query_width*args.crop_threshold
                            if query_crop_height/query_height < args.crop_threshold:
                                query_crop_height = query_height*args.crop_threshold
                            if support_crop_width/support_width < args.crop_threshold:
                                support_crop_width = support_width*args.crop_threshold
                            if support_crop_height/support_height < args.crop_threshold:
                                support_crop_height = support_height*args.crop_threshold
                        
                            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                            main_title = f"Distance: {distance:.2f}"
                            fig.suptitle(main_title, fontsize=16)
                            axes[0].imshow(query_image)
                            axes[0].imshow(query_mask, alpha=0.3)
                            axes[0].add_patch(Rectangle((query_bbox[1], query_bbox[0]), query_crop_height, query_crop_width, fill=False, edgecolor='red', lw=3))
                            axes[0].set_title("Query Image: "+query_name)
                            axes[0].axis('off')
                            axes[1].imshow(support_image)
                            axes[1].imshow(support_mask, alpha=0.3)
                            axes[1].add_patch(Rectangle((support_bbox[1], support_bbox[0]), support_crop_height, support_crop_width, fill=False, edgecolor='red', lw=3))
                            axes[1].set_title("Support Image: "+support_name)
                            axes[1].axis('off')
                            plt.tight_layout()
                            plt.show()
                        
                        for i in range(5):
                            print("#" * 100 + "\n")
                pbar.set_description(f"NCM accuracy: {acc/(run_idx+1):.4f}")
                pbar.update(1)

        return  stats(scores, "")

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                if cosine:
                    new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                if cosine:
                    means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def kmeans(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(500):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                allocation = new_allocation
                means = (runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])) * args.n_queries) / runs.shape[2]
            winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def softkmeans(train_features, features, run_classes, run_indices, n_shots, transductive_temperature_softkmeans=args.transductive_temperature_softkmeans, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            runs = postprocess(runs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(30):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                means = torch.sum(runs[:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3]))
                means = means/(n_shots+soft_allocations.sum(dim = 1).reshape(batch_few_shot_runs, -1, 1))
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(batch_few_shot_runs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def get_features(model, loader, n_aug = args.sample_aug):
    model.eval()
    if args.masks_dir: 
        with torch.inference_mode():
            all_features = {}
            for batch_idx, (data,img_name) in enumerate(tqdm(loader,unit="batch")): 
                data = data.to(args.device) # [B, 3, 84, 84]
                if args.model.lower() == "dino":
                    features = model(data)
                else :
                    _, features = model(data) # [B, dim]
                for i in range(len(img_name)):
                    class_name = img_name[i][:9]
                    if class_name not in all_features:
                        all_features[class_name] = {}
                    if img_name[i] not in all_features[class_name]:
                        all_features[class_name][img_name[i]] = []
                    all_features[class_name][img_name[i]].append(features[i])
        return all_features
            
                
    for augs in tqdm(range(n_aug),unit="aug",desc="get_features"):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(tqdm(loader,unit="batch")):        
            with torch.no_grad():
                data, target = data.to(args.device), target.to(args.device)
                if args.model.lower() == "dino":
                    features = model(data)
                else:
                    _, features = model(data)
                all_features.append(features.cpu())
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        if not(args.use_masks):
            return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return crops_ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), crops_ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
def update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, train_clean)
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, val_loader)
    novel_features = get_features(model, novel_loader)

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res

def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    if args.save_features != "" and not(os.path.exists(args.save_features + str(args.n_shots[index]))):
        if not(args.use_masks):
            torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        else:
            all_features = train_features | val_features | novel_features
            torch.save(all_features, args.save_features + str(args.n_shots[index]))
            print("Saved features")
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
            if args.save_features != "":
                if not(args.use_masks):
                    torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
                else:
                    all_features = train_features | val_features | novel_features
                    torch.save(all_features, args.save_features + str(args.n_shots[index]))
                    print("Saved features")
                    
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf

print("eval_few_shot, ", end='')
