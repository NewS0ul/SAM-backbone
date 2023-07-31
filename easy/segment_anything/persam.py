import os
import cv2
from torch.nn import functional as F
from build_sam import sam_model_registry
from predictor import SamPredictor
def persam(ref_image_path,ref_mask_path,test_images_path,output_path,sam_type,sam_path):
    os.makedirs(output_path,exist_ok=True)
    
    ref_image = cv2.imread(ref_image_path)
    ref_image_path = cv2.cvtColor(ref_image,cv2.COLOR_BGR2RGB)
    
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask,cv2.COLOR_BGR2RGB)
    
    print("=====> Load SAM")
    sam = sam_model_registry[sam_type](checkpoint=sam_path).cuda()
    predictor = SamPredictor(sam)
    
    print("======> Obtain Location Prior" )
    ref_mask = predictor.set_image(ref_image,ref_mask)
    ref_feat = predictor.features.squeeze().permute(1,2,0)
    
    ref_mask = F.interpolate(ref_mask,size=ref_feat.shape[0:2],mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]
    
    target_feat = ref_feat[ref_mask>0]
    target_embedding = target_feat.mean(dim=0).unsqueeze(0)
    target_feat = target_embedding/target_embedding.norm(dim=-1,keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
    
    print("======> Start Testing")
    for image_name in os.listdir(test_images_path):
        test_image_path = os.path.join(test_images_path,image_name)
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
        
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()
        
        C,h,w = test_feat.shape
        test_feat = test_feat/test_feat.norm(dim=0,keepdim=True)
        test_feat = test_feat.reshape(C,h*w)
        sim = target_feat @ test_feat
        
        sim = sim.reshape(1,1,h,w)
        sim = F.interpolate(sim,scale_factor=7,mode="bilinear")
        sim = predictor.model.postprocess_masks(sim,input_size=predictor.input_size,original_size=predictor.original_size).squeeze()
        
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
        
        sim = (sim-sim.mean())/torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0),size=(64,64),mode="bilinear")
        attn_sim = sim.sigmoid().unsqueeze(0).flatten(3)
        
        masks, scores,logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
        ) #Finir de complèter avec le git de persam
        
    
    
    
    