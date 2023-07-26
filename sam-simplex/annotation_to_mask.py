annotation_path = "/nasbrain/f21lin/annotations/annotation.json"
output_dir = "/nasbrain/f21lin/annotations/"
import json
from pycocotools import mask
from PIL import Image
import os

with open(annotation_path, 'r') as f:
    coco_data = json.load(f)
images = coco_data['images']
annotations = coco_data['annotations']

for image_info in images:
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']
    image_filename = image_info['file_name']
    image_mask = Image.new('L', (image_width, image_height), 0)  # Create a blank binary mask
    for annotation in annotations : 
        if annotation["image_id"] == image_id:
            rle = mask.frPyObjects(annotation['segmentation'], image_height, image_width)
            binary_mask = mask.decode(rle)
            binary_mask = Image.fromarray(binary_mask*255)
            image_mask = Image.alpha_composite(image_mask.convert("RGBA"), binary_mask.convert("RGBA"))
    binary_mask_path = os.path.join(output_dir, image_filename.split('.')[0] + '.png')
    image_mask.save(binary_mask_path)

