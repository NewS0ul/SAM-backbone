import os
from PIL import Image
from tqdm import tqdm

all_files = os.listdir("/nasbrain/datasets/miniimagenetimages/images")

all_width = []
all_height = []

for filename in tqdm(all_files):
    path = "/nasbrain/datasets/miniimagenetimages/images/" + filename
    image = Image.open(path)
    width, height = image.size
    all_width.append(width)
    all_height.append(height)
mean_height = sum(all_height)/len(all_height)
mean_aspect_ratio = sum([width/height for width,height in zip(all_width,all_height)])/len(all_width)
print("Mean height:",mean_height)
print("Mean aspect ratio:",mean_aspect_ratio)
