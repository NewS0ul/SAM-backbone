import pandas as pd
import cv2
from torchvision import transforms
from sam_crop import SAMTransform
import numpy as np
import sys

norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
samTransform = SAMTransform(norm)

df = pd.read_csv("/nasbrain/f21lin/all_images.csv")
image = cv2.imread("/nasbrain/datasets//miniimagenetimages/images/"+df.iloc[1407]["filename"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
crops = samTransform(image)
print(len(crops))
print(sum(sys.getsizeof(crop) for crop in crops))
