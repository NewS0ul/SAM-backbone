import os
import shutil
import pandas as pd
from tqdm import tqdm
import random

if os.path.exists("/nasbrain/f21lin/testing/miniimagenetimages/images"):
    shutil.rmtree("/nasbrain/f21lin/testing/miniimagenetimages/images")
os.mkdir("/nasbrain/f21lin/testing/miniimagenetimages/images")

df = pd.read_csv("/nasbrain/datasets/miniimagenetimages/test.csv")
labels = df["label"].unique()
image_per_class = {}
for label in labels:
    image_per_class[label] = df[df["label"] == label]["filename"].values.tolist()

selected_images = []
for label in labels:
    selected_images.append(random.sample(image_per_class[label], 100))
selected_images = [item for sublist in selected_images for item in sublist]
for image in tqdm(selected_images):
    shutil.copy("/nasbrain/datasets/miniimagenetimages/images/" + image, "/nasbrain/f21lin/testing/miniimagenetimages/images/" + image)

train_classes = labels[:10]
val_classes = labels[10:15]
test_classes = labels[15:20]

train_df = df[df["label"].isin(train_classes)&df["filename"].isin(selected_images)]
val_df = df[df["label"].isin(val_classes)&df["filename"].isin(selected_images)]
test_df = df[df["label"].isin(test_classes)&df["filename"].isin(selected_images)]
train_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/train.csv", index=False)
val_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/validation.csv", index=False)
test_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/test.csv", index=False)


