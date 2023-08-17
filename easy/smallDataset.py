import os
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np

if os.path.exists("/nasbrain/f21lin/testing/miniimagenetimages/images"):
    shutil.rmtree("/nasbrain/f21lin/testing/miniimagenetimages/images")
os.mkdir("/nasbrain/f21lin/testing/miniimagenetimages/images")

df_train = pd.read_csv("/nasbrain/datasets/miniimagenetimages/train.csv")
df_val = pd.read_csv("/nasbrain/datasets/miniimagenetimages/validation.csv")
df_test = pd.read_csv("/nasbrain/datasets/miniimagenetimages/test.csv")
df = pd.concat([df_train, df_val, df_test])
labels = df["label"].unique()
labels = np.random.permutation(labels)

train_classes = labels[:20]
val_classes = labels[20:30]
test_classes = labels[30:40]

selected_train_images = []
selected_val_images = []
selected_test_images = []

for label in train_classes:
    label_images = df[df["label"] == label]["filename"].values
    selected_train_images.extend(np.random.choice(label_images, 100, replace=False))

for label in val_classes:
    label_images = df[df["label"] == label]["filename"].values
    selected_val_images.extend(np.random.choice(label_images, 100, replace=False))

for label in test_classes:
    label_images = df[df["label"] == label]["filename"].values
    selected_test_images.extend(np.random.choice(label_images, 100, replace=False))
selected_train_df = df[df["filename"].isin(selected_train_images)]
selected_val_df = df[df["filename"].isin(selected_val_images)]
selected_test_df = df[df["filename"].isin(selected_test_images)]

selected_train_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/train.csv", index=False)
selected_val_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/validation.csv", index=False)
selected_test_df.to_csv("/nasbrain/f21lin/testing/miniimagenetimages/test.csv", index=False)

selected_images = np.concatenate((selected_train_images, selected_val_images, selected_test_images))
for image in tqdm(selected_images):
    shutil.copy("/nasbrain/datasets/miniimagenetimages/images/" + image, "/nasbrain/f21lin/testing/miniimagenetimages/images/" + image)

