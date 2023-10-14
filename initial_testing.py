import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def file_locs_and_labels(root_dir):
    """
    Returns a dataframe of names and class labels of all images in root_dir.
    """
    # List of subfolders sorted by class label, omitting secret files
    sub_folders = sorted((f for f in os.listdir(root_dir) \
                          if not f.startswith(".")), key=str.lower)

    # List of class labels
    labels = [int(name.split(".")[0]) for name in sub_folders]

    data = []

    for sub_f, label in zip(sub_folders, labels):
        for _, __, f in os.walk(os.path.join(root_dir, sub_f)):
            for file in f:
                if file.endswith(".jpg"):
                    data.append((os.path.join(sub_f, file), label))

    df = pd.DataFrame(data, columns=['file_name', 'label'])

    return df


root_dir = './test'
sub_folders = os.listdir(root_dir)

img_paths, mask_paths = [], []
for sub_f in sub_folders:
    for _, __, f in os.walk(os.path.join(root_dir, sub_f)):
        for file in f:
            if file.endswith(".png"):
                mask_path = os.path.join(sub_f, file)
                mask_paths.append(mask_path)
            else:
                img_paths.append(os.path.join(sub_f, file))

print([*zip(img_paths, mask_paths)])
