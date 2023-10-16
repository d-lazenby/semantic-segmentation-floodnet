import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_image_path_dfs(split):
    for _, __, f in os.walk(os.path.join('./', split, f'{split}-org-img')):
        df = pd.DataFrame(data=f, columns=['images'])

    df['masks'] = df['images'].str.extract("(\d{4})") + "_lab.png"

    df['images'] = f'semantic-segmentation-floodnet/{split}/{split}-org-img/' + df['images']
    df['masks'] = f'semantic-segmentation-floodnet/{split}/{split}-label-img/' + df['masks']

    return df


splits = ['train', 'val', 'test']
dfs = []
for split in splits:
    df = get_image_path_dfs(split)
    dfs.append(df)

df_train, df_val, df_test = dfs[0], dfs[1], dfs[2]

print(df_train)
print(df_val)
print(df_test)

# TODO: write script that writes image and mask paths to CSVs for train, test and val images
