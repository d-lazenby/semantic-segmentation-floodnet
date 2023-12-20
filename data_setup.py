"""
Contains functionality for creating custom segmentation dataset and data loaders
"""

import os

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations

NUM_WORKERS = os.cpu_count()

class SegmentationDataset(Dataset):

    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row['images']
        mask_path = row['masks']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('uint8')

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # mask is in two dimensions so expand this to include a single channel: (h, w) -> (h, w, c)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data['image']
            mask = data['mask']

        # PyTorch uses (c, h, w) so we transpose from (h, w, c)
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1))

        image = torch.Tensor(image) / 255.0
        mask = torch.Tensor(mask).long()
        # mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask

def create_dataloaders(
    train_transform: albumentations.core.composition.Compose,
    valid_transform: albumentations.core.composition.Compose,
    batch_size: int, 
    training_prop: float = 1.0,
    train_dir: str = './csv/train_final.csv',
    valid_dir: str = './csv/val_final.csv',
    test_dir: str = './csv/test_final.csv',
    num_workers: int = NUM_WORKERS
):
  """
  Example: 
    train_loader, valid_loader, test_loader = create_dataloaders(
      train_transform=train_transform,
      valid_transform=valid_transform,
      batch_size=BATCH_SIZE,
      training_prop=1.0,
      num_workers=NUM_WORKERS)
  """


  # Load label/mask pair paths in dataframe
  train_df = pd.read_csv(train_dir)
  valid_df = pd.read_csv(valid_dir)
  test_df = pd.read_csv(test_dir)

  TRAIN_LIMIT = round(training_prop * train_df.shape[0])
  VALID_LIMIT = round(training_prop * valid_df.shape[0])
  
  # Truncate training and validation sets
  tr_df = train_df.iloc[:TRAIN_LIMIT].copy()
  va_df = valid_df.iloc[:VALID_LIMIT].copy()

  print(f"Number of training samples: {len(tr_df)};\nnumber of validation samples: {len(va_df)};\nnumber of testing samples: {len(test_df)};")

  train_ds = SegmentationDataset(tr_df, train_transform)
  valid_ds = SegmentationDataset(va_df, valid_transform)
  test_ds = SegmentationDataset(test_df, valid_transform)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

  return train_loader, valid_loader, test_loader
