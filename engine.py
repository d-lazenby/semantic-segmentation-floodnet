
"""
Contains code for training and validating a SegmentationModel
"""

from tqdm.auto import tqdm
from pathlib import Path

import numpy as np

import torch

import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex


def train_fn(data_loader, model, optimizer, device, metric):

    model.train()
    total_loss, j_ind = 0.0, 0.0
    for images, masks in tqdm(data_loader):

        images, masks = images.to(device), masks.to(device)

        logits, loss = model(images, masks)
        y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()
        j_ind += metric(y_pred, masks.squeeze()).item()

    return total_loss / len(data_loader), j_ind / len(data_loader)


def valid_fn(data_loader, model, device, metric):

    model.eval()
    total_loss, j_ind = 0.0, 0.0

    with torch.inference_mode():
        for images, masks in tqdm(data_loader):

            images, masks = images.to(device), masks.to(device)

            logits, loss = model(images, masks)
            y_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            total_loss += loss.item()
            j_ind += metric(y_pred, masks.squeeze()).item()

    return total_loss / len(data_loader), j_ind / len(data_loader)


def train_model(
    model, 
    train_loader,
    valid_loader,
    optimizer,
    epochs,
    device,
    model_name='best_model.pt'):
  
  results = {
      "train_loss": [],
      "train_JI": [],
      "valid_loss": [],
      "valid_JI": [],
  }

  best_valid_loss = np.Inf

  # Create target directory
  model_dir = Path('models')
  model_dir.mkdir(parents=True,
                  exist_ok=True)
  
  model_path = model_dir / model_name 

  metric = MulticlassJaccardIndex(num_classes=10, ignore_index=0).to(device)

  for epoch in range(epochs):
    train_loss, train_ji = train_fn(data_loader=train_loader, 
                                    model=model, 
                                    optimizer=optimizer, 
                                    device=device,
                                    metric=metric)
    
    valid_loss, valid_ji = valid_fn(data_loader=valid_loader, 
                                    model=model,
                                    device=device,
                                    metric=metric)
    
    if valid_loss < best_valid_loss:
      torch.save(obj=model.state_dict(), 
                  f=model_path)
      print(f'Model saved at {model_path}')
      best_valid_loss = valid_loss
    
    print(f"Epoch {epoch+1} --------------------------")
    print(f"train_loss {train_loss:.4f} | train_jacc {train_ji:.4f} | valid_loss {valid_loss:.4f} | valid_jacc {valid_ji:.4f}")

    
    results["train_loss"].append(train_loss)
    results["train_JI"].append(train_ji)
    results["valid_loss"].append(valid_loss)
    results["valid_JI"].append(valid_ji)

  return results
