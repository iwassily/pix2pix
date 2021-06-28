
import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def make_dataloaders(tr_path, val_path, tr_batch_size=4, val_batch_size=8, order='io'):
  # 'io' for [input, output], 'oi'
  dataset = ImageFolder(tr_path, transform=tt.ToTensor())
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

  inputs = torch.empty( (len(dataloader), 3, 256, 256), dtype=torch.float32)
  outputs = torch.empty( (len(dataloader), 3, 256, 256), dtype=torch.float32)

  for i, image in enumerate(dataloader):
    if (order=='io'):
        inputs[i] = image[0][0, :, :, :256]
        outputs[i] = image[0][0, :, :, 256:512]
    else:
        outputs[i] = image[0][0, :, :, :256]
        inputs[i] = image[0][0, :, :, 256:512]

  train_loader = DataLoader(list(zip(inputs, outputs)),  num_workers=2, pin_memory=True, drop_last=True, batch_size=tr_batch_size, shuffle=True) 

  dataset = ImageFolder(val_path, transform=tt.ToTensor())
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

  inputs = torch.empty( (len(dataloader), 3, 256, 256), dtype=torch.float32)
  outputs = torch.empty( (len(dataloader), 3, 256, 256), dtype=torch.float32)

  for i, image in enumerate(dataloader):
    if (order=='io'):
        inputs[i] = image[0][0, :, :, :256]
        outputs[i] = image[0][0, :, :, 256:512]
    else:
        outputs[i] = image[0][0, :, :, :256]
        inputs[i] = image[0][0, :, :, 256:512]

  validation_loader = DataLoader(list(zip(inputs, outputs)),  num_workers=2, pin_memory=True, drop_last=True, batch_size=val_batch_size, shuffle=True) 

  return train_loader, validation_loader