import argparse
import os
import torch.backends.cudnn as cudnn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import ImageFilter as IF
from torch import nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from PIL import Image
import skimage.measure
from skimage.measure import block_reduce
from torch.utils.data.dataloader import DataLoader
from dataset import Dataset

from tqdm import tqdm
from utils import AverageMeter
from model import RES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)


dataset = Dataset('final_input', 128,
                      '25', '2', None,
                      True)
  
dataloader = DataLoader(dataset=dataset,
                            batch_size=20,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)

model = RES()
print(model)
model = model.to(device)
criterion = nn.MSELoss(reduction='sum')

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(50):
    epoch_losses = AverageMeter()
    with tqdm(total=(len(dataset) - len(dataset) % 20)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, 50))
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels) / (2 * len(inputs))

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            _tqdm.update(len(inputs))

    torch.save(model.state_dict(), os.path.join('newoutput', '{}_epoch_{}.pth'.format('sigma0-test-6', epoch)))

state_dict = model.state_dict()
print(state_dict)