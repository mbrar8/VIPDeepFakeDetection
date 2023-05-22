import torch
import torch.nn as nn
import os
import json
import cv2 as cv
import numpy as np
from vit_dataset import DFDCImageDataset
from vit import ViT
from torch.utils.data import DataLoader
import torch.optim as optim


"""
Train ViT for DFD. Preprocessing - Faster RCNN on videos - resize bboxes to first frame size. Given new res wxh, crop top portion with size (wxw). Resize image to 224x224. Run EfficientNetB0 on this wxw image, split into patches of size 32x32 or 7x7 (leave up to experiment?) Feed these patches into a linear projection into a transformer encoder. The transformer's output is the input to another encoder. The dataset up until this consists of Faster RCNN results - does not include cropping yet"
"""


if __name__ == " __main__()":
    
    folders = ["dfdc_train_part_0"]

    dataset = DFDCImageDataset(folders)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = ViT()

    lossfxn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        for i, data in enumerate(dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = lossfxn(output, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './vit.pth')


    

