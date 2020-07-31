import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, datasets

from model import RetinaNet
from utilities.dataset import resize_512, mapToBoxes, toPyramidForTraining

obj = transforms.ToTensor()

masks = list(sorted(os.listdir(os.path.join("/home/michalgorszczak/PennFudanPed", "PedMasks"))))
list_of_masks_numbers = [np.unique(np.array((Image.open(os.path.join("/home/michalgorszczak/PennFudanPed", "PedMasks", x)))))[1:] for x in
                         masks]
list_of_masks = [np.array(resize_512(Image.open(os.path.join("/home/michalgorszczak/PennFudanPed", "PedMasks", x)))) for x in masks]

positions = mapToBoxes(list_of_masks, list_of_masks_numbers)



train_transform = transforms.Compose(
    [resize_512, transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

image_data = datasets.ImageFolder('/home/michalgorszczak/PennFudanPed', transform=train_transform)

data_loader = torch.utils.data.DataLoader(image_data, batch_size=1, )

data_imgs = torch.empty((0, 3, 512, 512))
data_masks = torch.empty((0, 3, 512, 512))

for data, file in data_loader:
    if file == 1:
        data_imgs = torch.cat((data_imgs, data), )
    else:
        data_masks = torch.cat((data_masks, data))

plt.imshow(data_imgs[0].permute(1, 2, 0))

model = RetinaNet()
model.cuda()
kwargs = {"alpha": 0.25, "gamma": 2, "reduction": 'mean'}
criterion_class = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

data_imgs = data_imgs.cuda()

for A in range(20):
    x_0 = 0
    print(A)
    for x in range(8, 160, 8):

        optimizer.zero_grad()

        result = model(data_imgs[x_0:x])

        q = toPyramidForTraining(positions[x_0:x])

        loss = []
        for y, val in enumerate(result):
            loss.append(criterion_class(val, q[y].cuda()))

        sum_loss = 0
        for y in loss:
            sum_loss += y.detach().cpu().numpy()
            y.backward(retain_graph=True)
        optimizer.step()

        print(f"loss: {sum_loss}")

        x_0 = x


model.eval()

rs = model(data_imgs[169:])
plt.imshow(rs[3].view(8, 8, 3).cpu().detach().numpy())
plt.show()
plt.imshow(data_imgs[169].permute(1, 2, 0).cpu().detach().numpy())
plt.show()
