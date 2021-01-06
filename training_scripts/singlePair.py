import parent
import data
import networks
import visualize
import train
import inverseConsistentNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os

import describe


random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
d1_triangles, d2_triangles = data.get_dataset_triangles(
    "train", data_size=50, hollow=False
)

network = networks.FCNet

d1, d2= (d1_triangles, d2_triangles)
lmbda =2048 
image_A, image_B = (x[0].cuda() for x in next(zip(d1, d2)))
image_A = image_A[:1].cuda()
image_B = image_B[:1].cuda()
print("=" * 50)
print(network, lmbda)
net = inverseConsistentNet.InverseConsistentNet(
    network(), lmbda, image_A.size()
)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
net.train()



def train2d(net, optimizer, image_A, image_B, epochs=400):

    loss_history = []
    print("[", end="")
    for epoch in range(epochs):
        print("-", end="")
        if (epoch + 1) % 50 == 0:
            print("]", end="\n[")
        for _ in range(1):
            optimizer.zero_grad()
            loss, inverse_consistency_loss, similarity_loss, transform_magnitude = net(
                image_A, image_B
            )

            loss.backward()
            optimizer.step()
        du = (net.phi_AB[:, :, 1:, :-1] - net.phi_AB[:, :, :-1, :-1]).detach().cpu()
        dv = (net.phi_AB[:, :, :-1, 1:] - net.phi_AB[:, :, :-1, :-1]).detach().cpu()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]

        loss_history.append(
            [
                inverse_consistency_loss.item(),
                similarity_loss.item(),
                transform_magnitude.item(),
                torch.log(torch.sum(dA < 0) + 0.1),
            ]
        )
    print("]")
    return loss_history


losses = []
for _ in range(400):
    x = np.array(train2d(net, optimizer, image_A, image_B, epochs=10))
    losses.append(x)
    # random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    visualize.visualizeRegistration(
        net,
        image_A,
        image_B,
        0,
        describe.run_dir + f"epoch{_:03}.png",
    )

torch.save(net.state_dict(), describe.run_dir + "network.trch")
torch.save(optimizer.state_dict(), describe.run_dir + "opt.trch")

