import parent
import data
import networks
import visualize
#import train
import inverseConsistentNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os

import describe

def train2d(net, optimizer, d1, epochs=400):

    loss_history = []
    print("[", end="")
    for epoch in range(epochs):
        print("-", end="", flush=True)
        if (epoch + 1) % 50 == 0:
            print("]", end="\n[")
        for A in d1:
            if A[0].size()[0] == 128:
                image_A = A[0].cuda()
                image_B = A[1].cuda()
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

d1 = data.get_dataset_sunnyside("train", scale=.7
)

d1_t = data.get_dataset_sunnyside("test", scale=.7
)
network = networks.tallUNet2

lmbda = 2048
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
print("=" * 50)
print(network, lmbda)
net = inverseConsistentNet.InverseConsistentNet(
    network(), lmbda, next(iter(d1))[0].size()
)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train()
xs = []
for _ in range(240):
    y = np.array(train2d(net, optimizer, d1, epochs=5))
    xs.append(y)
    x = np.concatenate(xs)
    plt.title(
        "Loss curve for " + type(net.regis_net).__name__ + " lambda=" + str(lmbda)
    )
    plt.plot(x[:, :3])
    plt.savefig(describe.run_dir + f"loss.png")
    plt.clf()
    plt.title("Log # pixels with negative Jacobian per epoch")
    plt.plot(x[:, 3])
    # random.seed(1)
    plt.savefig(describe.run_dir + f"lossj.png")
    plt.clf()

    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    image_A, image_B = [im.cuda() for im in next(iter(d1_t))]
    for N in range(3):
        visualize.visualizeRegistration(
            net,
            image_A,
            image_B,
            N,
            describe.run_dir + f"epoch{_:03}" + "case" + str(N) + ".png",
        )
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
os.mkdir(describe.run_dir + "final/")
for N in range(30):
    visualize.visualizeRegistrationCompact(net, image_A, image_B, N)
    plt.savefig(describe.run_dir + f"final/{N}.png")
    plt.clf()

torch.save(net.state_dict(), describe.run_dir + "network.trch")
torch.save(optimizer.state_dict(), describe.run_dir + "opt.trch")
