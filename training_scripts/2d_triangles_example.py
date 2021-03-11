import parent
import data
import networks
import network_wrappers
import visualize
import train
import inverseConsistentNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import pickle

import describe

import argparse

parser = argparse.ArgumentParser()

batch_size = 128

d1, d2 = data.get_dataset_triangles(
    "train", data_size=50, hollow=False, batch_size=batch_size
)
d1_t, d2_t = data.get_dataset_triangles(
    "test", data_size=50, hollow=False, batch_size=batch_size
)

lmbda = 2048
noise_std,lmbda_noise = 0.,0
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
print("=" * 50)
net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=2)),
    lambda x, y: torch.mean((x - y) ** 2),
    lmbda,noise_std,lmbda_noise,
)
input_shape = next(iter(d1))[0].size()
network_wrappers.assignIdentityMap(net, input_shape)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train()

xs = []
for _ in range(40):
    y = np.array(train.train2d(net, optimizer, d1, d2, epochs=50))
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
    with open(describe.run_dir + "loss.pickle", "wb") as f:
        pickle.dump(x, f)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
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
