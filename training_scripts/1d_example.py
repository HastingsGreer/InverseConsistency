import icon_registration.data as data
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.visualize as visualize
import icon_registration.train as train
import icon_registration.inverseConsistentNet as inverseConsistentNet

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import pickle

import footsteps

footsteps.initialize(run_name="1dtest")

batch_size = 128

d1, d2 = data.get_dataset_1d("train", data_size=50, batch_size=batch_size)
d1_t, d2_t = data.get_dataset_1d("test", data_size=50, batch_size=batch_size)


lmbda = 20
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
print("=" * 50)
net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.FunctionFromVectorField(networks.FCNet1D(size=50)),
    # Our image similarity metric. The last channel of x and y is whether the value is interpolated or extrapolated,
    # which is used by some metrics but not this one
    lambda x, y: torch.mean((x[:, :1] - y[:, :1]) ** 2),
    lmbda,
)

input_shape = next(iter(d1))[0].size()
print(input_shape)
network_wrappers.assignIdentityMap(net, input_shape)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train()


xs = []
for _ in range(40):
    y = np.array(train.train1d(net, optimizer, d1, d2, epochs=50))
    xs.append(y)
    x = np.concatenate(xs)
    plt.title(
        "Loss curve for " + type(net.regis_net).__name__ + " lambda=" + str(lmbda)
    )
    plt.plot(x[:, :3])
    plt.savefig(footsteps.output_dir + f"loss.png")
    plt.clf()
    plt.title("Log # pixels with negative Jacobian per epoch")
    plt.plot(x[:, 3])
    # random.seed(1)
    plt.savefig(footsteps.output_dir + f"lossj.png")
    plt.clf()
    with open(footsteps.output_dir + "loss.pickle", "wb") as f:
        pickle.dump(x, f)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
    net(image_A, image_B)
    for N in range(3):

        def plot(im):
            plt.plot(im[0].cpu().detach())

        plt.subplot(2, 2, 1)
        plt.title("image_A")
        plot(image_A[N])
        plt.subplot(2, 2, 2)
        plt.title("image_B")
        plot(image_B[N])
        plt.subplot(2, 2, 3)
        plt.title("warped_image_A")
        plot(net.warped_image_A[N])
        plt.subplot(2, 2, 4)
        plt.title("vectorfield")
        plot(net.phi_AB_vectorfield[N])
        plt.savefig(
            footsteps.output_dir + f"epoch{_:03}" + "case" + str(N) + ".png",
        )
        plt.clf()


random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
os.mkdir(footsteps.output_dir + "final/")
for N in range(30):
    visualize.visualizeRegistrationCompact(net, image_A, image_B, N)
    plt.savefig(footsteps.output_dir + f"final/{N}.png")
    plt.clf()

torch.save(net.state_dict(), footsteps.output_dir + "network.trch")
torch.save(optimizer.state_dict(), footsteps.output_dir + "opt.trch")
