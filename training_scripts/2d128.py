from .. import data
import networks
import visualize
import train
import inverseConsistentNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os

print("name of run: ", end="")
resultspath = "results/" + input() + "/"

d1_triangles, d2_triangles = data.get_dataset_triangles(
    "train", data_size=50, hollow=True
)
d1_triangles_test, d2_triangles_test = data.get_dataset_triangles(
    "test", data_size=50, hollow=True
)

os.mkdir(resultspath)


network = networks.tallUNet2

d1, d2, d1_t, d2_t = (d1_triangles, d2_triangles, d1_triangles_test, d2_triangles_test)
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
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train()
for _ in range(1):
    x = np.array(train.train2d(net, optimizer, d1, d2, epochs=50))

    plt.title(
        "Loss curve for " + type(net.regis_net).__name__ + " lambda=" + str(lmbda)
    )
    plt.plot(x[:, :3])
    plt.title("Log # pixels with negative Jacobian per epoch")
    plt.plot(x[:, 3])
    # random.seed(1)
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
            resultspath + "epoch" + str(_) + "case" + str(N) + ".png",
        )
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
os.mkdir(resultspath + "final/")
for N in range(30):
    visualize.visualizeRegistrationCompact(net, image_A, image_B, N)
    plt.savefig(resultspath + f"final/{N}.png")
    plt.clf()

torch.save(net.state_dict(), resultspath + "network.trch")
torch.save(optimizer.state_dict(), resultspath + "opt.trch")
