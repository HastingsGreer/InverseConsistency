from collections import OrderedDict
import torch
import numpy as np
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.visualize as visualize
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.data as data
import footsteps
import os
import matplotlib.pyplot as plt
import random
import pickle

batch_size = 128
data_size = 50
d1, d2 = data.get_dataset_triangles(
    "train", data_size=data_size, hollow=True, batch_size=batch_size
)
d1_t, d2_t = data.get_dataset_triangles(
    "test", data_size=data_size, hollow=True, batch_size=batch_size
)

image_A, image_B = (x[0].cuda() for x in next(zip(d1, d2)))

net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.DoubleNet(
        network_wrappers.FunctionFromMatrix(
            networks.ConvolutionalMatrixNet(dimension=2),
        ),
        network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=2)),
    ),
    lambda x, y: torch.mean((x - y) ** 2),
    700,
)

input_shape = next(iter(d1))[0].size()
net.assign_identity_map(input_shape)
# pretrained_weights = torch.load("results/affine_triangle_pretrain/epoch000case0.png")
# pretrained_weights = OrderedDict(
#    [
#        (a.split("regis_net.")[1], b)
#        for a, b in pretrained_weights.items()
#        if "regis_net" in a
#    ]
# )

# net.affine_regis_net.load_state_dict(pretrained_weights)
net.cuda()

import icon_registration.train as train

optim = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train().cuda()


xs = []
for _ in range(240):
    y = np.array(train.train2d(net, optim, d1, d2, epochs=50))
    xs.append(y)
    x = np.concatenate(xs)
    plt.title("Loss curve for " + type(net.regis_net).__name__)
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
    for N in range(3):
        visualize.visualizeRegistration(
            net,
            image_A,
            image_B,
            N,
            footsteps.output_dir + f"epoch{_:03}" + "case" + str(N) + ".png",
        )

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
