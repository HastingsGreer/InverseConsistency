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


batch_size = 128

for data_size in (32, 64, 128, 256, 512):
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    d1, d2 = data.get_dataset_triangles(
        "train", data_size=data_size, hollow=False, batch_size=batch_size
    )
    d1_t, d2_t = data.get_dataset_triangles(
        "test", data_size=data_size, hollow=False, batch_size=batch_size
    )
    t_image_A, t_image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
    for lr in (0.0001, 0.001, 0.01):
        for loss in ("InverseConsistentNet", "GradientICON"):
            for lmbda in {
                "InverseConsistentNet": (64, 256, 1024),
                "GradientICON": (0.2, 1, 2),
            }[loss]:
                for architecture in ("FCNet", "UNet"):
                    output_dir = (
                        footsteps.output_dir
                        + f"{data_size}-{lr}-{loss}-{lmbda}-{architecture}/"
                    )
                    os.mkdir(output_dir)
                    if architecture == "FCNet":
                        inner_net = networks.FCNet(size=data_size)
                    else:
                        inner_net = networks.tallUNet2(dimension=2)

                    net = getattr(inverseConsistentNet, loss)(
                        network_wrappers.FunctionFromVectorField(inner_net),
                        # Our image similarity metric. The last channel of x and y is whether the value is interpolated or extrapolated,
                        # which is used by some metrics but not this one
                        lambda x, y: torch.mean((x[:, :1] - y[:, :1]) ** 2),
                        lmbda,
                    )

                    input_shape = next(iter(d1))[0].size()
                    net.assign_identity_map(input_shape)
                    net.cuda()
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
                    net.train()

                    xs = []
                    for _ in range(10):
                        y = np.array(train.train2d(net, optimizer, d1, d2, epochs=50))
                        xs.append(y)

                    x = np.concatenate(xs)
                    plt.title(
                        "Loss curve for "
                        + type(net.regis_net).__name__
                        + " lambda="
                        + str(lmbda)
                    )
                    plt.plot(x[:, :3])
                    plt.savefig(output_dir + f"loss.png")
                    plt.clf()
                    plt.title("Log # pixels with negative Jacobian per epoch")
                    plt.plot(x[:, 3])
                    # random.seed(1)
                    plt.savefig(output_dir + f"lossj.png")
                    plt.clf()
                    with open(output_dir + "loss.pickle", "wb") as f:
                        pickle.dump(x, f)
                    # torch.manual_seed(1)
                    # torch.cuda.manual_seed(1)
                    # np.random.seed(1)
                    net(t_image_A, t_image_B)
                    for N in range(6):
                        visualize.visualizeRegistration(
                            net,
                            t_image_A,
                            t_image_B,
                            N,
                            output_dir + f"epoch{_:03}" + "case" + str(N) + ".png",
                        )
