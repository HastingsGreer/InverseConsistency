import torch
import icon_registration.train as train
import matplotlib.pyplot as plt
import numpy as np
import icon_registration.visualize as visualize
import icon_registration.data as data
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
import icon_registration.inverseConsistentNet as inverseConsistenNet
import footsteps

footsteps.initialize(run_name="retina_experiment")


def make_network():
    inner_net = network_wrappers.FunctionFromVectorField(
        networks.tallUNet2(dimension=2)
    )

    for _ in range(3):
        inner_net = network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(inner_net, 2),
            network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=2)),
        )
    lmbda = 0.2
    net = inverseConsistenNet.GradientICON(
        inner_net, inverseConsistenNet.BlurredSSD(sigma=3), lmbda
    )
    return net


if __name__ == "__main__":
    ds1, ds2 = data.get_dataset_retina()

    net = make_network()

    input_shape = next(iter(ds1))[0].size()

    next(iter(ds2))  # keep them synchonized

    network_wrappers.assignIdentityMap(net, input_shape)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    net.train()
    # do training
    loss_curves = np.array(train.train2d(net, optimizer, ds1, ds2, epochs=90))

    # save results
    plt.plot(loss_curves[:, :3])
    plt.savefig(footsteps.output_dir + "loss.png")
    plt.clf()

    image_A, image_B = (x[0].cuda() for x in next(zip(ds1, ds2)))
    for N in range(3):
        visualize.visualizeRegistration(
            net,
            image_A,
            image_B,
            N,
            f"{footsteps.output_dir}test{N}.png",
            linewidth=0.25,
        )
