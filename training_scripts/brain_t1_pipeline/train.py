import random

import footsteps
import icon_registration
import icon_registration.data as data
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.losses as losses
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
import icon_registration.train as train
import torch
import torch.nn.functional as F
from icon_registration.mermaidlite import (
    compute_warped_image_multiNC,
    identity_map_multiN,
)

BATCH_SIZE = 8
input_shape = [BATCH_SIZE, 1, 130, 155, 130]

GPUS = 4


def make_network():
    phi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    hires_net = inverseConsistentNet.GradientICON(
        network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(
                network_wrappers.DoubleNet(phi, psi), dimension=3
            ),
            network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        ),
        losses.LNCC(sigma=5),
        .7,
    )
    network_wrappers.assignIdentityMap(hires_net, input_shape)
    return hires_net


def make_batch():
    image = torch.cat([random.choice(brains) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image


if __name__ == "__main__":
    footsteps.initialize()
    brains = torch.load(
        "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(
        (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
    )

    net_par.train()

    icon_registration.train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net)
