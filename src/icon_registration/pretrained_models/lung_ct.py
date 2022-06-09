import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.config as config

from .. import losses, network_wrappers, networks
from ..mermaidlite import compute_warped_image_multiNC, identity_map_multiN


def make_network():

    phi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    xi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    net = losses.GradientICON(
        network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(network_wrappers.DoubleNet(phi, psi), 3),
            xi,
        ),
        losses.LNCC(sigma=5),
        1,
    )

    return net


def LungCT_registration_model(pretrained=True):
    # The definition of our final 4 step registration network.

    net = make_network()
    input_shape = [1, 1, 175, 175, 175]

    net.assign_identity_map(input_shape)

    if pretrained:
        from os.path import exists

        if not exists("lung_model_wms/"):
            print("Downloading pretrained model (200mb)")
            import urllib.request

            urllib.request.urlretrieve(
                "https://github.com/uncbiag/ICON/releases/download/pretrained_lung_model/lung_model_wms.zip",
                "lung_model_wms.zip",
            )
            shutil.unpack_archive("lung_model_wms.zip", "lung_model_wms")

        trained_weights = torch.load(
            "lung_model_wms/warped_masked_smuth/net91800",
            map_location=torch.device("cpu"),
        )
        net.regis_net.load_state_dict(trained_weights, strict=False)

    net.to(config.device)
    net.eval()
    return net
