import parent

from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch

import inverseConsistentNet
import networks
import data

BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
working_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]


net = inverseConsistentNet.InverseConsistentNet(
    networks.tallUNet(dimension=3),
    lmbda=166,
    input_shape=working_shape,
    random_sampling=False,
)
net.load_state_dict(torch.load("network_weights/lowres_knee_network"))

knees, medknees = data.get_knees_dataset()
