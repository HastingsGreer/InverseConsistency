import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch

from torch.cuda.amp import GradScaler, autocast

BATCH_SIZE = 8
input_shape = [BATCH_SIZE, 1, 130, 155, 130]

GPUS = 1


def make_network():
    phi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    psi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    hires_net = icon.GradientICON(
        icon.TwoStepRegistration(
            icon.DownsampleRegistration(
                phi, dimension=3
            ),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        ),
        icon.LNCC(sigma=5),
        .7,
    )
    hires_net.assign_identity_map(input_shape)
    return hires_net


def make_batch():
    image = torch.cat([random.choice(brains) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

def train_batchfunction(
    net,
    optimizer,
    make_batch,
    steps=100000,
    unwrapped_net=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """

    if unwrapped_net is None:
        unwrapped_net = net

    scaler = GradScaler()

    moving_image, fixed_image = make_batch()
    for iteration in range(0, steps):

        optimizer.zero_grad()
        #with autocast(dtype=torch.float16):
        loss_object = net(moving_image, fixed_image)
        loss = torch.mean(loss_object.all_loss)
        print(loss.dtype)
        scaler.scale(loss).backward()

        print(icon.train.to_floats(loss_object))
        scaler.step(optimizer)
        scaler.update()

if __name__ == "__main__":
    footsteps.initialize("perftest")
    brains = torch.load(
        "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net, steps=5)

    import time
    start = time.monotonic()

    train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net, steps=5)
    end = time.monotonic()

    with open(footsteps.output_dir + 'time', "w") as f:
        f.write(str(end - start))
    print(end - start)

