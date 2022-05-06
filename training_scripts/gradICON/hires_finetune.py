import torch.nn.functional as F
from icon_registration.mermaidlite import (
    compute_warped_image_multiNC,
    identity_map_multiN,
)
import torch
import random
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.data as data
import footsteps


BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4

phi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
)
psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

pretrained_lowres_net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.DoubleNet(phi, psi),
    inverseConsistentNet.ssd_only_interpolated,
    100,
)

pretrained_lowres_net.assign_identity_map(input_shape)


trained_weights = torch.load("results/gradknee-2/knee_aligner_resi_net8100")

pretrained_lowres_net.load_state_dict(trained_weights)

hires_net = inverseConsistentNet.GradientICON(
    network_wrappers.DoubleNet(
        network_wrappers.DownsampleNet(pretrained_lowres_net.regis_net, dimension=3),
        network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
    ),
    inverseConsistentNet.ssd_only_interpolated,
    0.2,
)
BATCH_SIZE = 8
SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]
hires_net.assign_identity_map(input_shape)

# for p in hires_net.regis_net.netPhi.parameters():
#    p.requires_grad = False
knees = torch.load("/playpen-ssd/tgreer/knees_big_2xdownsample_train_set")

if GPUS == 1:
    net_par = hires_net.cuda()
else:
    net_par = torch.nn.DataParallel(hires_net).cuda()
optimizer = torch.optim.Adam(
    (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
)


net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    optimizer.zero_grad()
    moving_image = make_batch()
    fixed_image = make_batch()
    loss, a, b, c, flips = net_par(moving_image, fixed_image)
    loss = torch.mean(loss)
    loss.backward()

    loss_curve.append(
        [torch.mean(l.detach().cpu()).item() for l in (a, b, c)]
        + [flips, hires_net.lmbda]
    )
    print(loss_curve[-1])
    optimizer.step()

    if _ % 300 == 0:
        try:
            import pickle

            with open(footsteps.output_dir + "loss_curve", "wb") as f:
                pickle.dump(loss_curve, f)
        except:
            pass
        torch.save(
            optimizer.state_dict(),
            footsteps.output_dir + "knee_aligner_resi_opt" + str(_),
        )
        torch.save(
            hires_net.regis_net.state_dict(),
            footsteps.output_dir + "knee_aligner_resi_net" + str(_),
        )
