import parent
import torch.nn.functional as F
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import inverseConsistentNet
import networks
import network_wrappers
import data
import describe

BATCH_SIZE =4 
SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4
phi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
)
psi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet2(dimension=3)
)
net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.DoubleNet(
        network_wrappers.DownsampleNet(network_wrappers.DoubleNet(phi, psi), dimension=3), 
        network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
    ),
    lambda x, y: torch.mean((x - y) ** 2),
    200
)

network_wrappers.assignIdentityMap(net, input_shape)

weights = torch.load("results/hires_finetune7/knee_aligner_resi_net20100")
net.load_state_dict(weights)



knees = torch.load("/playpen/tgreer/knees_big_2xdownsample_train_set")

if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00002)

optimizer_state = torch.load("results/hires_finetune7/knee_aligner_resi_opt20100")

optimizer.load_state_dict(optimizer_state)

net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    if net.lmbda < 800:
        net.lmbda += .05
    optimizer.zero_grad()
    moving_image = make_batch()
    fixed_image = make_batch()
    loss, a, b, c = net_par(moving_image, fixed_image)
    loss = torch.mean(loss)
    loss.backward()

    loss_curve.append([torch.mean(l.detach().cpu()).item() for l in (a, b, c)])
    print(loss_curve[-1])
    optimizer.step()

    if _ % 300 == 0:
        try:
            import pickle

            with open(describe.run_dir + "loss_curve", "wb") as f:
                pickle.dump(loss_curve, f)
        except:
            pass
        torch.save(
            optimizer.state_dict(), describe.run_dir + "knee_aligner_resi_opt" + str(_)
        )
        torch.save(
            net.state_dict(), describe.run_dir + "knee_aligner_resi_net" + str(_)
        )
