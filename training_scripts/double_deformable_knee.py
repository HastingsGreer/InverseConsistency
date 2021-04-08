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

BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4


phi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
)
psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.DoubleNet(phi, psi),
    inverseConsistentNet.ssd_only_interpolated,
    70,
)

network_wrappers.assignIdentityMap(net, input_shape)

# load weights
#net_weights = torch.load("results/double_deformable_knee3/knee_aligner_resi_net26400")
#opt_weights = torch.load("results/double_deformable_knee3/knee_aligner_resi_opt26400")

#network_wrappers.adjust_batch_size(net, 32)
#net.load_state_dict(net_weights)
#network_wrappers.adjust_batch_size(net, BATCH_SIZE)

knees = torch.load("/playpen/tgreer/blurred_small_knees")

if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)


net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    for subbatch in range(1):
        optimizer.zero_grad()
        moving_image = make_batch()
        fixed_image = make_batch()
        loss, a, b, c, flips = net_par(moving_image, fixed_image)
        loss = torch.mean(loss) 
        loss.backward()

    if torch.mean(flips).cpu().item() > 25:
        net.lmbda += .1
    print(f"FLIPS DETECTED. lambda = {net.lmbda}")
    loss_curve.append([torch.mean(l.detach().cpu()).item() for l in (a, b, c)] + [flips, net.lmbda])
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
