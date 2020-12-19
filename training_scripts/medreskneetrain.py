import parent
import torch.nn.functional as F
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import inverseConsistentNet
import networks
import data
import describe

BATCH_SIZE = 4
SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
working_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4 

net = inverseConsistentNet.InverseConsistentNet(
    networks.tallishUNet2(dimension=3),
    lmbda=512,
    input_shape=working_shape,
    random_sampling=False,
)

knees, medknees = data.get_knees_dataset()
knees = [F.avg_pool3d(knee, 2) for knee in knees]
if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image[:, None]
    image = image.cuda()
    return image

loss_curve = []
for _ in range(0, 100000):
    optimizer.zero_grad()
    for subbatch in range(8):
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
        torch.save(optimizer.state_dict(), describe.run_dir + "knee_aligner_resi_opt" + str(_))
        torch.save(net.state_dict(), describe.run_dir + "knee_aligner_resi_net" + str(_))
