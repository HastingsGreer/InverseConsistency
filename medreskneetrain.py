from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import inverseConsistentNet
import networks
import data

BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
working_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]


net = inverseConsistentNet.InverseConsistentNet(
    networks.tallUNet(dimension=3),
    lmbda=512,
    input_shape=working_shape,
    random_sampling=False,
)
net.load_state_dict(torch.load("network_weights/lowres_knee_network"))

knees, medknees = data.get_knees_dataset()


net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

net_par.train()


def make_batch():
    image = torch.cat([random.choice(medknees) for _ in range(4 * BATCH_SIZE)])
    image = image.reshape(4 * BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE)
    image = image.cuda()
    return image


for _ in range(0, 100000):
    optimizer.zero_grad()
    for subbatch in range(1):
        moving_image = make_batch()
        fixed_image = make_batch()
        loss, a, b, c = net_par(moving_image, fixed_image)
        loss = torch.mean(loss)
        loss.backward()

    print([torch.mean(l.detach().cpu()).item() for l in (a, b, c)])

    optimizer.step()

    if _ % 300 == 0:
        torch.save(optimizer.state_dict(), "results/knee_aligner_resi_opt" + str(_))
        torch.save(net.state_dict(), "results/knee_aligner_resi_net" + str(_))
