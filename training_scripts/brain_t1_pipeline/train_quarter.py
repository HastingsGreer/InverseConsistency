
import torch.nn.functional as F
from icon_registration.mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.data as data
import footsteps



BATCH_SIZE = 32
input_shape = [BATCH_SIZE, 1, 65, 77, 65]

GPUS = 4
def make_network():
    phi = network_wrappers.FunctionFromVectorField(
        networks.tallUNet2(dimension=3)
    )
    psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))


    hires_net = inverseConsistentNet.GradientICON(
        network_wrappers.DoubleNet(phi, psi),
        inverseConsistentNet.ssd_only_interpolated,
        .05,
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
    brains = torch.load("/playpen-ssd/tgreer/ICON_brain_preprocessed_data/quarter_res_stripped/brain_train_4xdown_scaled")
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(
        (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
    )


    net_par.train()




    loss_curve = []
    for _ in range(0, 100000):
        optimizer.zero_grad()
        moving_image = make_batch()
        fixed_image = make_batch()
        loss, a, b, c, flips = net_par(moving_image, fixed_image)
        loss = torch.mean(loss)
        loss.backward()

        loss_curve.append([torch.mean(l.detach().cpu()).item() for l in (a, b, c)] + [flips, hires_net.lmbda])
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
                optimizer.state_dict(), footsteps.output_dir + "brain_aligner_resi_opt" + str(_)
            )
            torch.save(
                hires_net.regis_net.state_dict(), footsteps.output_dir + "brain_aligner_resi_net" + str(_)
            )
