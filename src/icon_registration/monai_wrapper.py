import icon_registration
import torch


class ConcatInputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        return self.net(torch.cat([image_A, image_B], axis=1))

class FirstChannelInputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        return self.net(image_A[:, :1], image_B[:, :1])


def make_ddf(module: icon_registration.RegistrationModule, image_A, image_B):
    res = module(image_A, image_B)
    if type(res) == icon_registration.losses.ICONLoss:
        # Module is a loss function wrapping a registration process

        field_0_1 = module.phi_AB_vectorfield - module.identity_map

    else:
        field_0_1 = res(module.identity_map) - module.identity_map
        
    network_shape_list = list(module.identity_map.shape[2:])

    scale = torch.Tensor(network_shape_list).to(module.identity_map.device)

    for _ in network_shape_list:
        scale = scale[:, None]
    scale = scale[None, :]
    field_spacing_1 = scale * field_0_1
