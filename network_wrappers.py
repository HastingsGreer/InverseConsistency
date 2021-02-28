import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN


def multiply_matrix_vectorfield(matrix, vectorfield):
    dimension = len(vectorfield.shape) - 2

    if dimension == 2:
        batch_matrix_multiply = "ijkl,imj->imkl"
    else:
        batch_matrix_multiply = "ijkln,imj->imkln"
    return torch.einsum(batch_matrix_multiply, vectorfield, matrix)


def assignIdentityMap(module, input_shape):
    module.input_shape = np.array(input_shape)
    module.spacing = 1.0 / (module.input_shape[2::] - 1)

    _id = identity_map_multiN(module.input_shape, module.spacing)
    module.register_buffer("identityMap", torch.from_numpy(_id))

    if "downscale_factor" in vars(module):
        child_shape = np.concatenate(
            [module.input_shape[:2], module.input_shape[2:] / module.downscale_factor]
        )
    else:
        child_shape = module.input_shape
    for child in module.children():
        assignIdentityMap(child, child_shape)


class FunctionFromVectorField(nn.Module):
    def __init__(self, net):
        super(FunctionFromVectorField, self).__init__()
        self.net = net

    def forward(self, x, y):
        vectorfield_phi = self.net(x, y)
        return lambda input_: input_ + compute_warped_image_multiNC(
            vectorfield_phi, input_, self.spacing, 1
        )


class FunctionFromMatrix(nn.Module):
    def __init__(self, net):
        super(FunctionFromMatrix, self).__init__()
        self.net = net

    def forward(self, x, y):
        matrix_phi = self.net(x, y)

        def ret(input_):
            shape = list(input_.shape)
            shape[1] = 1
            input_homogeneous = torch.cat(
                [input_, torch.ones(shape, device=input_.device)], axis=1
            )
            return multiply_matrix_vectorfield(matrix_phi, input_homogeneous)[:, :-1]

        return ret


class DoubleAffineNet(nn.Module):
    def __init__(self, netPhi, netPsi):
        super(DoubleAffineNet, self).__init__()
        self.netPsi = netPsi
        self.netPhi = netPhi

    def forward(self, x, y):
        shape = list(self.identityMap.shape)
        shape[1] = 1
        id_homogeneous = torch.cat(
            [self.identityMap, torch.ones(shape, device=torch.cuda)], axis=1
        )
        phi = self.netPsi(x, y)
        phi_inv_map = multiply_matrix_vectorfield(torch.inverse(phi), id_homogeneous)
        y_comp_phi_inv = compute_warped_image_multiNC(
            y, phi_inv_map[:, : len(self.spacing), :, :], self.spacing, 1
        )
        psi = self.netPhi(x, y_comp_phi_inv)
        if len(self.spacing) == 2:
            identityM = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        elif len(self.spacing) == 3:
            identityM = torch.tensor(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
            )
        return phi + psi - identityM.cuda()


class DoubleNet(nn.Module):
    def __init__(self, netPhi, netPsi):
        super(DoubleNet, self).__init__()
        self.netPsi = netPsi
        self.netPhi = netPhi

    def forward(self, x, y):

        phi = self.netPhi(x, y)
        phi_vectorfield = phi(self.identityMap)
        self.x_comp_phi = compute_warped_image_multiNC(x, phi_vectorfield, self.spacing, 1)
        psi = self.netPsi(self.x_comp_phi, y)

        ret = lambda input_: phi(psi(input_))
        return ret


class DownsampleNet(nn.Module):
    def __init__(self, net, dimension):
        super(DownsampleNet, self)
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        self.downscale_factor = 2

    def forward(self, x, y):

        x = self.avg_pool(x, 2, ceil_mode=True)
        y = self.avg_pool(y, 2, ceil_mode=True)
        return self.net(x, y)


class AffineFromUNet(nn.Module):
    def __init__(self, unet, identityMap, dimension=2):
        super(AffineFromUNet, self).__init__()
        self.unet = unet
        self.dimension = dimension
        self.register_buffer("identityMapCentered", identityMap - 0.5)
        self.register_buffer(
            "Minv",
            torch.inverse(
                torch.sum(
                    torch.sum(
                        torch.einsum(
                            "imjk,injk->imnjk",
                            self.identityMapCentered,
                            self.identityMapCentered,
                        ),
                        axis=-1,
                    ),
                    axis=-1,
                )
            ),
        )

    def forward(self, x, y):
        D = self.unet(x, y)
        if self.dimension == 2:
            b = torch.mean(torch.mean(D, axis=-1), axis=-1)
            A = torch.einsum(
                "imjk,injk->imnjk",
                D,
                self.identityMapCentered,
            ) + torch.einsum(
                "imjk,injk->imnjk",
                self.identityMapCentered,
                D,
            )
            A = torch.sum(A, axis=-1)
            A = torch.sum(A, axis=-1)
            A = torch.matmul(A, self.Minv)

            b = torch.reshape(b, (-1, 2, 1))
            x = torch.cat([A, b], axis=-1)

            x = torch.cat(
                [x, torch.Tensor([[[0, 0, 1]]]).cuda().expand(x.shape[0], -1, -1)], 1
            )
            x = x + torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).cuda()
            x = torch.matmul(
                torch.Tensor([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]).cuda(), x
            )
            x = torch.matmul(
                x, torch.Tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]).cuda()
            )
            return x
