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
            [module.input_shape[:2], module.input_shape[2:] // module.downscale_factor]
        )
    else:
        child_shape = module.input_shape
    for child in module.children():

        # To save memory, these standard torch modules do not need to know what the identity map is.
        blacklist = [
            nn.ModuleList,
            nn.Linear,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ]
        continue_recursion = True
        for module_type in blacklist:
            if isinstance(child, module_type):
                continue_recursion = False
        if continue_recursion:
            assignIdentityMap(child, child_shape)


def adjust_batch_size(model, size):
    shape = model.input_shape
    shape[0] = size
    assignIdentityMap(model, shape)


class FunctionFromVectorField(nn.Module):
    def __init__(self, net):
        super(FunctionFromVectorField, self).__init__()
        self.net = net

    def forward(self, x, y):
        vectorfield_phi = self.net(x, y)

        def ret(input_):
            if (
                False
                and hasattr(input_, "isIdentity")
                and vectorfield_phi.shape == input_.shape
            ):
                return input_ + vectorfield_phi
            else:
                return input_ + compute_warped_image_multiNC(
                    vectorfield_phi, input_, self.spacing, 1
                )
        return ret

class FunctionFromVectorArray(nn.Module):
    def __init__(self, vector_field,spacing):
        super(FunctionFromVectorArray, self).__init__()
        self.vector_field = vector_field
        self.spacing = spacing

    def forward(self):
        vectorfield_phi = self.vector_field
        def ret(input_):
            if (
                    False
                    and hasattr(input_, "isIdentity")
                    and vectorfield_phi.shape == input_.shape
            ):
                return  input_ + vectorfield_phi
            else:
                return input_ + compute_warped_image_multiNC(
                    vectorfield_phi, input_, self.spacing, 1
                )
        return ret


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


class RandomShift(nn.Module):
    def __init__(self, stddev):
        super(RandomShift, self).__init__()
        self.stddev = stddev

    def forward(self, x, y):
        shift_shape = (x.shape[0], len(x.shape - 2))
        shift = self.stddev * torch.randn(shift_shape, device=x.device)
        return lambda input_: input_ + shift


class DoubleNet(nn.Module):
    def __init__(self, netPhi, netPsi):
        super(DoubleNet, self).__init__()
        self.netPsi = netPsi
        self.netPhi = netPhi

    def forward(self, x, y):
        # Tag for optimization. Must be set at the beginning of forward because it is not preserved by .cuda()
        self.identityMap.isIdentity = True
        phi = self.netPhi(x, y)
        phi_vectorfield = phi(self.identityMap)
        self.x_comp_phi = compute_warped_image_multiNC(
            x, phi_vectorfield, self.spacing, 1
        )
        psi = self.netPsi(self.x_comp_phi, y)

        ret = lambda input_: phi(psi(input_))
        return ret


class DownsampleNet(nn.Module):
    def __init__(self, net, dimension):
        super(DownsampleNet, self).__init__()
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        # This member variable is read by assignIdentityMap when
        # walking the network tree and assigning identityMaps
        # to know that all children of this module operate at a lower
        # resolution.
        self.downscale_factor = 2

    def forward(self, x, y):

        x = self.avg_pool(x, 2, ceil_mode=True)
        y = self.avg_pool(y, 2, ceil_mode=True)
        return self.net(x, y)


class AffineFromUNet(nn.Module):
    def __init__(self, unet, dimension=2):
        super(AffineFromUNet, self).__init__()
        self.unet = unet
        self.dimension = dimension

    def forward(self, x, y):
        identityMapCentered = self.identityMap - 0.5
        Minv = torch.inverse(
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
        )
        D = self.unet(x, y)
        if self.dimension == 2:
            b = torch.mean(torch.mean(D, axis=-1), axis=-1)
            A = torch.einsum(
                "imjk,injk->imnjk",
                D,
                identityMapCentered,
            ) + torch.einsum(
                "imjk,injk->imnjk",
                identityMapCentered,
                D,
            )
            A = torch.sum(A, axis=-1)
            A = torch.sum(A, axis=-1)
            A = torch.matmul(A, Minv)

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
        if self.dimension == 3:
            b = torch.mean(torch.mean(torch.mean(D, axis=-1), axis=-1), axis=-1)
            A = torch.einsum(
                "imjkq,injkq->imnjkq",
                D,
                identityMapCentered,
            ) + torch.einsum(
                "imjkq,injkq->imnjkq",
                identityMapCentered,
                D,
            )
            A = torch.sum(A, axis=-1)
            A = torch.sum(A, axis=-1)
            A = torch.sum(A, axis=-1)
            A = torch.matmul(A, Minv)

            b = torch.reshape(b, (-1, 3, 1))
            x = torch.cat([A, b], axis=-1)

            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 0, 1]]])
                    .cuda()
                    .expand(x.shape[0], -1, -1, -1),
                ],
                1,
            )
            x = (
                x
                + torch.Tensor(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
                ).cuda()
            )
            x = torch.matmul(
                torch.Tensor([[1, 0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]).cuda(), x
            )
            x = torch.matmul(
                x, torch.Tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]).cuda()
            )
            return x
