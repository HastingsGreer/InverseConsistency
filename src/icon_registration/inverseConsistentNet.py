import torch
from torch import nn
import numpy as np
from .mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import icon_registration.config as config


class InverseConsistentNet(nn.Module):
    def __init__(self, network, similarity, lmbda):

        super(InverseConsistentNet, self).__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):
        
        assert self.identityMap.shape[2:] == image_A.shape[2:]
        assert self.identityMap.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identityMap.isIdentity = True


        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identityMap)
        self.phi_BA_vectorfield = self.phi_BA(self.identityMap)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(self.input_shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1),
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1),
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identityMap
            + torch.randn(*self.identityMap.shape).to(config.device)
            * 1
            / self.identityMap.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        transform_magnitude = torch.mean(
            (self.identityMap - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        return (
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield)
        )
    
class GradientICON(nn.Module):
    def __init__(self, network, similarity, lmbda):

        super(GradientICON, self).__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):
        
        assert self.identityMap.shape[2:] == image_A.shape[2:]
        assert self.identityMap.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identityMap.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identityMap)
        self.phi_BA_vectorfield = self.phi_BA(self.identityMap)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(self.input_shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1),
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1),
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identityMap
            + torch.randn(*self.identityMap.shape).to(config.device)
            * 1
            / self.identityMap.shape[-1]
        )

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = .001

        if len(self.identityMap.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.]]]]).to(config.device)
            dy = torch.Tensor([[[[0.]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identityMap.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.]]], [[[0.]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.]]], [[[delta]]], [[[0.]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.]]], [[[0.]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (inverse_consistency_error - inverse_consistency_error_d) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)
       
        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identityMap - self.phi_AB_vectorfield) ** 2
        )
        return (
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield)
        )


def normalize(image):
    dimension = len(image.shape) - 2
    if dimension == 2:
        dim_reduce = [2, 3]
    elif dimension == 3:
        dim_reduce = [2, 3, 4]
    image_centered = image - torch.mean(image, dim_reduce, keepdim=True)
    stddev = torch.sqrt(torch.mean(image_centered ** 2, dim_reduce, keepdim=True))
    return image_centered / stddev


def ncc(image_A, image_B):
    A = normalize(image_A[:, :1])
    B = normalize(image_B)
    dimension = len(image_A.shape) - 2
    res = torch.mean(A * B)
    return 1 - res


def ssd_only_interpolated(image_A, image_B):
    if len(image_A.shape) - 2 == 3:
        dimensions_to_sum_over = [2, 3, 4]
    else:
        dimensions_to_sum_over = [2, 3]
    inbounds_mask = image_A[:, 1:]
    image_A = image_A[:, :1]
    inbounds_squared_distance = inbounds_mask * (image_A - image_B) ** 2
    sum_squared_distance = torch.sum(inbounds_squared_distance, dimensions_to_sum_over)
    divisor = torch.sum(inbounds_mask, dimensions_to_sum_over)
    ssds = sum_squared_distance / divisor
    return torch.mean(ssds)

def flips(phi):
    if len(phi.size()) == 5:
        a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
        b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
        c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        return torch.sum(dV < 0) / phi.shape[0]
    else:
        ## TODO: implement flips for 2-d registration. shouldn't be hard.
        return -1

