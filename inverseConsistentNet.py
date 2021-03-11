import torch
from torch import nn
import numpy as np
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
from network_wrappers import FunctionFromVectorArray

class InverseConsistentNet(nn.Module):
    def __init__(self, network, similarity, lmbda,noise_std,lmbda_noise):
        super(InverseConsistentNet, self).__init__()
        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity
        self.noise_standard_deviation = noise_std
        self.lmbda_noise = lmbda_noise

    def forward(self, image_A, image_B):
        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identityMap.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identityMap)
        self.phi_BA_vectorfield = self.phi_BA(self.identityMap)

        self.warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_AB_vectorfield, self.spacing, 1
        )
        self.warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_BA_vectorfield, self.spacing, 1
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identityMap
            + torch.randn(*self.identityMap.shape).cuda()
            * 1
            / self.identityMap.shape[-1]
        )
        Inoise1 = (
            self.identityMap
            + torch.randn(*self.identityMap.shape).cuda()
            * self.noise_standard_deviation
            / self.identityMap.shape[-1]
        )
        Inoise1 = FunctionFromVectorArray(Inoise1,self.spacing)()
        Inoise2 = (
                self.identityMap
                + torch.randn(*self.identityMap.shape).cuda()
                * self.noise_standard_deviation
                / self.identityMap.shape[-1]
        )
        Inoise2 = FunctionFromVectorArray(Inoise2,self.spacing)()
        # inverse consistency one way
        approximate_Iepsilon1_with_no_noise = self.phi_AB(self.phi_BA(Iepsilon))
        # computation of the approximate inverse consistency with noise
        temp = self.phi_BA(Iepsilon)

        approx_1 = Inoise1(temp)
        approximate_Iepsilon1 = Inoise2(self.phi_AB(approx_1))

        approximate_Iepsilon2_with_no_noise = self.phi_BA(self.phi_AB(Iepsilon))
        # computation of the approximate inverse consistency with noise
        approx_2 = Inoise2(self.phi_AB(Iepsilon))
        approximate_Iepsilon2 = Inoise1(self.phi_BA(approx_2))

        inverse_consistency_loss_with_noise = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)
        inverse_consistency_loss_no_noise = torch.mean(
            (Iepsilon - approximate_Iepsilon1_with_no_noise) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2_with_no_noise) ** 2)

        transform_magnitude = torch.mean(
            (self.identityMap - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss_no_noise + similarity_loss + self.lmbda_noise *inverse_consistency_loss_with_noise

        return (
            all_loss,
            inverse_consistency_loss_no_noise,
            similarity_loss,
            transform_magnitude,
        )
