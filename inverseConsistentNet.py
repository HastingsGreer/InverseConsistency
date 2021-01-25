import torch
from torch import nn
import numpy as np
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN


class InverseConsistentNet(nn.Module):
    def __init__(self, network, lmbda, input_shape, random_sampling=True):
        super(InverseConsistentNet, self).__init__()

        self.sz = np.array(input_shape)
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))
        self.map_shape = self.identityMap.shape
        self.regis_net = network
        self.lmbda = lmbda

        self.random_sampling = random_sampling

    def adjust_batch_size(self, BATCH_SIZE):
        self.sz[0] = BATCH_SIZE
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))

    def forward(self, image_A, image_B):
        # Compute Displacement Maps
        self.D_AB = self.regis_net(image_A, image_B)
        self.phi_AB = self.D_AB + self.identityMap

        self.D_BA = self.regis_net(image_B, image_A)
        self.phi_BA = self.D_BA + self.identityMap

        # Compute Image similarity

        self.warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_AB, self.spacing, 1
        )

        self.warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_BA, self.spacing, 1
        )

        similarity_loss = torch.mean((self.warped_image_A - image_B) ** 2) + torch.mean(
            (self.warped_image_B - image_A) ** 2
        )

        # Compute Inverse Consistency
        # One way

        Iepsilon = (
            self.identityMap
            + torch.randn(*self.map_shape).cuda() * 1 / self.map_shape[-1]
        )

        D_BA_epsilon = compute_warped_image_multiNC(
            self.D_BA, Iepsilon, self.spacing, 1
        )

        self.approximate_identity = (
            compute_warped_image_multiNC(
                self.D_AB, D_BA_epsilon + Iepsilon, self.spacing, 1
            )
            + D_BA_epsilon
        )
        # And the Other
        D_AB_epsilon = compute_warped_image_multiNC(
            self.D_AB, Iepsilon, self.spacing, 1
        )

        self.approximate_identity2 = (
            compute_warped_image_multiNC(
                self.D_BA, D_AB_epsilon + Iepsilon, self.spacing, 1
            )
            + D_AB_epsilon
        )

        inverse_consistency_loss = self.lmbda * torch.mean(
            (self.approximate_identity) ** 2 + (self.approximate_identity2) ** 2
        )
        transform_magnitude = self.lmbda * torch.mean(
            (self.identityMap - self.phi_AB) ** 2
        )
        self.all_loss = inverse_consistency_loss + similarity_loss
        return [
            x
            for x in (
                self.all_loss,
                inverse_consistency_loss,
                similarity_loss,
                transform_magnitude,
            )
        ]


class InverseConsistentAffineNet(nn.Module):
    def __init__(self, network, lmbda, input_shape):
        super(InverseConsistentAffineNet, self).__init__()

        self.sz = np.array(input_shape)
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))

        _id_projective = np.concatenate([_id, np.ones(input_shape)], axis=1)
        self.register_buffer(
            "identityMapProjective", torch.from_numpy(_id_projective).float()
        )
        self.map_shape = self.identityMap.shape
        self.regis_net = network
        self.lmbda = lmbda

    def adjust_batch_size(self, BATCH_SIZE):
        self.sz[0] = BATCH_SIZE
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))

    def forward(self, image_A, image_B):
        # Compute Displacement Maps

        batch_matrix_multiply = "ijkl,imj->imkl"
        self.matrix_AB = self.regis_net(image_A, image_B)

        self.phi_AB = torch.einsum(
            batch_matrix_multiply, self.identityMapProjective, self.matrix_AB
        )

        self.matrix_BA = self.regis_net(image_B, image_A)

        self.phi_BA = torch.einsum(
            batch_matrix_multiply, self.identityMapProjective, self.matrix_BA
        )

        # Compute Image similarity

        self.warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_AB[:, :2], self.spacing, 1
        )

        self.warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_BA[:, :2], self.spacing, 1
        )

        similarity_loss = torch.mean((self.warped_image_A - image_B) ** 2) + torch.mean(
            (self.warped_image_B - image_A) ** 2
        )

        # Compute Inverse Consistency
        # One way

        self.approximate_zero = (
            torch.einsum(batch_matrix_multiply, self.phi_AB, self.matrix_BA)[:, :2]
            - self.identityMap
        )
        self.approximate_zero2 = (
            torch.einsum(batch_matrix_multiply, self.phi_BA, self.matrix_AB)[:, :2]
            - self.identityMap
        )

        inverse_consistency_loss = self.lmbda * torch.mean(
            (self.approximate_zero) ** 2 + (self.approximate_zero2) ** 2
        )
        transform_magnitude = self.lmbda * torch.mean(
            (self.identityMap - self.phi_AB[:, :2]) ** 2
        )
        self.all_loss = inverse_consistency_loss + similarity_loss
        return [
            x
            for x in (
                self.all_loss,
                inverse_consistency_loss,
                similarity_loss,
                transform_magnitude,
            )
        ]


class InverseConsistentAffineDeformableNet(nn.Module):
    def __init__(self, affine_network, network, lmbda, input_shape):
        super(InverseConsistentAffineDeformableNet, self).__init__()

        self.sz = np.array(input_shape)
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))

        _id_projective = np.concatenate([_id, np.ones(input_shape)], axis=1)
        self.register_buffer(
            "identityMapProjective", torch.from_numpy(_id_projective).float()
        )

        self.map_shape = self.identityMap.shape

        self.affine_regis_net = affine_network
        self.regis_net = network

        self.lmbda = lmbda

    def adjust_batch_size(self, BATCH_SIZE):
        self.sz[0] = BATCH_SIZE
        self.spacing = 1.0 / (self.sz[2::] - 1)

        _id = identity_map_multiN(self.sz, self.spacing)
        self.register_buffer("identityMap", torch.from_numpy(_id))

        _id_projective = np.concatenate([_id, np.ones(input_shape)], axis=1)
        self.register_buffer(
            "identityMapProjective", torch.from_numpy(_id_projective).float()
        )

    def forward(self, image_A, image_B):
        # Compute Displacement Maps

        batch_matrix_multiply = "ijkl,imj->imkl"
        self.matrix_AB = self.affine_regis_net(image_A, image_B)

        self.phi_AB_affine = torch.einsum(
            batch_matrix_multiply, self.identityMapProjective, self.matrix_AB
        )

        self.phi_AB_affine_inv = torch.einsum(
            batch_matrix_multiply,
            self.identityMapProjective,
            torch.inverse(self.matrix_AB),
        )

        self.matrix_BA = self.affine_regis_net(image_B, image_A)

        self.phi_BA_affine = torch.einsum(
            batch_matrix_multiply, self.identityMapProjective, self.matrix_BA
        )

        self.phi_BA_affine_inv = torch.einsum(
            batch_matrix_multiply,
            self.identityMapProjective,
            torch.inverse(self.matrix_BA),
        )

        # resample using affine for deformable step. Use inverse to get residue in correct coordinate space

        self.affine_warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_AB_affine_inv[:, :2], self.spacing, 1
        )

        self.affine_warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_BA_affine_inv[:, :2], self.spacing, 1
        )

        self.D_AB = nn.functional.pad(
            self.regis_net(image_A, self.affine_warped_image_B), (0, 0, 0, 0, 0, 1)
        )
        self.phi_AB = self.phi_AB_affine + self.D_AB

        self.D_BA = nn.functional.pad(
            self.regis_net(image_B, self.affine_warped_image_A), (0, 0, 0, 0, 0, 1)
        )
        self.phi_BA = self.phi_BA_affine + self.D_BA

        # Compute Image similarity

        self.warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_AB[:, :2], self.spacing, 1
        )

        self.warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_BA[:, :2], self.spacing, 1
        )

        similarity_loss = torch.mean((self.warped_image_A - image_B) ** 2) + torch.mean(
            (self.warped_image_B - image_A) ** 2
        )

        # Compute Inverse Consistency
        # One way

        self.approximate_zero = (
            torch.einsum(batch_matrix_multiply, self.phi_AB, self.matrix_BA)[:, :2]
            + compute_warped_image_multiNC(
                self.D_BA[:, :2], self.phi_AB[:, :2], self.spacing, 1
            )
            - self.identityMap
        )
        self.approximate_zero2 = (
            torch.einsum(batch_matrix_multiply, self.phi_BA, self.matrix_AB)[:, :2]
            + compute_warped_image_multiNC(
                self.D_AB[:, :2], self.phi_BA[:, :2], self.spacing, 1
            )
            - self.identityMap
        )
        inverse_consistency_loss = self.lmbda * torch.mean(
            (self.approximate_zero) ** 2 + (self.approximate_zero2) ** 2
        )
        transform_magnitude = self.lmbda * torch.mean(
            (self.identityMap - self.phi_AB[:, :2]) ** 2
        )
        self.all_loss = inverse_consistency_loss + similarity_loss
        return [
            x
            for x in (
                self.all_loss,
                inverse_consistency_loss,
                similarity_loss,
                transform_magnitude,
            )
        ]
