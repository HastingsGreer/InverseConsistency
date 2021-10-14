import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        import icon_registration
        import icon_registration.inverseConsistentNet
        import icon_registration.networks
        import icon_registration.data
        import icon_registration.network_wrappers

    def test_pytorch_cuda(self):
        import torch

        x = torch.Tensor([7]).cuda
