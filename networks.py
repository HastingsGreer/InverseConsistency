import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN


class Autoencoder(nn.Module):
    def __init__(self, num_layers, channels):
        super(Autoencoder, self).__init__()
        self.num_layers = num_layers
        down_channels = channels[0]
        up_channels = channels[1]
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        for depth in range(self.num_layers):
            self.downConvs.append( 
                nn.Conv2d(
                    down_channels[depth], 
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2
                )
            )
            self.upConvs.append(
                nn.ConvTranspose2d(
                    up_channels[depth + 1], 
                    up_channels[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2
                )
            )
        self.lastConv = nn.Conv2d(
            16, 2, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)
     
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.relu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.relu(self.upConvs[depth](x))
            x = x[:, :, :skips[depth].size()[2], :skips[depth].size()[3]]
        x = self.lastConv(x)
        return x / 10

def tallAE():
    return Autoencoder(5, np.array([
        [2, 16, 32, 64, 256, 512],
        [16, 32, 64, 128, 256, 512],
    ]))

class Residual(nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.bn2 = nn.BatchNorm2d(num_features=features)
        
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
    def forward(self, x):
        y = F.relu(self.bn1(x))
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        return y + x

class UNet(nn.Module):
    def __init__(self, num_layers, channels):
        super(UNet, self).__init__()
        self.num_layers = num_layers
        down_channels = channels[0]
        up_channels_out = channels[1]
        up_channels_in = channels[2]
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        #self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm2d(num_features = up_channels_out[_])
             for _ in range(self.num_layers)])
        for depth in range(self.num_layers):
            self.downConvs.append( 
                nn.Conv2d(
                    down_channels[depth], 
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2
                )
            )
            self.upConvs.append(
                nn.ConvTranspose2d(
                    up_channels_in[depth], 
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2
                )
            )
            #self.residues.append(
            #    Residual(up_channels_out[depth])    
            #)
        self.lastConv = nn.Conv2d(
            18, 2, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)
     
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            x = F.relu(self.downConvs[depth](x))
        for depth in reversed(range(self.num_layers)):
            x = F.relu(
                self.upConvs[depth](x))
            x = self.batchNorms[depth](x)
            
            x = x[:, :, :skips[depth].size()[2], :skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10

def pad_or_crop(x, shape):
    y = x[:, :shape[1]]
    if x.size()[1] < shape[1]:
        
        y = F.pad(y, (0, 0, 0, 0, shape[1]-x.size()[1], 0))
    assert(y.size()[1] == shape[1])
    
    return y
    
class UNet2(nn.Module):
    def __init__(self, num_layers, channels):
        super(UNet2, self).__init__()
        self.num_layers = num_layers
        down_channels = channels[0]
        up_channels_out = channels[1]
        up_channels_in = channels[2]
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm2d(num_features = up_channels_out[_])
             for _ in range(self.num_layers)])
        for depth in range(self.num_layers):
            self.downConvs.append( 
                nn.Conv2d(
                    down_channels[depth], 
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2
                )
            )
            self.upConvs.append(
                nn.ConvTranspose2d(
                    up_channels_in[depth], 
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2
                )
            )
            self.residues.append(
                Residual(up_channels_out[depth])    
            )
        self.lastConv = nn.Conv2d(
            18, 2, kernel_size=3, padding=1)
        torch.nn.init.zeros_(self.lastConv.weight)
     
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(F.avg_pool2d(x, 2, ceil_mode=True), y.size())
            
        for depth in reversed(range(self.num_layers)):
            y = self.upConvs[depth](F.leaky_relu(x))
            x = y + F.interpolate(pad_or_crop(x, y.size()), scale_factor=2, mode="bilinear")
            #x = self.residues[depth](x)
            #x = self.batchNorms[depth](x)
            
            x = x[:, :, :skips[depth].size()[2], :skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        return x / 10
    
def tallUNet():
    return UNet(5, np.array([
        [2, 16, 32, 64, 256, 512],
        [16, 32, 64, 128, 256],
        [48, 96, 192, 512, 512]
    ]))

def tallUNet2():
    return UNet2(5, np.array([
        [2, 16, 32, 64, 256, 512],
        [16, 32, 64, 128, 256],
        [48, 96, 192, 512, 512]
    ]))


class RegisNet(nn.Module):
    def __init__(self):
        super(RegisNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(22, 10, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 10, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(42, 10, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(52, 2, kernel_size=5, padding=2)

    def forward(self, x, y):
        x= torch.cat([x, y], 1)
        
        x = torch.cat([x, F.relu(self.conv1(x))], 1)
        x = torch.cat([x, F.relu(self.conv2(x))], 1)
        x = torch.cat([x, F.relu(self.conv3(x))], 1)
        x = torch.cat([x, F.relu(self.conv4(x))], 1)
        x = torch.cat([x, F.relu(self.conv5(x))], 1)
        
        return self.conv6(x)
    
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.dense1 = nn.Linear(28 * 28 * 2, 8000)
        self.dense2 = nn.Linear(8000, 3000)
        self.dense3 = nn.Linear(3000, 28 * 28 * 2)
        torch.nn.init.zeros_(self.dense3.weight)
    def forward(self, x, y):
        x = torch.reshape(torch.cat([x, y], 1), (-1, 2 * 28 * 28))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        x = torch.reshape(x, (-1, 2, 28, 28))
        return x
        
#Here we define our inverse consistency loss

class InverseConsistentNet(nn.Module):
    def __init__(self, network, lmbda, input_shape, random_sampling=True):
        super(InverseConsistentNet, self).__init__()       
        
        self.sz = np.array(input_shape)
        self.spacing = 1./(self.sz[2::]-1) 
        
        _id = identity_map_multiN(self.sz, self.spacing)
        self.identityMap = torch.from_numpy(_id).cuda()
        self.map_shape = self.identityMap.shape
        self.regis_net = network().cuda()
        self.lmbda = lmbda
        
        self.random_sampling=random_sampling
        

    def forward(self, image_A, image_B):
        

        self.D_AB = self.regis_net(image_A, image_B) 
        self.phi_AB = self.D_AB + self.identityMap
        
        self.D_BA = self.regis_net(image_B, image_A)
        self.phi_BA = self.D_BA + self.identityMap
        
        self.warped_image_A = compute_warped_image_multiNC(
            image_A, self.phi_AB, self.spacing, 1)
        
        self.warped_image_B = compute_warped_image_multiNC(
            image_B, self.phi_BA, self.spacing, 1)
        
        Iepsilon = self.identityMap + torch.randn(*self.map_shape).cuda() * 1/self.map_shape[-1]
        
        D_BA_epsilon = compute_warped_image_multiNC(self.D_BA, Iepsilon, self.spacing, 1)
        
        

        self.approximate_identity = compute_warped_image_multiNC( 
                self.D_AB, D_BA_epsilon + Iepsilon, self.spacing, 1
            ) + D_BA_epsilon

        inverse_consistency_loss = self.lmbda * torch.mean(
                (self.approximate_identity)**2
        )
        similarity_loss = (
            torch.mean((self.warped_image_A - image_B)**2) + 
            torch.mean((self.warped_image_B - image_A)**2)
        )
        transform_magnitude= self.lmbda * torch.mean(
            (self.identityMap - self.phi_AB)**2
        )
        self.all_loss =  inverse_consistency_loss + similarity_loss
        return [x.item() for x in (inverse_consistency_loss, similarity_loss, transform_magnitude)]
