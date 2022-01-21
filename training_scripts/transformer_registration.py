
import torch
from torch import nn
#image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
class RegistrationTransformer(nn.Module):
    def __init__(self, size):
        self.size = size
        super(RegistrationTransformer, self).__init__()
        
        feature_n = 120
        self.position_embedding = nn.Parameter(
            torch.randn((49, 1, feature_n)) / 10
        )
        
        self.embed = nn.Conv2d(
                    1,
                    feature_n,
                    kernel_size=4,
                    padding=0,
                    stride=4,
                ).cuda()
        
        self.decode = nn.ConvTranspose2d(
                    feature_n,
                    2,
                    kernel_size=4,
                    padding=0,
                    stride=4,
                ).cuda()
        
        #torch.nn.init.zeros_(self.decode.weight)
        #torch.nn.init.zeros_(self.decode.bias)



        
        self.t = torch.nn.Transformer(d_model=120)
        
        
        
        
    def forward(self, x, y):
        x = self.embed_and_reshape(x)
        y = self.embed_and_reshape(y)
        
        out = self.t(x, y)
        out = self.reshape_and_decode(out) / 20
        
        return out
        
        
    def embed_and_reshape(self, a):
        a = self.embed(a)
        a = a.reshape((a.shape[0], a.shape[1], a.shape[2] * a.shape[3]))
        a = a.permute(2, 0, 1)
        a = a + self.position_embedding
        return a
    
    def reshape_and_decode(self, seq):
        a = seq.permute(1, 2, 0)
        size = int(math.sqrt(a.shape[-1]))
        a = a.reshape(a.shape[0], a.shape[1], size, size)
        a = self.decode(a)
        return a
    

import icon_registration.data as data
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.visualize as visualize
import icon_registration.train as train
import icon_registration.inverseConsistentNet as inverseConsistentNet
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import pickle

import footsteps

import argparse
import math


batch_size = 128

d1, d2 = data.get_dataset_triangles(
    "train", data_size=28, hollow=False, batch_size=batch_size
)
d1_t, d2_t = data.get_dataset_triangles(
    "test", data_size=28, hollow=False, batch_size=batch_size
)


lmbda = 150
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
print("=" * 50)
net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.FunctionFromVectorField(RegistrationTransformer(28)),
    lambda x, y: torch.mean((x[:1] - y[:1]) ** 2),
    lmbda,
)

input_shape = next(iter(d1))[0].size()
network_wrappers.assignIdentityMap(net, input_shape)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.train()


xs = []


for _ in range(0,130):
    y = np.array(train.train2d(net, optimizer, d1, d2, epochs=50))
    xs.append(y)
    x = np.concatenate(xs)
    plt.title(
        "Loss curve for " + type(net.regis_net).__name__ + " lambda=" + str(lmbda)
    )
    plt.plot(x[:, :3])
    plt.savefig(footsteps.output_dir + f"loss.png")
    plt.clf()
    plt.title("Log # pixels with negative Jacobian per epoch")
    plt.plot(x[:, 3])
    # random.seed(1)
    plt.savefig(footsteps.output_dir + f"lossj.png")
    plt.clf()
    with open(footsteps.output_dir + "loss.pickle", "wb") as f:
        pickle.dump(x, f)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    # np.random.seed(1)
    image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
    for N in range(3):
        visualize.visualizeRegistration(
            net,
            image_A,
            image_B,
            N,
            footsteps.output_dir + f"epoch{_:03}" + "case" + str(N) + ".png",
        )

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
image_A, image_B = (x[0].cuda() for x in next(zip(d1_t, d2_t)))
os.mkdir(footsteps.output_dir + "final/")
for N in range(30):
    visualize.visualizeRegistrationCompact(net, image_A, image_B, N)
    plt.savefig(footsteps.output_dir + f"final/{N}.png")
    plt.clf()

torch.save(net.state_dict(), footsteps.output_dir + "network.trch")
torch.save(optimizer.state_dict(), footsteps.output_dir + "opt.trch")
