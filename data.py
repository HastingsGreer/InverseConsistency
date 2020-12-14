import torch
import torchvision
import numpy as np
def get_dataset_mnist(split, number=5):
    ds = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./files/",
            transform=torchvision.transforms.ToTensor(),
            download=True,
            train=(split == "train")
        ),
        batch_size=500
    )
    images = []
    for _, batch in enumerate(ds):
        label = np.array(batch[1])
        batch_nines = label == number
        images.append(np.array(batch[0])[batch_nines])
    images = np.concatenate(images)

    ds = torch.utils.data.TensorDataset(torch.Tensor(images))
    d1, d2 = (torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, ) for _ in (1,1))
    return d1, d2


def get_dataset_triangles(split, data_size=128, hollow=False):
    x, y = np.mgrid[0:1:data_size * 1j, 0:1:data_size*1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((6000, 1, 1)) * .3 + .4
    cy = np.random.random((6000, 1, 1)) * .3 + .4
    r = np.random.random((6000, 1, 1)) * .2 + .2
    theta = np.random.random((6000, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((6000, 1, 1)) > .5

    triangles = (np.sqrt((x - cx)**2 + (y - cy)**2) 
    - r * np.cos(np.pi / 3) / np.cos((np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3))

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx)**2 + (y - cy)**2) - r) )
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(images, 1)))
    d1, d2 = (torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, ) for _ in (1,1))
    return d1, d2


