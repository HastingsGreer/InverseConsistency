import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from icon_registration import config


def get_dataset_mnist(split, number=5):
    ds = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./files/",
            transform=torchvision.transforms.ToTensor(),
            download=True,
            train=(split == "train"),
        ),
        batch_size=500,
    )
    images = []
    for _, batch in enumerate(ds):
        label = np.array(batch[1])
        batch_nines = label == number
        images.append(np.array(batch[0])[batch_nines])
    images = np.concatenate(images)

    ds = torch.utils.data.TensorDataset(torch.Tensor(images))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=128,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_1d(data_size=128, samples=6000, batch_size=128):
    x = np.arange(0, 1, 1 / data_size)
    x = np.reshape(x, (1, data_size))
    cx = np.random.random((samples, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1)) * 0.2 + 0.2

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2) - r))

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(circles, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_triangles(
    split=None, data_size=128, hollow=False, samples=6000, batch_size=128
):
    x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((samples, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((samples, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(images, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


def get_dataset_retina(
    extra_deformation=False,
    downsample_factor=4,
    blur_sigma=None,
    warps_per_pair=20,
    fixed_vertical_offset=None,
    include_boundary=False,
):
    try:
        import elasticdeform
        import hub
    except:

        raise Exception(
            """the retina dataset requires the dependencies hub and elasticdeform.
            Try pip install hub elasticdeform"""
        )

    ds_name = f"retina{extra_deformation}{downsample_factor}{blur_sigma}{warps_per_pair}{fixed_vertical_offset}{include_boundary}.trch"

    import os

    if os.path.exists(ds_name):
        augmented_ds1_tensor, augmented_ds2_tensor = torch.load(ds_name)
    else:

        res = []
        for batch in hub.load("hub://activeloop/drive-train").pytorch(
            num_workers=0, batch_size=4, shuffle=False
        ):
            if include_boundary:
                res.append(batch["manual_masks/mask"] ^ batch["masks/mask"])
            else:
                res.append(batch["manual_masks/mask"])
        res = torch.cat(res)
        ds_tensor = res[:, None, :, :, 0] * -1.0 + (not include_boundary)

        if fixed_vertical_offset is not None:
            ds2_tensor = torch.cat(
                [torch.zeros(20, 1, fixed_vertical_offset, 565), ds_tensor], axis=2
            )
            ds1_tensor = torch.cat(
                [ds_tensor, torch.zeros(20, 1, fixed_vertical_offset, 565)], axis=2
            )
        else:
            ds2_tensor = ds_tensor
            ds1_tensor = ds_tensor

        warped_tensors = []
        print("warping images to generate dataset")
        for _ in tqdm.tqdm(range(warps_per_pair)):
            ds_2_list = []
            for el in ds2_tensor:
                case = el[0]
                # TODO implement random warping on gpu
                case_warped = np.array(case)
                if extra_deformation:
                    case_warped = elasticdeform.deform_random_grid(
                        case_warped, sigma=60, points=3
                    )
                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=25, points=3
                )

                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=12, points=6
                )
                ds_2_list.append(torch.tensor(case_warped)[None, None, :, :])
                ds_2_tensor = torch.cat(ds_2_list)
            warped_tensors.append(ds_2_tensor)

        augmented_ds2_tensor = torch.cat(warped_tensors)
        augmented_ds1_tensor = torch.cat([ds1_tensor for _ in range(warps_per_pair)])

        torch.save((augmented_ds1_tensor, augmented_ds2_tensor), ds_name)

    batch_size = 10
    import torchvision.transforms.functional as Fv

    if blur_sigma is None:
        ds1 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds1_tensor, downsample_factor)
        )
    else:
        ds1 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds1_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )
    d1 = torch.utils.data.DataLoader(
        ds1,
        batch_size=batch_size,
        shuffle=False,
    )
    if blur_sigma is None:
        ds2 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds2_tensor, downsample_factor)
        )
    else:
        ds2 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds2_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )

    d2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=batch_size,
        shuffle=False,
    )

    return d1, d2


def get_dataset_sunnyside(split, scale=1):
    import pickle

    with open("/playpen/tgreer/sunnyside.pickle", "rb") as f:
        array = pickle.load(f)
    if split == "train":
        array = array[1000:]
    elif split == "test":
        array = array[:1000]
    else:
        raise ArgumentError()

    array = array[:, :, :, 0]
    array = np.expand_dims(array, 1)
    array = array * scale
    array1 = array[::2]
    array2 = array[1::2]
    array12 = np.concatenate([array2, array1])
    array21 = np.concatenate([array1, array2])
    ds = torch.utils.data.TensorDataset(torch.Tensor(array21), torch.Tensor(array12))
    ds = torch.utils.data.DataLoader(
        ds,
        batch_size=128,
        shuffle=True,
    )
    return ds


def get_cartilage_dataset():
    cartilage = torch.load("/playpen/tgreer/cartilage_uint8s.trch")
    return cartilage


def get_knees_dataset():
    brains = torch.load("/playpen/tgreer/kneestorch")
    #    with open("/playpen/tgreer/cartilage_eval_oriented", "rb") as f:
    #        cartilage = pickle.load(f)

    medbrains = []
    for b in brains:
        medbrains.append(F.avg_pool3d(b, 4))

    return brains, medbrains


def make_batch(data, BATCH_SIZE, SCALE):
    image = torch.cat([random.choice(data) for _ in range(BATCH_SIZE)])
    image = image.reshape(BATCH_SIZE, 1, SCALE * 40, SCALE * 96, SCALE * 96)
    image = image.to(config.device)
    return image
