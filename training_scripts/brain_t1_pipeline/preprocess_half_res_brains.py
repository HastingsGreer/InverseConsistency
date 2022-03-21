import os
import argparse
import footsteps
footsteps.initialize(output_root="/playpen-ssd/tgreer/ICON_brain_preprocessed_data/")


def process(iA, isSeg=False):
    iA = iA[None, None, :, :, :]
    iA = torch.nn.functional.avg_pool3d(iA, 2)
    return iA

for split in ["train"]:
    with open(f"splits/{split}.txt") as f:
        image_paths = f.readlines()

    import torch

    import itk
    import tqdm
    import numpy as np
    import glob

    ds = []
    for name in tqdm.tqdm(list(iter(image_paths))[:]):
        name = name[:-1] # remove newline

        image = torch.tensor(np.asarray(itk.imread(name)))

        ds.append(process(image))

    torch.save(ds, f"{footsteps.output_dir}/brain_{split}_2xdown_scaled")

