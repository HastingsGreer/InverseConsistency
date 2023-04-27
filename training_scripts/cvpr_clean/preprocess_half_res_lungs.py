import os
import argparse
import footsteps
footsteps.initialize(output_root="/playpen-ssd/tgreer/ICON_lung_preprocessed_data/")


def process(iA, isSeg=False):
    iA = iA[None, None, :, :, :]
    #SI flip
    iA = torch.flip(iA, dims=(2,))
    if isSeg:
        iA = iA.float()
        iA = torch.nn.functional.max_pool3d(iA, 2)
        iA[iA>0] = 1
    else:

        iA = torch.clip(iA, -1000, 1000)
        iA = iA / 1000


        iA = torch.nn.functional.avg_pool3d(iA, 2)
    return iA

for split in ["train", "test"]:
    with open(f"splits/{split}.txt") as f:
        pair_paths = f.readlines()
    root = "/playpen-raid2/Data/Lung_Registration_transposed/"

    import torch

    import itk
    import tqdm
    import numpy as np
    import glob

    ds = []
    for name in tqdm.tqdm(list(iter(pair_paths))[:]):
        name = name[:-1] # remove newline

        image_insp = torch.tensor(np.asarray(itk.imread(glob.glob(root + name + "/" + name + "_INSP_STD*_COPD_img.nii.gz")[0])))
        image_exp= torch.tensor(np.asarray(itk.imread(glob.glob(root + name + "/" + name + "_EXP_STD*_COPD_img.nii.gz")[0])))

        ds.append((process(image_insp), process(image_exp)))

    torch.save(ds, f"{footsteps.output_dir}/lungs_{split}_2xdown_scaled")

    if split in ['train', 'test', 'eval']:
        ds_seg = []
        for name in tqdm.tqdm(list(iter(pair_paths))[:]):
            name = name[:-1] # remove newline
            seg_insp = torch.tensor(np.asarray(itk.imread(glob.glob(root + name + "/" + name + "_INSP_STD*_COPD_label.nii.gz")[0])))
            seg_exp= torch.tensor(np.asarray(itk.imread(glob.glob(root + name + "/" + name + "_EXP_STD*_COPD_label.nii.gz")[0])))
            ds_seg.append((process(seg_insp, True), process(seg_exp, True)))

        torch.save(ds_seg, f"{footsteps.output_dir}/lungs_seg_{split}_2xdown_scaled")
