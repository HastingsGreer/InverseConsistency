import os

with open("../oai_paper_pipeline/splits/test/pair_name_list.txt") as f:
    test_pair_names = f.readlines()
with open("../oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()


test_paths = set()

[[test_paths.add(p) for p in [pp.split()[0], pp.split()[1]]] for pp in test_pair_paths]

import torch

import itk
import tqdm
import numpy as np

ds = []
for tt in tqdm.tqdm(test_pair_paths):
    tt = tt.replace("playpen", "playpen-raid")
    tt = tt.split()
    data = []
    for t in tt:
        image = torch.tensor(np.asarray(itk.imread(t)))
        if "RIGHT" in t:
            image = torch.flip(image, [0])
        elif "LEFT" in t:
            pass
        else:
            raise ConfusedError()
        data.append(image)

    iA, iB, cA, cB = data
    iA = iA[None, None, :, :, :]
    iB = iB[None, None, :, :, :]

    cA = cA[None, None, :, :, :]
    cB = cB[None, None, :, :, :]
    cA = cA.byte()
    cB = cB.byte()

    ds.append([iA, iB, cA, cB])

torch.save(ds, "/playpen/tgreer/knees_test_set_fullres")
