import os

with open("/playpen/tgreer/splits/cross/full_resolution/test/pair_name_list.txt") as f:
    test_pair_names = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/train/pair_name_list.txt") as f:
    train_pair_names = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/train/pair_path_list.txt") as f:
    train_pair_paths = f.readlines()

len(train_pair_paths[0].split())

train_paths = set()

[
    [train_paths.add(p) for p in [pp.split()[0], pp.split()[1]]]
    for pp in train_pair_paths
]
0

len(train_paths)


# In[6]:


test_paths = set()

[[test_paths.add(p) for p in [pp.split()[0], pp.split()[1]]] for pp in test_pair_paths]
0

len(test_paths)


# In[7]:


len(test_pair_names)


# In[8]:


import torch


# In[ ]:


import itk
import tqdm
import numpy as np

ds = []
for tt in tqdm.tqdm(list(iter(train_paths))):
    tt = tt.replace("playpen", "playpen-raid")
    image = torch.tensor(np.asarray(itk.imread(tt)))
    if "RIGHT" in tt:
        image = torch.flip(image, [0])
    elif "LEFT" in tt:
        pass
    else:
        raise AssertionError()

    iA = image[None, None, :, :, :]

    iA = torch.nn.functional.avg_pool3d(iA, 2)

    ds.append(iA)

torch.save(ds, "/playpen/tgreer/knees_big_2xdownsample_train_set")


# In[91]:


# In[ ]:
