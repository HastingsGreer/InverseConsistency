#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os


# In[3]:


with open("/playpen/tgreer/splits/cross/full_resolution/test/pair_name_list.txt") as f:
    test_pair_names = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/train/pair_name_list.txt") as f:
    train_pair_names = f.readlines()
with open("/playpen/tgreer/splits/cross/full_resolution/train/pair_path_list.txt") as f:
    train_pair_paths = f.readlines()


# In[4]:


len(train_pair_paths[0].split())


# In[5]:


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

ds = []
for tt in tqdm.tqdm(test_pair_paths):
    tt = tt.replace("playpen", "playpen-raid")
    tt = tt.split()
    data = []
    for t in tt:
        image = torch.tensor(itk.GetArrayFromImage(itk.imread(t)))
        if "RIGHT" in t:
            image = torch.flip(image, [0])
        elif "LEFT" in t:
            pass
        else:
            raise FuckityError()
        data.append(image)

    iA, iB, cA, cB = data
    iA = iA[None, None, :, :, :]
    iB = iB[None, None, :, :, :]

    iA = torch.nn.functional.avg_pool3d(iA, 2)
    iB = torch.nn.functional.avg_pool3d(iB, 2)

    cA = cA[None, None, :, :, :]
    cB = cB[None, None, :, :, :]
    cA = cA.byte()
    cB = cB.byte()

    ds.append([iA, iB, cA, cB])


# In[ ]:


torch.save(ds, "/playpen/tgreer/knees_test_set_hires")


# In[91]:


# In[ ]:
