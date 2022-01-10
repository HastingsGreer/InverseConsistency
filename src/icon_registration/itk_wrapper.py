import itk
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.pretrained_models
import icon_registration.network_wrappers

def register_pair(model, image_A, image_B):
    assert( isinstance(image_A, itk.Image))
    assert( isinstance(image_B, itk.Image))
    icon_registration.network_wrappers.adjust_batch_size(model, 1)
    model.cuda() 

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)
    A_trch = torch.Tensor(A_npy).cuda()[None, None]
    B_trch = torch.Tensor(B_npy).cuda()[None, None]

    shape = model.identityMap.shape

    print("A shape:", A_trch.shape)
    print("B shape:", B_trch.shape)


    A_resized = F.interpolate(A_trch, size=shape[2:], mode="trilinear", align_corners=False)
    B_resized = F.interpolate(B_trch, size=shape[2:], mode="trilinear", align_corners=False)
    
    print("A shape:", A_resized.shape)
    print("B shape:", B_resized.shape)
    with torch.no_grad():
        x = model(A_resized, B_resized)

    return None, None



