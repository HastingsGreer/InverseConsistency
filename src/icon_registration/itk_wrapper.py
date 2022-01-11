import itk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.pretrained_models
import icon_registration.network_wrappers

import icon_registration.visualize

def register_pair(model, image_A, image_B):


    outdir = "/home/hastings/blog/_assets/ICON_test/"
    import subprocess
    subprocess.run("rm -r " + outdir + "*", shell=True)
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
    print("A_max", torch.max(A_resized))

    plt.imshow(A_resized[0, 0, 40].cpu())
    plt.colorbar()
    plt.savefig(outdir + "A_in.png")
    plt.clf()

    plt.imshow(B_resized[0, 0, 40].cpu())
    plt.colorbar()
    plt.savefig(outdir + "B_in.png")
    plt.clf()
    with torch.no_grad():
        x = model(A_resized, B_resized)
    
    phi_AB = model.phi_AB(model.identityMap)[0].cpu()
    disp_AB = phi_AB - model.identityMap[0].cpu()
    disp_AB *= torch.Tensor([[[[80]]], [[[192]]], [[[192]]]])
    icon_registration.visualize.show_as_grid(phi_AB[[1, 2], 40])
    plt.savefig(outdir + "transform2.png")
    plt.clf()

    # Pass 1: try just using the itk utils to register not respecting spacing

    fake_A = itk.image_from_array(A_resized.float().cpu()[0, 0])
    fake_B = itk.image_from_array(B_resized.float().cpu()[0, 0])
    
    print(type(fake_A))
    
    
    tr = itk.DisplacementFieldTransform[(itk.D, 3)].New()       

    itk_disp_field = array_to_vector_image(disp_AB.double().numpy()[[2, 1, 0]])
    tr.SetDisplacementField(itk_disp_field)

    interpolator = itk.LinearInterpolateImageFunction.New(fake_A)

    warped_a = itk.resample_image_filter(fake_A, 
        transform=tr, 
        interpolator=interpolator,
        size=itk.size(fake_A),
        output_spacing=itk.spacing(fake_A)
        )

    warped_a_arr = np.array(warped_a)

    plt.imshow(warped_a_arr[40])
    plt.colorbar()
    plt.savefig(outdir + "warpedA.png")
    plt.clf()
    plt.imshow(warped_a_arr[:, 40])
    plt.colorbar()
    plt.savefig(outdir + "warpedA2.png")
    plt.clf()
    return tr, None

def array_to_vector_image(array):
    # array is a numpy array of doubles of shape 
    # 3, H, W, D

    # returns an itk.Image of vectors
    # returns image with [1, 1, 1] spacing :(
    assert isinstance(array, np.ndarray)

    arrayT = array.transpose([1, 2, 3, 0])

    PixelType = itk.Vector[itk.D, 3]
    ImageType = itk.Image[PixelType, 3]

    vector_image = itk.PyBuffer[ImageType].GetImageViewFromArray(arrayT, array.shape[1:])

    print(vector_image.GetLargestPossibleRegion().GetSize())

    return vector_image
    
    
