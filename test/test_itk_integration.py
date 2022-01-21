import itk
import numpy as np
import unittest
import matplotlib.pyplot as plt
import numpy as np

import icon_registration.test_utils
import icon_registration.pretrained_models
import icon_registration.itk_wrapper
class TestItkRegistration(unittest.TestCase):
    def test_itk_registration(self):


        outdir = "/home/hastings/blog/_assets/ICON_test/"
        #import subprocess
        #subprocess.run("rm -r " + outdir + "*", shell=True)

        icon_registration.test_utils.download_test_data()

        model = icon_registration.pretrained_models.OAI_knees_registration_model(
            pretrained=True
        )

        image_A = itk.imread(str(
            icon_registration.test_utils.TEST_DATA_DIR / 
            "knees_diverse_sizes" / 
            #"9126260_20060921_SAG_3D_DESS_LEFT_11309302_image.nii.gz")
             "9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz")
        )

        image_B = itk.imread(str(
            icon_registration.test_utils.TEST_DATA_DIR / 
            "knees_diverse_sizes" / 
            "9225063_20090413_SAG_3D_DESS_RIGHT_12784112_image.nii.gz")
        )
        print(image_A.GetLargestPossibleRegion().GetSize())
        print(image_B.GetLargestPossibleRegion().GetSize())
        print(image_A.GetSpacing())
        print(image_B.GetSpacing())

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(model, image_A, image_B)


        assert(isinstance(phi_AB, itk.CompositeTransform))
        interpolator = itk.LinearInterpolateImageFunction.New(image_A)

        print("pre segfault")
        #warped_image_A = itk.resample_image_filter(image_A, 
        #    transform=phi_AB, 
        #    interpolator=interpolator,
        #    size=[192, 192, 80],
        #)
        warped_image_A = itk.resample_image_filter(image_A, 
            transform=phi_AB, 
            interpolator=interpolator,
            size=itk.size(image_B),
            output_spacing=itk.spacing(image_B),
            output_direction=image_B.GetDirection(),
            output_origin=image_B.GetOrigin()
        )
        print("post_segfault")

        plt.imshow(np.array(itk.checker_board_image_filter(warped_image_A, image_B))[40])
        plt.colorbar()
        plt.savefig(outdir + "grid.png")
        plt.clf()
        plt.imshow(np.array(warped_image_A)[40])
        plt.savefig(outdir + "warped.png")
        plt.clf()

    def test_diff_spacing_identity(self):
        # Try to create a map that maps an image into another image with different stuff

        print("=============================================================")
        
        image_A = itk.imread(str(
            icon_registration.test_utils.TEST_DATA_DIR / 
            "knees_diverse_sizes" / 
            #"9126260_20060921_SAG_3D_DESS_LEFT_11309302_image.nii.gz")
             "9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz")
        )

        image_B = itk.imread(str(
            icon_registration.test_utils.TEST_DATA_DIR / 
            "knees_diverse_sizes" / 
            "9225063_20090413_SAG_3D_DESS_RIGHT_12784112_image.nii.gz")
        )


        middle_shape = [160, 160, 80]
        

    def test_all_images(self):
        import os
        files = os.listdir(icon_registration.test_utils.TEST_DATA_DIR / "knees_diverse_sizes")
        print(files)
        for f in files:
            image = itk.imread(icon_registration.test_utils.TEST_DATA_DIR / "knees_diverse_sizes" / f)
            print("\n[".join(str(image.GetDirection()).split("[")))


    def test_identity_remove_special_transform(self):
        itk_logo_location = str(icon_registration.test_utils.TEST_DATA_DIR / "itkLogo.jpg")
        logo = itk.imread(itk_logo_location)
        print(logo.GetDirection())
        print(logo.GetSpacing())
        
        logo_no_spacing = itk.image_from_array(itk.array_from_image(logo))

