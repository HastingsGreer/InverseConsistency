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
        import os

        os.environ["FOOTSTEPS_NAME"] = "test"
        import footsteps

        icon_registration.test_utils.download_test_data()

        model = icon_registration.pretrained_models.OAI_knees_registration_model(
            pretrained=True
        )

        image_A = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "knees_diverse_sizes"
                /
                # "9126260_20060921_SAG_3D_DESS_LEFT_11309302_image.nii.gz")
                "9487462_20081003_SAG_3D_DESS_RIGHT_11495603_image.nii.gz"
            )
        )

        image_B = itk.imread(
            str(
                icon_registration.test_utils.TEST_DATA_DIR
                / "knees_diverse_sizes"
                / "9225063_20090413_SAG_3D_DESS_RIGHT_12784112_image.nii.gz"
            )
        )
        print(image_A.GetLargestPossibleRegion().GetSize())
        print(image_B.GetLargestPossibleRegion().GetSize())
        print(image_A.GetSpacing())
        print(image_B.GetSpacing())

        phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
            model, image_A, image_B
        )

        assert isinstance(phi_AB, itk.CompositeTransform)
        interpolator = itk.LinearInterpolateImageFunction.New(image_A)

        warped_image_A = itk.resample_image_filter(
            image_A,
            transform=phi_AB,
            interpolator=interpolator,
            size=itk.size(image_B),
            output_spacing=itk.spacing(image_B),
            output_direction=image_B.GetDirection(),
            output_origin=image_B.GetOrigin(),
        )

        plt.imshow(
            np.array(itk.checker_board_image_filter(warped_image_A, image_B))[40]
        )
        plt.colorbar()
        plt.savefig(footsteps.output_dir + "grid.png")
        plt.clf()
        plt.imshow(np.array(warped_image_A)[40])
        plt.savefig(footsteps.output_dir + "warped.png")
        plt.clf()

        reference = np.load(icon_registration.test_utils.TEST_DATA_DIR / "warped.npy")

        np.save(
            footsteps.output_dir + "warped.npy",
            itk.array_from_image(warped_image_A)[40],
        )

        self.assertLess(
            np.mean(np.abs(reference - itk.array_from_image(warped_image_A)[40])), 1e-6
        )
