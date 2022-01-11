import itk
import unittest

import icon_registration.test_utils
import icon_registration.pretrained_models
import icon_registration.itk_wrapper
class TestItkRegistration(unittest.TestCase):
    def test_itk_registration(self):

        icon_registration.test_utils.download_test_data()

        model = icon_registration.pretrained_models.OAI_knees_registration_model(
            pretrained=True
        )

        image_A = itk.imread(str(
            icon_registration.test_utils.TEST_DATA_DIR / 
            "knees_diverse_sizes" / 
            "9126260_20060921_SAG_3D_DESS_LEFT_11309302_image.nii.gz")
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
        
#        assert(isinstance(phi_AB, itk.DisplacementFieldTransform))
#        assert(isinstance(phi_BA, itk.DisplacementFieldTransform))

        



