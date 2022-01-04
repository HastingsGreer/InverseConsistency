Support registrations for changing image sizes and on the original OAI image sizes. This might require a wrapper for upsampling.

 - Add test data with different image sizes : done
 - Add tests of `icon_registration.itk_wrapper` which exersize this capability

Create an even easier wrapper interface that can be imported to a) get the map and its inverse from an image pair, b) accepts itkImages (or what you and Matt determined should be the input) and also returns the spatial transform and the warped image in such a format.

 - Create subpackage `icon_registration.itk_wrapper` for this functionality : done
 - It has method `register_pair(image_A, image_B)` that handles this and above (in progress)

Based on this wrapper also create an easy way to evaluate the results on a test set with labels.

Provide and clean up scripts for the training.

Support training with optionally available segmentation masks that can be incorporated in the loss.

Provide some nice wrapper functionality for some good visualization (transformation grid, checkerboard, warped images and segmentations, ...)

Document.
