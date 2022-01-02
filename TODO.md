Support registrations for changing image sizes and on the original OAI image sizes. This might require a wrapper for upsampling.

Create an even easier wrapper interface that can be imported to a) get the map and its inverse from an image pair, b) accepts itkImages (or what you and Matt determined should be the input) and also returns the spatial transform and the warped image in such a format.

Based on this wrapper also create an easy way to evaluate the results on a test set with labels.

Provide and clean up scripts for the training.

Support training with optionally available segmentation masks that can be incorporated in the loss.

Provide some nice wrapper functionality for some good visualization (transformation grid, checkerboard, warped images and segmentations, ...)

Document.
