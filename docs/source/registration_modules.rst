.. _How to extend ICON:


Extending ICON
==============

Writing a similarity measure
----------------------------

Similarity measures in :mod:`icon_registration` are python functions that take two pytorch tensors representing batches of images. These pytorch tensors have two channels: the first channel is image intensity.

The second channel is 1 if the intensity at that voxel is interpolated, or zero otherwise. For some types of images, it is useful to disregard image similarity in extrapolated regions of a warped image. For images with a black background such as skull-stripped brains, this is not necessary.

We will implement a simple absolute difference similarity metric

.. code:: python

  def absolute_similarity(image_A, image_B):
  	# since we are not using the interpolation information, we strip it off before computing similarity.
  	image_A, image_B = image_A[:, 0], image_B[:, 0]
  
  	return torch.mean(torch.abs(image_A - image_B))

Writing a RegistrationModule 
-----------------------------

Representing a Transform
^^^^^^^^^^^^^^^^^^^^^^^^
In order to write a registration method, we first have to understand how `icon_registration` represents a transform. In `icon_registration`, a transform is any python function that takes as input a tensor of coordinates, and returns those coordinates transformed. 

This is intended to closely conform to the mathematical notion of a transform: a function :math:`\phi: \mathbb{R}^N \rightarrow \mathbb{R}^N`

This is an extremely flexible definition, since the transform controls how its internal parameters are used to warp coordinates. On the flip side, it means that transforms are responsible for knowing how to interpret their parameters.

The convention in icon is to store the parameters of the transform in the closure of the function; however an object oriented approach can also work. We will cover both options in this tutorial.

Subclassing RegistrationModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we know how a transform is represented, we can write a new registration technique by subclassing :class:`RegistrationModule`: a registration method is simply any such subclass that takes two images as input to its `forward` method and returns a transform.

The registration methods available in :mod:`icon_registration` are defined in :mod:`~icon_registration.network_wrappers` 

They are named like :class:`.FunctionFromMatrix` or :class:`.FunctionFromVectorField` because they wrap an internal pytorch module that produces, for example, a matrix, from two images, and turn that matrix into a transform, ie a function.

So! Let us write a toy, euler integration based SVF registration.

Registration methods in :mod:`icon_registration` are subclasses of :class:`RegistrationModule` with the following api:
The forward method of a registration method take in two tensors representing batches of images, `image_A` and `image_B`, and returns a python function.

This corresponds to a mathematical notion of a registration method as an operator from a pair of images to a function: 

.. math::
  
  \Phi: (( \mathbb{R}^N \rightarrow \mathbb{R}) \times (\mathbb{R}^N \rightarrow \mathbb{R})) \rightarrow (\mathbb{R}^N \rightarrow \mathbb{R}^N)

When we construct our `FunctionFromStationaryVelocityField` object, we will
pass in a pytorch module `net` that is responsible for generating the velocity
field from the input images. This way, it is easy experiment with different architectures.

..
        Object oriented stationary velocity field
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        .. code:: python

          class SVFTransform:
                  def __init__(self, velocity_field, spacing, n_steps=16):
                          self.n_steps = n_steps
                          self.spacing = spacing
                          self.velocity_delta = velocity_field / n_steps
                  def __call__(self, coordinate_tensor):
                          for _ in range(16):
                                  coordinate_tensor = coordinate_tensor + compute_warped_image_multiNC(
                                          self.velocity_delta, coordinate_tensor, self.spacing, 1)
                          return coordinate_tensor
          
          class FunctionFromStationaryVelocityField(icon_registration.RegistrationModule):
                  def __init__(self, net, n_steps=16):
                          super().__init__()
                          self.net = net
                          self.n_steps = n_steps
          
                  def forward(self, x, y):
                          velocity_field = self.net(x, y)
                          return SVFTransform(velocity_field, self.spacing, self.n_steps)


Stationary velocity field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

  class FunctionFromStationaryVelocityField(icon_registration.RegistrationModule):
      def __init__(self, net, n_steps=16):
          super().__init__()
          self.net = net
          self.n_steps = n_steps
  
      def forward(self, x, y):
          velocityfield_delta = self.net(x, y) / self.n_steps
          def transform(coordinate_tensor):
              for _ in range(self.n_steps):
                coordinate_tensor = input_ + compute_warped_image_multiNC( 
                  velocityfield_delta, coordinate_tensor, self.spacing, 1)
              return coordinate_tensor
          return transform

Building a registration network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These components can now be mixed and matched with existing :mod:`icon_registration` components. For example, if we want to perform a two step registration, with coarse affine registration followed by fine registration using our custom stationary velocity field, and we want to use our custom absolute difference similarity measure, we write


.. code:: python
   
   registration_network = icon.GradientICON(
       icon.TwoStepRegistration(
           icon.FunctionFromMatrix(networks.ConvolutionalMatrixNet(dimension=2)),
           FunctionFromStationaryVelocityField(
                networks.tallUNet2()
           )
       ),
       absolute_similarity,
       lmbda=.4
   )
       
