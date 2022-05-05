from icon_registration.losses import (LNCC, BlurredSSD, GradientICON,
                                      InverseConsistentNet, gaussian_blur,
                                      ssd_only_interpolated)
from icon_registration.registration_module import (DownsampleRegistration,
                                                   FunctionFromMatrix,
                                                   FunctionFromVectorField,
                                                   RegistrationModule,
                                                   TwoStepRegistration)
from icon_registration.train import train1d, train2d
