ICON and MONAI
==============

Using MONAI U-Nets in ICON
^^^^^^^^^^^^^^^^^^^^^^^^^^

MONAI and ICON are already almost compatible in this regard. The differences are that

- ICON expects U-Nets to take two images as inputs, while MONAI U-Nets take one image. 
- ICON U-Nets come with the last layer initialized to weights and biases of zero so that training starts at the identity map, for most MONAI U-Nets this needs to be done manually (RegUNet comes with an argument for this)



.. plot::
   :include-source:
   :context:

   import monai
   import icon_registration as icon
   import icon_registration.monai_wrapper
   import torch


   monai_deformable_net = icon.FunctionFromVectorField(
    icon_registration.monai_wrapper.ConcatInputs(
       monai.networks.nets.AttentionUnet(
             2, 2, 2, channels=[16, 32, 64, 128, 256], strides=(2, 2, 2, 2, 2))))
   
   
   
   torch.nn.init.zeros_(monai_deformable_net.net.net.model[2].conv.weight)
   torch.nn.init.zeros_(monai_deformable_net.net.net.model[2].conv.bias)


This can be mixed and matched with other ICON registration modules in a registration pipeline

.. plot::
    :include-source:
    :context:

    import icon_registration.networks

    inner_net = icon.TwoStepRegistration(
        icon.TwoStepRegistration(
            icon.FunctionFromMatrix(
                icon_registration.networks.ConvolutionalMatrixNet(dimension=2)),
            icon.FunctionFromMatrix(
                icon_registration.networks.ConvolutionalMatrixNet(dimension=2)),
        ),
        monai_deformable_net
    )
Using MONAI similarity metrics in ICON
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ICON losses assume that the first channel is intensity, and the second channel in the warped image indicates whether the value is interpolated or extrapolated. MONAI doesn't use this convention, so we strip the second channel.

.. plot::
    :include-source:
    :context:
    :nofigs:

    model = icon.GradientICON(
       inner_net,
       icon_registration.monai_wrapper.FirstChannelInputs(
           monai.losses.GlobalMutualInformationLoss(),
       ),
       .7
    )

    model.assign_identity_map([1, 1, 64, 64])
    model.cuda()
    model.train()

    device="cuda"
    #warp_layer = Warp("bilinear", "border").to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

Using an ICON RegistrationModule with MONAI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let's load a MONAI dataset

.. plot::
    :include-source:
    :context:

       
    from monai.utils import set_determinism, first
    from monai.transforms import (
        EnsureChannelFirstD,
        Compose,
        LoadImageD,
        RandRotateD,
        RandZoomD,
        ScaleIntensityRanged,
    )
    from monai.data import DataLoader, Dataset, CacheDataset
    from monai.config import print_config, USE_COMPILED
    from monai.networks.nets import GlobalNet
    from monai.networks.blocks import Warp
    from monai.apps import MedNISTDataset
    import os
    import tempfile

    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    train_data = MedNISTDataset(root_dir=root_dir, section="training", download=True, transform=None)
    training_datadict = [
        {"fixed_hand": item["image"], "moving_hand": item["image"]}
        for item in train_data.data if item["label"] == 4  # label 4 is for xray hands
    ]
    print("\n first training items: ", training_datadict[:3])
    train_transforms = Compose(
        [
            LoadImageD(keys=["fixed_hand", "moving_hand"]),
            EnsureChannelFirstD(keys=["fixed_hand", "moving_hand"]),
            ScaleIntensityRanged(keys=["fixed_hand", "moving_hand"],
                                 a_min=0., a_max=255., b_min=0.0, b_max=1.0, clip=True,),
            RandRotateD(keys=["moving_hand"], range_x=np.pi/4, prob=1.0, keep_size=True, mode="bicubic"),
            RandZoomD(keys=["moving_hand"], min_zoom=0.9, max_zoom=1.1, prob=1.0, mode="bicubic", align_corners=False),
        ]
    )
    check_ds = Dataset(data=training_datadict, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
    check_data = first(check_loader)
    fixed_image = check_data["fixed_hand"][0][0]
    moving_image = check_data["moving_hand"][0][0]

    print(f"moving_image shape: {moving_image.shape}")
    print(f"fixed_image shape: {fixed_image.shape}")

    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("moving_image")
    plt.imshow(moving_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("fixed_image")
    plt.imshow(fixed_image, cmap="gray")

    plt.show()


.. plot::
    :include-source:
    :context:

    train_ds = CacheDataset(data=training_datadict[:1000], transform=train_transforms,
                        cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    max_epochs = 40
    epoch_loss_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss, step = 0, 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()

            moving = batch_data["moving_hand"].to(device)
            fixed = batch_data["fixed_hand"].to(device)
            loss_obj = model(moving, fixed)
            loss = loss_obj.all_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
            #       f"train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}") 
   plt.plot(epoch_loss_values)

.. plot::
    :context:
    :include-source:

    import torchvision
    def show(tensor):
        plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
        plt.xticks([])
        plt.yticks([])
    image_A = moving
    image_B = fixed
    plt.subplot(2, 2, 1)
    show(image_A)
    plt.subplot(2, 2, 2)
    show(image_B)
    plt.subplot(2, 2, 3)
    show(model.warped_image_A)
    plt.contour(torchvision.utils.make_grid(model.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach())
    plt.contour(torchvision.utils.make_grid(model.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach())
    plt.subplot(2, 2, 4)
    show(model.warped_image_A - image_B)
    plt.tight_layout()


