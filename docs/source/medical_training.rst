
Training on a medical dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While we can learn to register 2-D images in a few minutes even on cpu, training for registering 3-D volumes is a more serious endeavor, especially at high resolutions. For that reason, we recommend: 

- Preprocessing all your data in a seperate script and storing it as a pytorch tensor to an ssd local to the gpus. Once this is done, 

- Recording all hyperparameters assosciated with each training run so that you can replicate it- this is super important if you are investing hours or days into a training run, and super easy with :mod:`footsteps`

- Generating and saving metrics, visualizations and weight checkpoints throughout training.
 

Preprocessing the Dataset
=========================

.. code-block:: python
   
        import footsteps
        import torch
        import itk
        import tqdm
        import numpy as np
        footsteps.initialize(output_root="/playpen-ssd/tgreer/ICON_brain_preprocessed_data/")

        def process(iA, isSeg=False):
            iA = iA[None, None, :, :, :]
            iA = torch.nn.functional.avg_pool3d(iA, 4)
            return iA

        downsample = 4
        for split in ["train", "test"]:
            with open(f"splits/{split}.txt") as f:
                image_paths = f.readlines()

            ds = []

            for name in tqdm.tqdm(list(iter(image_paths))[:]):
                name = name.split(".nii.gz")[0] + "_restore_brain.nii.gz"

                image = torch.tensor(np.asarray(itk.imread(name)))

                ds.append(process(image))

            torch.save(ds, f"{footsteps.output_dir}/brain_{split}_4xdown_scaled")


This is the script that you most likely need to modify for your own machine and dataset. To run it, we start with a list of filenames for our splits on our buld storage filesystem, load them using itk, downsample them to the resolution we are training at, and then write them as a tensor to our local ssd. This takes close to an hour to run, but means in all subsequent runs we can start training after a few seconds. If your dataset does not fit in RAM (we use a lot of RAM) then this script will need to be more advanced.

Training the Model
==================

.. code-block:: python

        import random

        import footsteps
        import icon_registration as icon
        import icon_registration.networks as networks
        import torch


        BATCH_SIZE = 8
        input_shape = [1, 1, 130, 155, 130]

        GPUS = 4


        def make_network():
            phi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
            psi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

            hires_net = icon.GradientICON(
                icon.TwoStepRegistration(
                    icon.DownsampleRegistration(
                        icon.TwoStepRegistration(phi, psi), dimension=3
                    ),
                    icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
                ),
                icon.LNCC(sigma=5),
                .7,
            )
            hires_net.assign_identity_map(input_shape)
            return hires_net


        def make_batch():
            image = torch.cat([random.choice(brains) for _ in range(GPUS * BATCH_SIZE)])
            image = image.cuda()
            image = image / torch.max(image)
            return image


        if __name__ == "__main__":
            footsteps.initialize()
            brains = torch.load(
                "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
            )
            hires_net = make_network()

            if GPUS == 1:
                net_par = hires_net.cuda()
            else:
                net_par = torch.nn.DataParallel(hires_net).cuda()
            optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

            net_par.train()

            icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net)
