
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
