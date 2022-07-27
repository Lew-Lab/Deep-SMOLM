# Deep-SMOLM
Deep-SMOLM is an estimator for estimating 3D orientation and 2D position of dipole-like emitters for single-molecule orientation localization microscopy.

## Usage

```
Please read the 'readme.docx'
```

## Deep-SMOLM training/estimating
```
Training: Deep_SMOLM_main.py
Estimating: Deep_SMOLM_est.py
Experimental data estimating: Deep_SMOLM_est_experiment.py
```

## Forward model
```
Generating simulated training images: forward_model_pixOL\generate_images_opt_large_dataset_more_info.m
```


## Preparing experimental data for Deep-SMOLM
```
Register two channels' images: create_tform_dense_emitters.m
Crop images based on registration map: tfrom_crop_image_for_deep_SMOLM\crop_image_baseon_local_tform2.m
```


## References
Deep-SMOLM: Deep Learning Resolves the 3D Orientations and 2D Positions of Overlapping Single Molecules with Optimal Nanoscale Resolution
