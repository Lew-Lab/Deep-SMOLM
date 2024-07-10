# Deep-SMOLM
Deep-SMOLM is an estimator for estimating 3D orientation and 2D position of dipole-like emitters for single-molecule orientation localization microscopy.

## Usage

```
Please read the 'readme.docx'
All the user defined parameters are in config_orientations.json
```

## Deep-SMOLM training/estimating
```
Training: Deep_SMOLM_main.py
Estimating: Deep_SMOLM_est.py
Experimental data estimating: Deep_SMOLM_est_experiment.py
```

## Forward model
```
Generating simulated training images: forward_model_pixOL\generate_images_pmask_perfect.m
```


## Preparing experimental data for Deep-SMOLM
```
Register two channels' images: create_tform_dense_emitters.m
Crop images based on registration map: tfrom_crop_image_for_deep_SMOLM\crop_image_baseon_local_tform.m
```


## References
T. Wu, P. Lu, M. A. Rahman, X. Li, and M. D. Lew, “Deep-SMOLM: deep learning resolves the 3D orientations and 2D positions of overlapping single molecules with optimal nanoscale resolution,” Opt. Express 30, 36761 (2022). [Article](https://doi.org/10.1364/OE.470146)
