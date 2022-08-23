# Future Localization

This README contains information regarding the future localization task as part of the forecasting benchmark of the Ego4D dataset.


## Data Download

Download necessary data by installing the [Ego4D CLI tool](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) and running: 
`python -m ego4d.cli.cli --output_directory="MY_DATA_LOCATION" --datasets fut_loc` 
Data necessary for the future localization task exists as a relatively small (~12GB) subset of the Ego4D dataset. 

## Data Format

The JSON files are organized as a list of image-trajectory pairs. The index of the JSON is the relative image location, while the "traj" field is a string describing the trajectory. This string is organized as follows:

    [up.x up.y up.z] TrajLength(n) [t_0 C_0.x C_0.y C_0.z b b] [t_1 C_1.x C_1.y C_1.z b b] ... [t_n ...]

 - **up**: A vector whose direction is the normal of the ground plane relative to the camera and
   magnitude is the camera height. 
 - **TrajLength**: The total number of
   samples in this trajectory 
- **C_i**: A point representing the 3D camera
   location at frame t_i (10 FPS)

## Building Model

To build the KNN model, run:

    python gen_model_fut_loc.py --json DATA_PATH/train.json --images DATA_PATH/features/train --output OUT_PATH

- -\-json path/to/data.json
- -\-images path/to/images_or_features
- -\-output path/to/output
- -\-processed (optional; use pre-processed features as .npy files)
- -\-help (prints help dialogue)



## Evaluation
To evaluate between two json files with equivalent indices (reference and new results)

    python eval_fut_loc.py --ref DATA_PATH/ref.json --new DATA_PATH/new.json --output OUT_PATH

- -\-ref path/to/ref.json
- -\-new path/to/new.json
- -\-output path/to/output (optional)
- -\-help (prints help dialogue)
