# Long-Term Action Anticipation

This README reports information on how to train and test the baseline model for the Long-Term Action Anticipation task part of the forecasting benchmark of the Ego4D dataset. The following sections discuss how to download and prepare the data, download the pre-trained models and train and test the different components of the baseline.

## Data and models
Download all necessary data and model checkpoints using the [Ego4D CLI tool](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) and link the necessary files to the project directory.

```
# set download dir for Ego4D
export EGO4D_DIR=/path/to/Ego4D/

# download annotation jsons, clips and models for the FHO tasks
python -m ego4d.cli.cli \
    --output_directory=${EGO4D_DIR} \
    --datasets annotations clips lta_models \
    --benchmarks FHO

# link data to the current project directory
mkdir -p data/long_term_anticipation/annotations/ data/long_term_anticipation/clips_hq/
ln -s ${EGO4D_DIR}/v1/annotations/* data/long_term_anticipation/annotations/
ln -s ${EGO4D_DIR}/v1/clips/* data/long_term_anticipation/clips_hq/

# link model files to current project directory
mkdir -p pretrained_models
ln -s ${EGO4D_DIR}/v1/lta_models/* pretrained_models/

```

The `data/long_term_anticipation/annotations` directory should contain the following files

 ```
fho_lta_train.json
fho_lta_val.json
fho_lta_test_unannotated.json
fho_lta_taxonomy.json
```

Where `fho_lta_train.json`, `fho_lta_val.json` and `fho_lta_test_unannotated.json` contain the training, validation and test annotations, respectively, and `fho_lta_taxonomy.json.json` contains the verb/noun class id to text mapping.

### Downsampling video clips
To allow dataloaders to load clips efficiently, we will downsample video clips to 320p using ffmpeg. The script can be found at `tools/long_term_anticipation/resize_clips.sh` and can be run in parallel as a SLURM array job. Remember to adjust the paths and SLURM parameters before running.

```
sbatch tools/long_term_anticipation/resize_clips.sh
```
This will create and populate `data/long_term_anticipation/clips/` with downsampled clips


## Training
We provide code and instructions to train the baseline model. Any of these steps can be skipped and the pretrained model can be used instead (e.g., to avoid generating a pre-trained recognition model from scratch). 

See each script for options and details including how to select a specific backbone (SlowFast vs. MViT), how to select different model components (backbones, aggregator modules, and heads) and whether to run locally (to debug) or on the cluster. Running the scripts unaltered will produce the SlowFast-Transformer baseline model. Other options are left in comments.

### Train an Ego4D recognition backbone model
```
bash tools/long_term_anticipation/ego4d_recognition.sh checkpoints/recognition/
```
The pretrained checkpoints for the recognition models can be found at:
```
pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
pretrained_models/long_term_anticipation/ego4d_mvit16x4.ckpt
```

### Train an Ego4D long term anticipation model
```
bash tools/long_term_anticipation/ego4d_forecasting.sh checkpoints/forecasting/
```

See script for different model configurations (backbones, aggregator modules, and heads). Tensorboard logs plot training and validation metrics over time. The pretrained checkpoints for the long-term anticipation models can be found at:
```
pretrained_models/long_term_anticipation/lta_slowfast_concat.ckpt
pretrained_models/long_term_anticipation/lta_slowfast_trf.ckpt
pretrained_models/long_term_anticipation/lta_mvit_concat.ckpt
```

## Generate predictions

Model predictions on the test set can be generated using the following script. See script for option and details.
```
# Generate model predictions (outputs.json)
bash tools/long_term_anticipation/evaluate_forecasting.sh output/
```