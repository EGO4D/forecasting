# Short-Term Object Interaction Anticipation

- [Short-Term Object Interaction Anticipation](#short-term-object-interaction-anticipation)
  - [Data](#data)
    - [Data download](#data-download)
    - [Pre-extracting RGB frames](#pre-extracting-rgb-frames)
      - [Low-resolution RGB frames](#low-resolution-rgb-frames)
      - [High-resolution image frames](#high-resolution-image-frames)
  - [Replicating the results of the baseline model](#replicating-the-results-of-the-baseline-model)
    - [Downloading pre-trained models and pre-extracted object detections](#downloading-pre-trained-models-and-pre-extracted-object-detections)
    - [Producing object detections (optional)](#producing-object-detections-optional)
    - [Testing the slowfast model](#testing-the-slowfast-model)
      - [Validation set](#validation-set)
      - [Test set](#test-set)
    - [Evaluating the results](#evaluating-the-results)
  - [Training the baseline](#training-the-baseline)
    - [Object detector](#object-detector)
      - [Generating COCO-style annotations](#generating-coco-style-annotations)
      - [Training the object detector](#training-the-object-detector)
    - [SlowFast model](#slowfast-model)

This README reports information on how to train and test the baseline model for the Short-Term Object Interaction Anticipation task part of the forecasting benchmark of the Ego4D dataset. The following sections discuss how to download and prepare the data, download the pre-trained models and train and test the different components of the baseline.

## Data
The first step is to download the data using the CLI avaiable at https://github.com/facebookresearch/Ego4d. 

### Data download
Canonical videos and annotations can be downloaded using the following command:

`python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets full_scale annotations --benchmarks FHO`
### Pre-extracting RGB frames
#### Low-resolution RGB frames
To facilitate the training and testing of the baseline model, we will pre-extract low-resolution (height=320 pixels) RGB frames from the videos. This is done by using the script `dump_frames_to_lmdb_files.py` located in the `tools/short_term_anticipation/` directory. The script takes as input the path to the videos, the path to the annotations, and the path to the output directory, and creates a lmdb database  for each video. By default, the script extracts the video frames preceeding each train/val/test annotation with a duration of 32 frames (a larger context can be set via the `--context_frames` argument). The extraction process can be launched with the following command:

`mkdir -p short_term_anticipation/data`

`python tools/short_term_anticipation/dump_frames_to_lmdb_files.py ~/ego4d_data/v1/annotations/ ~/ego4d_data/v1/full_scale/ short_term_anticipation/data/lmdb`

With the default setting, we expect the output lmdb to occupy about 60GB of disk space.
#### High-resolution image frames
To perform object detection, we will need to extract RGB frames corresponding to the annotations from the videos at their original resolution. We can use the following command to extract the RGB frames:

`mkdir short_term_anticipation/data/object_frames/`

`python tools/short_term_anticipation/extract_object_frames.py ~/ego4d_data/v1/annotations/ ~/ego4d_data/v1/full_scale/ short_term_anticipation/data/object_frames/`
## Replicating the results of the baseline model
We provide pre-trained models and scripts to replicate the results of the baseline model. The following sections discuss how to download the pre-trained models and train and test the different components of the baseline.

### Downloading pre-trained models and pre-extracted object detections
The pre-trained models and pre-extracted object detections can be downloaded using the CLI with the following command:

`python -m ego4d.cli.cli --output_directory="~/ego4d_data" --datasets sta_models`

Once this is done, we need to copy the files to the appropriate paths with the following commands:

```
mkdir short_term_anticipation/models
cp ~/ego4d_data/v1/sta_models/object_detections.json short_term_anticipation/data/object_detections.json
cp ~/ego4d_data/v1/sta_models/object_detector.pth short_term_anticipation/models/object_detector.pth
cp ~/ego4d_data/v1/sta_models/slowfast_model.ckpt short_term_anticipation/models/slowfast_model.ckpt
```
### Producing object detections (optional)
Pre-extracted object detections downloaded at the previous step can be used to train/test the slowfast model. **Alternatively**, we can produce object detections on the validation and test set using the object detection model with the following command:

`python tools/short_term_anticipation/produce_object_detections.py short_term_anticipation/models/object_detector.pth ~/ego4d_data/v1/annotations/ short_term_anticipation/data/object_frames/ short_term_anticipation/data/object_detections.json`
### Testing the slowfast model
#### Validation set
The following command will run the baseline on the validation set:

```
mkdir -p short_term_anticipation/results
python scripts/run_sta.py \
    --cfg configs/Ego4dShortTermAnticipation/SLOWFAST_32x1_8x4_R50.yaml \
    TRAIN.ENABLE False TEST.ENABLE True ENABLE_LOGGING False \
    CHECKPOINT_FILE_PATH short_term_anticipation/models/slowfast_model.ckpt \
    RESULTS_JSON short_term_anticipation/results/short_term_anticipation_results_val.json \
    CHECKPOINT_LOAD_MODEL_HEAD True \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    CHECKPOINT_VERSION "" \
    TEST.BATCH_SIZE 1 NUM_GPUS 1 \
    EGO4D_STA.OBJ_DETECTIONS short_term_anticipation/data/object_detections.json \
    EGO4D_STA.ANNOTATION_DIR ~/ego4d_data/v1/annotations/ \
    EGO4D_STA.RGB_LMDB_DIR short_term_anticipation/data/lmdb/ \
    EGO4D_STA.TEST_LISTS "['fho_sta_val.json']"
```

The command will save the results in the `results/short_term_anticipation/baseline_results_val.json` file.

#### Test set
The following command will run the baseline on the test set:

```
mkdir -p short_term_anticipation/results
python scripts/run_sta.py \
    --cfg configs/Ego4dShortTermAnticipation/SLOWFAST_32x1_8x4_R50.yaml \
    TRAIN.ENABLE False TEST.ENABLE True ENABLE_LOGGING False \
    CHECKPOINT_FILE_PATH short_term_anticipation/models/slowfast_model.ckpt \
    RESULTS_JSON short_term_anticipation/results/short_term_anticipation_results_test.json \
    CHECKPOINT_LOAD_MODEL_HEAD True \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    CHECKPOINT_VERSION "" \
    TEST.BATCH_SIZE 1 NUM_GPUS 1 \
    EGO4D_STA.OBJ_DETECTIONS short_term_anticipation/data/object_detections.json \
    EGO4D_STA.ANNOTATION_DIR ~/ego4d_data/v1/annotations/ \
    EGO4D_STA.RGB_LMDB_DIR short_term_anticipation/data/lmdb/ \
    EGO4D_STA.TEST_LISTS "['fho_sta_test_unannotated.json']"
```

The command will save the results in the `results/short_term_anticipation/baseline_results_test.json` file. The results can be evaluated with the following the instructions reported in the [Evaluating the results](#evaluating-the-results) section.


### Evaluating the results
We provide scripts to evaluate the results of the baseline model. The validation results can be evaluated with the following command:

```
python tools/short_term_anticipation/evaluate_short_term_anticipation_results.py short_term_anticipation/results/short_term_anticipation_results_val.json ~/ego4d_data/v1/annotations/fho_sta_val.json
```

## Training the baseline
We provide code and instructions to train the baseline model. The baseline model uses two components:
 
  * A Fast R-CNN model to detect objects in the test video frames;
  * A Slow-Fast model to predict verb labels and estimate time to contact for each detected objects.

In the following sections, we discuss how to train each component of the baseline model.

### Object detector
We use the Detectron2 library to train the Faster RCNN model and adopt a ResNet-101 baseline trained with a "3x" schedule adapted to the size of the Ego4D dataset. 

#### Generating COCO-style annotations
To train the object detector, we will first need to produce the COCO-style annotations from the JSON annotations. We can create the COCO-style annotations for the train and val sets with the following commands:

`mkdir short_term_anticipation/annotations`

`python tools/short_term_anticipation/create_coco_annotations.py ~/ego4d_data/v1/annotations/fho_sta_train.json short_term_anticipation/annotations/train_coco.json`

`python tools/short_term_anticipation/create_coco_annotations.py ~/ego4d_data/v1/annotations/fho_sta_val.json short_term_anticipation/annotations/val_coco.json`

#### Training the object detector
The model can be trained using the following command:

 `python tools/short_term_anticipation/train_object_detector.py short_term_anticipation/annotations/train_coco.json short_term_anticipation/annotations/val_coco.json short_term_anticipation/data/object_frames/ short_term_anticipation/models/object_detector/`

After training the model, we can use produce object detections on the training, validation and test sets following the instructions reported in the [Producing object detections](#producing-object-detections) section.

### SlowFast model
The model uses a SlowFast model pre-trained on KINETICS-400 which can be downloaded with the following commands:

```
mkdir pretrained_models/
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl -O pretrained_models/SLOWFAST_8x8_R50.pkl
```

The following command can be used to train the Slow-Fast model:

```
mkdir -p short_term_anticipation/models/slowfast_model/
python scripts/run_sta.py \
    --cfg configs/Ego4dShortTermAnticipation/SLOWFAST_32x1_8x4_R50.yaml \
    EGO4D_STA.ANNOTATION_DIR ~/ego4d_data/v1/annotations \
    EGO4D_STA.RGB_LMDB_DIR short_term_anticipation/data/lmdb \
    EGO4D_STA.OBJ_DETECTIONS short_term_anticipation/data/object_detections.json 
    OUTPUT_DIR short_term_anticipation/models/slowfast_model/
```

After training the model, we can copy the model weights from `short_term_anticipation/models/slowfast_model/lightning_logs/version_x/checkpoints/best_model_checkpoint.ckpt` to `short_term_anticipation/models/slowfast_model.ckpt` and follow the instructions reported at the [Testing the Slow-Fast model](#testing-the-slowfast-model) section. `version_x` and `best_model_checkpoint.ckpt` identify the current version and the best epoch of the model. For instance, the path could be: `short_term_anticipation/models/slowfast_model/lightning_logs/version_0/checkpoints/epoch=22-step=22585.ckpt`.

Results can then be evaluated following the instructions reported in the [Evaluating the results](#evaluating-the-results) section.
