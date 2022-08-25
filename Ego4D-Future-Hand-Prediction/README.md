# Ego4D Hand Movement Prediction Baseline

## Installation:
Our method requires the same dependencies as SlowFast. We refer to the official implementation fo [SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation details.

## Data Preparation:

**Input**: 60 frames before PRE-1.5s frame (p3). See the definition in paper I-1.1  

**Output**: 5 frames with hand positions on {p3,p2,p1,p,c}; left/right hand position format: x_l, y_l, x_r, y_r

**Note on Ground Truth**: In the dataloader, we choose pad zeros when hand ground truth is not available.

- The resulting data should be organized as following:
```
PATH_TO_DATA_DIR
│ 
└─── annotations
│   │   fho_hands_train.json
│   │   fho_hands_val.json
│   │   fho_hands_test_unannotated.json
│   │   fho_hands_trainval.json (contains all samples from training and validation set)
|
└─── cropped_videos_ant
    │   ClipId1_FrameId1.mp4
    │   ClipId2_FrameId2.mp4
    │   ...  
```
- Make sure to follow [Submission Guidelines](https://eval.ai/web/challenges/challenge-page/1630/submission) for the format of clip name in folder **cropped_videos_ant**.  

## Training: 
```shell
python tools/run_net.py --cfg /path/to/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50.yaml OUTPUT_DIR /path/to/ego4d-hand_ant/output/
```

## Submission and Evaluation:
- Generate inference results on test set (defaulted as output.pkl) for evaluation  
```shell
python tools/run_net.py --cfg /path/to/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50.yaml TRAIN.ENABLE False
```
- Generate submission file for [evalai](https://eval.ai/web/challenges/challenge-page/1630/overview) platform 
```shell
python tools/generate_submission.py /path/to/output.pkl 30
```

- Evaluation function
```shell
# 'test.json' is not provided, just for demonstration 
python tools/eval.py /path/to/output.pkl 30
```


## Important directories and explanation: 
| Directory | Location | Description |
| --------- | -------- | -------- |
| cropped_videos_ant | ./slowfast/datasets/ego4dhand.py | Put your rescaled video clips in this folder |
| PATH_TO_DATA_DIR: ../data-path/ | ./configs/Ego4D/I3D_8x8_R50.yaml | Put your cropped_videos_ant folder and annotation folders under this path |
| OUTPUT_DIR: ../checkpoints/ | ./configs/Ego4D/I3D_8x8_R50.yaml  ./tools/test_net.py | Define store location of checkpoints and output file |
| SAVE_RESULTS_PATH: output.pkl | ./configs/Ego4D/I3D_8x8_R50.yaml  ./tools/test_net.py | Define output file name |
