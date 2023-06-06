## Getting Started with DVIS

This document provides a brief intro of the usage of DVIS.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Training
We provide a script `train_net_video.py`, that is made to train all the configs provided in DVIS.

To train a model with "train_net_video.py", first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the pre-trained weights of MinVIS from [here](MODEL_ZOO.md) or [minvis_model_zoo](https://github.com/NVlabs/MinVIS/blob/main/MODEL_ZOO.md) and put them in the current working directory.
Once these are set up, run:
```
# train the DVIS_Online
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/DVIS_Online_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/minvis_pretrained_weights.pth

# train the DVIS_Offline
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/DVIS_Offline_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/DVIS_Online_weights.pth 

```

#### Training on a new dataset
If you want to train on a new dataset, first need to fine-tune the segmenter according to the MinVIS process. Please download the COCO (Instance/Panoptic/Semantic) pre-trained weights of Mask2Former from [here](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) and put them in the current working directory.
Once these are set up, run:
```
# finetune the segmenter
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/MinVIS_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/Mask2Former_COCO_pretrained_weights.pth

# train the DVIS_Online
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/DVIS_Online_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/MinVIS_weights.pth

# train the DVIS_Offline
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/DVIS_Offline_config_file.yaml \
  --resume MODEL.WEIGHTS /path/to/DVIS_Online_weights.pth 
```

### Evaluation

Prepare the datasets following [datasets/README.md](./datasets/README.md) and download trained weights from [here](MODEL_ZOO.md).
Once these are set up, run:
```
python train_net_video.py \
  --num-gpus 8 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth 
```


### Visualization

1. Pick a trained model and its config file. To start, you can pick from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/ovis/DVIS_Online_R50.yaml`.
2. We provide `demo_long_video.py` to visualize outputs of a trained model. Run it with:
```
python demo_long_video.py \
  --config-file /path/to/config.yaml \
  --input /path/to/images_folder \
  --output /path/to/output_folder \  
  --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth

# if the video if long (> 300 frames), plese set the 'windows_size'
python demo_long_video.py \
  --config-file /path/to/config.yaml \
  --input /path/to/images_folder \
  --output /path/to/output_folder \  
  --windows_size 300 \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth
```
The input is a folder containing video frames saved as images. For example, `ytvis_2019/valid/JPEGImages/00f88c4f0a`.

