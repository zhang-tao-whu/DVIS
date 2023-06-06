# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_minvis_config(cfg):
    cfg.INPUT.SAMPLING_FRAME_RATIO = 1.0
    cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE = False


def add_dvis_config(cfg):
    cfg.INPUT.REVERSE_AGU = False
    cfg.MODEL.TRACKER = CN()
    cfg.MODEL.TRACKER.DECODER_LAYERS = 6
    cfg.MODEL.REFINER = CN()
    cfg.MODEL.REFINER.DECODER_LAYERS = 6

    cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE = 3
    cfg.MODEL.MASK_FORMER.TEST.TASK = 'vis'

    cfg.MODEL.MASK_FORMER.TEST.MAX_NUM = 20

    cfg.DATASETS.DATASET_RATIO = [1.0, ]
    # Whether category ID mapping is needed
    cfg.DATASETS.DATASET_NEED_MAP = [False, ]
    # dataset type, selected from ['video_instance', 'video_panoptic', 'video_semantic',
    #                              'image_instance', 'image_panoptic', 'image_semantic']
    cfg.DATASETS.DATASET_TYPE = ['video_instance', ]
    cfg.DATASETS.DATASET_TYPE_TEST = ['video_instance', ]

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    cfg.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4


