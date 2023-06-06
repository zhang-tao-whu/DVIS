# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_minvis_config, add_dvis_config

from .video_mask2former_transformer_decoder import\
    VideoMultiScaleMaskedTransformerDecoder_minvis, VideoMultiScaleMaskedTransformerDecoder_dvis
from .meta_architecture import MinVIS, DVIS_online, DVIS_offline

# video
from .data_video import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    YTVISEvaluator,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    VPSEvaluator,
    VSSEvaluator,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
