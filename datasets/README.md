# Prepare Datasets for DVIS

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

DVIS has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  ytvis_2019/
  ytvis_2021/
  ovis/
  VIPSeg/
  VSPW_480p/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

The [model zoo](../MODEL_ZOO.md)
contains configs and models that use these builtin datasets.


## Expected dataset structure for [YouTubeVIS 2019](https://competitions.codalab.org/competitions/20128):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [YouTubeVIS 2021](https://competitions.codalab.org/competitions/28988):

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [Occluded VIS](http://songbai.site/ovis/):

```
ovis/
  annotations/
    annotations_{train,valid,test}.json
  {train,valid,test}/
```
## Expected dataset structure for [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset):

After downloading the VIPSeg dataset, it still needs to be processed according to the official script. To save time, you can directly download the processed VIPSeg dataset from [baiduyun](https://pan.baidu.com/s/1SMausnr6pVDJXTGISeFMuw) (password is `dvis`). 
```
VIPSeg/
  VIPSeg_720P/
    images/
    panomasksRGB/
    panoptic_gt_VIPSeg_{train,val,test}.json
```

## Expected dataset structure for [VSPW](https://codalab.lisn.upsaclay.fr/competitions/7869#participate):

```
VSPW_480p/
  data/
  {train,val,test}.txt
```

## Register your own dataset:

- If it is a VIS/VPS/VSS dataset, convert it to YTVIS/VIPSeg/VSPW format. If it is a image instance dataset, convert it to COCO format.
- Register it in `/dvis/data_video/datasets/{builtin,vps,vss}.py`