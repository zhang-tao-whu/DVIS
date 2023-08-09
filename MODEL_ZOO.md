# DVIS Model Zoo

## Introduction

This file documents a collection of trained DVIS models.
The "Config" column contains a link to the config file. Running `train_net_video.py --num-gpus $num_gpus` with this config file will train a model with the same setting. ResNet-50 results are trained with 8 2080Ti GPUs and Swin-L results are trained with 8 V100 GPUs.

## Video Instance Segmentation

### Occluded VIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 Online -->
 <tr><td align="center">DVIS_online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">30.4</td>
<td align="center">54.9</td>
<td align="center">29.7</td>
<td align="center"><a href="configs/ovis/DVIS_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: R50 Online 720p -->
 <tr><td align="center">DVIS_online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">30.9</td>
<td align="center">55.2</td>
<td align="center">30.8</td>
<td align="center"><a href="configs/ovis/DVIS_Online_R50_720p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
<!-- ROW: SwinL Online 480p -->
 <tr><td align="center">DVIS_online</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">46.0</td>
<td align="center">70.7</td>
<td align="center">48.6</td>
<td align="center"><a href="configs/ovis/swin/DVIS_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Online 720p -->
 <tr><td align="center">DVIS_online</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">47.6</td>
<td align="center">72.5</td>
<td align="center">50.1</td>
<td align="center"><a href="configs/ovis/swin/DVIS_Online_SwinL_720p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: R50 Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">33.7</td>
<td align="center">59.6</td>
<td align="center">33.7</td>
<td align="center"><a href="configs/ovis/DVIS_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: R50 Offline 720p -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">34.6</td>
<td align="center">60.1</td>
<td align="center">33.3</td>
<td align="center"><a href="configs/ovis/DVIS_Offline_R50_720p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">48.8</td>
<td align="center">74.8</td>
<td align="center">50.7</td>
<td align="center"><a href="configs/ovis/swin/DVIS_Offline_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Offline 720p -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">50.0</td>
<td align="center">75.6</td>
<td align="center">52.9</td>
<td align="center"><a href="configs/ovis/swin/DVIS_Offline_SwinL_720p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
</tbody></table>

### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 Online -->
 <tr><td align="center">DVIS_online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">51.0</td>
<td align="center">72.8</td>
<td align="center">56.8</td>
<td align="center"><a href="configs/youtubevis_2019/DVIS_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Online 480p -->
 <tr><td align="center">DVIS_online</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">63.7</td>
<td align="center">86.5</td>
<td align="center">70.0</td>
<td align="center"><a href="configs/youtubevis_2019/swin/DVIS_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: R50 Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">52.1</td>
<td align="center">76.3</td>
<td align="center">57.5</td>
<td align="center"><a href="configs/youtubevis_2019/DVIS_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">64.8</td>
<td align="center">88.2</td>
<td align="center">72.6</td>
<td align="center"><a href="configs/youtubevis_2019/swin/DVIS_Offline_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
</tbody></table>

### YouTubeVIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 Online -->
 <tr><td align="center">DVIS_online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">46.3</td>
<td align="center">68.2</td>
<td align="center">50.6</td>
<td align="center"><a href="configs/youtubevis_2021/DVIS_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Online 480p -->
 <tr><td align="center">DVIS_online</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">58.6</td>
<td align="center">80.8</td>
<td align="center">64.6</td>
<td align="center"><a href="configs/youtubevis_2021/swin/DVIS_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: R50 Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">360P</td>
<td align="center">47.3</td>
<td align="center">70.6</td>
<td align="center">51.7</td>
<td align="center"><a href="configs/youtubevis_2021/DVIS_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">60.0</td>
<td align="center">82.6</td>
<td align="center">68.4</td>
<td align="center"><a href="configs/youtubevis_2021/swin/DVIS_Offline_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
</tbody></table>

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">VPQ<sub>thing</sub> </th>
<th valign="bottom">VPQ<sub>stuff</sub> </th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <!-- ROW: MinVIS -->
 <tr><td align="center">MinVis</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="configs/VIPSeg/MinVIS_R50_720p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1JJRr06zXBTX5hyDdKEQzow?pwd=dvis">baidupan</a></td>
</tr>
<!-- ROW: MinVIS -->
 <tr><td align="center">MinVis</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="configs/VIPSeg/swin/MinVIS_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1JJRr06zXBTX5hyDdKEQzow?pwd=dvis">baidupan</a></td>
</tr> 
<!-- ROW: SwinL Online -->
 <tr><td align="center">DVIS_online</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">54.7</td>
<td align="center">54.8</td>
<td align="center">54.6</td>
<td align="center"><a href="configs/VIPSeg/swin/DVIS_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
<!-- ROW: SwinL Offline -->
 <tr><td align="center">DVIS_offline</td>
<td align="center">SwinL(IN21k)</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">57.6</td>
<td align="center">59.9</td>
<td align="center">55.5</td>
<td align="center"><a href="configs/VIPSeg/swin/DVIS_Offline_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1kfSbdHTD8ve47gn-n_7xDQ?pwd=dvis">baidupan</a>|<a href="https://pjxnzg3129.feishu.cn/drive/folder/VW6XfCduZlujiFdgHeCcFss4ndh">feishu</a></td>
</tr>
</tbody></table>
