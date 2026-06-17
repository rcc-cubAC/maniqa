# MANIQA + SuperPoint 视频筛帧

No-Reference 画质评估（[MANIQA](https://github.com/IIGROUP/MANIQA), CVPR 2022）+ 关键点检测
（[SuperPoint](https://github.com/rpautrat/SuperPoint), PyTorch 版）。按 `Demo / Method / Model / Module`
风格组织：可单图打画质分 / 数特征点，也可对视频抽帧并**同时**按画质与特征点数量保留质量最高的若干帧。

## Version

```bash
MANIQA    : Koniq10k checkpoint
SuperPoint: v6 (from TF), PyTorch weights
```

## Download

```bash
# MANIQA (543MB, 含完整 ViT backbone)
https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k -> ckpt_koniq10k.pt
# SuperPoint (5MB, MIT)
https://github.com/rpautrat/SuperPoint/blob/master/weights/superpoint_v6_from_tf.pth
```

统一放到 dino-detect 约定位置（运行时也可用 `--ckpt` / `--sp-ckpt` 覆盖）：

```bash
~/chLi/Model/MANIQA/ckpt_koniq10k.pt
~/chLi/Model/SuperPoint/superpoint_v6_from_tf.pth
```

## Setup

```bash
conda create -n maniqa python=3.10
conda activate maniqa
./setup.sh
```

## Run

单图画质分（`image/kunkun.png` 应得 `0.3398`）：

```bash
python demo.py
```

视频抽帧 + 融合画质与特征点，保留前 70%：

```bash
python filter_frames.py /path/to/video.mp4 /path/to/out_dir \
  --fps 10 --keep 0.7 --quality-weight 0.5
```

- `--quality-weight w`：`fused = w*画质 + (1-w)*特征点`（各自 min-max 归一化）。`w=1` 纯画质，`w=0` 纯特征点。
- `--min-keypoints N`：硬门槛，先剔除关键点少于 N 的帧。
- `--no-keypoints`：只用画质筛选。

输出目录里是保留下来的帧 PNG，外加 `scores.txt`（每帧 `quality / keypoints / fused / kept`）。

## How filtering works

每帧两路信号：MANIQA 画质分（越高越清晰）与 SuperPoint 关键点数量（越多越利于匹配/重建）。
两者量纲差异大，故各自 **min-max 归一化到 [0,1]** 后按权重线性融合，再按融合分保留 top-ratio——
既避免糊帧，又避免纹理贫乏的帧。权重与硬门槛均可调。

## Structure

```bash
maniqa/
  Demo/      predictor（单图画质）、keypoint_detector（单图特征点）、frame_filter（视频筛帧）
  Method/    predict（裁块/归一化/打分）、detect_keypoints（特征点）、fuse（归一化/融合）、video（抽帧/选top/拷贝）
  Model/     maniqa、swin、superpoint，及 Layer/（ta_block, swin_block, window_attention, vgg_block ...）
  Module/    Predictor（画质打分）、KeypointDetector（特征点）、FrameQualityFilter（视频筛帧）
```

## Acknowledgment

- MANIQA — IIGROUP/MANIQA (Apache-2.0)
- SuperPoint (PyTorch) — rpautrat/SuperPoint, weights MIT (Rémi Pautrat, Paul-Edouard Sarlin)
