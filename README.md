# MANIQA

No-Reference 图像质量评估（Multi-dimension Attention Network for No-Reference IQA, CVPR 2022）。
按 `Demo / Method / Model / Module` 风格重组，可单图打分，也可对视频抽帧并保留质量最高的若干帧。

## Version

```bash
Koniq10k checkpoint
```

## Download

```bash
https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k
-->
ckpt_koniq10k.pt        # 543MB，含完整 ViT backbone
```

放到 `~/chLi/Model/MANIQA/ckpt_koniq10k.pt`（或自定，运行时用 `--ckpt` 指定）。

## Setup

```bash
conda create -n maniqa python=3.10
conda activate maniqa
./setup.sh
```

## Run

单图质量打分（`image/kunkun.png` 应得 `0.3398`）：

```bash
python demo.py
```

视频抽帧 + 保留质量前 70% 的帧：

```bash
python filter_frames.py /path/to/video.mp4 /path/to/out_dir \
  --ckpt ~/chLi/Model/MANIQA/ckpt_koniq10k.pt --fps 10 --keep 0.7
```

输出目录里是保留下来的帧 PNG，外加 `scores.txt`（每帧分数与是否保留）。

## Structure

```bash
maniqa/
  Demo/      # 入口示例：predictor（单图）、frame_filter（视频筛帧）
  Method/    # 原子函数：predict（裁块/归一化/打分）、video（抽帧/选 top/拷贝）
  Model/     # 网络：maniqa、swin，及 Layer/（ta_block, swin_block, window_attention ...）
  Module/    # 高层封装：Predictor（打分器）、FrameQualityFilter（视频筛帧）
```

## Enjoy it~
