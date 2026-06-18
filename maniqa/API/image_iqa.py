"""图像质量评估的稳定 API：MANIQA 画质 + SuperPoint 关键点数量。

接口风格对齐 pixel-align-deform 用到的各库（如 flux_mv/API/geo_sampler.py）：

    get_default_model_paths()                         -> 默认权重路径
    build_model(...)                                  -> ImageIQA
    query_cameras_quality(image_iqa, camera_list)     -> torch.Tensor [N, 2]

``build_model`` 默认 ``is_offload_cpu=True``：两个模型平时只加载到 CPU，仅在
``query_cameras_quality`` 推理窗口内整批搬到 GPU、跑完立刻卸回 CPU。
"""
from __future__ import annotations

import os

import torch

from typing import TYPE_CHECKING, List

from maniqa.Module.image_iqa import ImageIQA

if TYPE_CHECKING:
    # 自研库（非 pip），仅类型标注用；见 maniqa/Module/image_iqa.py 说明。
    from camera_control.Module.camera import Camera


home = os.environ['HOME']

# 与 dino-detect 统一的权重约定位置：~/chLi/Model/<MODEL>/<weight>
maniqa_model_file_path = f'{home}/chLi/Model/MANIQA/ckpt_koniq10k.pt'
superpoint_model_file_path = f'{home}/chLi/Model/SuperPoint/superpoint_v6_from_tf.pth'


def get_default_model_paths() -> dict:
    '''返回默认权重路径，便于上层 pipeline 统一聚合各模块路径。'''
    return {
        'maniqa_model_file_path': maniqa_model_file_path,
        'superpoint_model_file_path': superpoint_model_file_path,
    }


def build_model(
    maniqa_model_file_path: str = maniqa_model_file_path,
    superpoint_model_file_path: str = superpoint_model_file_path,
    maniqa_dtype=torch.float16,
    superpoint_dtype=torch.float32,
    device: str = 'cuda:0',
    is_offload_cpu: bool = True,
    allow_tf32: bool = True,
) -> ImageIQA:
    '''构建 :class:`ImageIQA`。

    ``is_offload_cpu`` 默认 ``True``：MANIQA / SuperPoint 平时常驻 CPU，仅推理时
    搬到 ``device`` 并在结束后立刻卸回；``False`` 则加载后整体常驻 ``device``。
    ``maniqa_dtype`` 默认 ``torch.float16``（约 6x，画质分较 fp32 ~1e-3 漂移）；
    SuperPoint 默认 fp32 保持关键点计数稳定。需严格 fp32 画质分时设 ``torch.float32``。
    '''
    return ImageIQA(
        maniqa_model_file_path=maniqa_model_file_path,
        superpoint_model_file_path=superpoint_model_file_path,
        maniqa_dtype=maniqa_dtype,
        superpoint_dtype=superpoint_dtype,
        device=device,
        is_offload_cpu=is_offload_cpu,
        allow_tf32=allow_tf32,
    )


def query_cameras_quality(image_iqa: ImageIQA, camera_list: List[Camera]) -> torch.Tensor:
    '''对 ``camera_list`` 每个 camera.toImage() 给出 [maniqa 画质分, superpoint 关键点数量]。

    Returns:
        torch.Tensor，形状 [N, 2]，float32，CPU。
    '''
    return image_iqa.query_cameras_quality(camera_list)
