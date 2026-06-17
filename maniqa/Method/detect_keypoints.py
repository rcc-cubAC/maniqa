import torch
import numpy as np
from typing import Union

from maniqa.Method.predict import loadImageCHW


def imageToTensor(image_chw: np.ndarray, device, dtype: torch.dtype) -> torch.Tensor:
    '''CHW RGB([0,1]) np 图 -> [1, C, H, W] torch 张量（SuperPoint 内部会自动转灰度）。'''
    return torch.from_numpy(image_chw[None]).to(device=device, dtype=dtype)


def countKeypoints(output: dict) -> int:
    '''从 SuperPoint 输出 dict 里数第 0 张图的关键点个数。'''
    keypoints = output["keypoints"]
    first = keypoints[0] if isinstance(keypoints, (list, tuple)) else keypoints
    return int(first.shape[0])


@torch.no_grad()
def countKeypointsFile(detector, image_file_path: str) -> Union[int, None]:
    '''完整链路：图片路径 -> 关键点数量；失败返回 None。

    对应 dino_detect 的 detectFile(detector, path) 写法。
    '''
    image_chw = loadImageCHW(image_file_path)
    if image_chw is None:
        return None
    return detector.count(image_chw)
