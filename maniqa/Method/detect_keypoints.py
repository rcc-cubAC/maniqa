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


def countKeypointsInMask(keypoints: torch.Tensor, mask_hw: Union[np.ndarray, None]) -> int:
    '''统计落在 mask 前景内的关键点数。

    keypoints: [K, 2] 的 (x, y)（x=列, y=行，与 SuperPoint 输出一致）。
    mask_hw: 对齐的 (H, W) bool；为 None 时返回全部关键点数（即不过滤）。
    '''
    k = int(keypoints.shape[0])
    if k == 0:
        return 0
    if mask_hw is None:
        return k
    kp = keypoints.detach().cpu().numpy()
    h, w = mask_hw.shape
    x = np.clip(np.rint(kp[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(kp[:, 1]).astype(np.int64), 0, h - 1)
    return int(np.count_nonzero(mask_hw[y, x]))


@torch.no_grad()
def countKeypointsFile(detector, image_file_path: str) -> Union[int, None]:
    '''完整链路：图片路径 -> 关键点数量；失败返回 None。

    对应 dino_detect 的 detectFile(detector, path) 写法。
    '''
    image_chw = loadImageCHW(image_file_path)
    if image_chw is None:
        return None
    return detector.count(image_chw)
