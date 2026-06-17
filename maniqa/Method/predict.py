import os
import cv2
import torch
import random
import numpy as np
from typing import Union


def setupSeed(seed: int = 20) -> None:
    '''固定 random / numpy / torch 的随机种子，让随机裁块、打分可复现。'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loadImageCHW(image_file_path: str) -> Union[np.ndarray, None]:
    '''读图 -> float32 CHW RGB in [0, 1]；失败返回 None。'''
    if not os.path.exists(image_file_path):
        print('[ERROR][loadImageCHW]')
        print('\t image file not exist!')
        print('\t image_file_path:', image_file_path)
        return None

    image_bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print('[ERROR][loadImageCHW]')
        print('\t failed to read image!')
        print('\t image_file_path:', image_file_path)
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))  # C, H, W


def resizeMinSide(image_chw: np.ndarray, min_side: int) -> np.ndarray:
    '''当最短边小于 min_side 时等比放大到 min_side；否则原样返回。

    保证随机裁块（crop_size 方块）一定能取到，且对足够大的图是 no-op。
    '''
    c, h, w = image_chw.shape
    if min(h, w) >= min_side:
        return image_chw
    scale = float(min_side) / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    hwc = np.transpose(image_chw, (1, 2, 0))
    hwc = cv2.resize(hwc, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return np.transpose(hwc, (2, 0, 1))


def randomCropPatches(image_chw: np.ndarray, num_crops: int, crop_size: int) -> np.ndarray:
    '''从 CHW 图随机裁 num_crops 个 crop_size 方块 -> [N, C, crop, crop]。

    每块先采样 top 再采样 left，RNG 顺序与原版 MANIQA 一致，便于复现分数。
    '''
    c, h, w = image_chw.shape
    patches = np.empty((num_crops, c, crop_size, crop_size), dtype=np.float32)
    for i in range(num_crops):
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)
        patches[i] = image_chw[:, top:top + crop_size, left:left + crop_size]
    return patches


def normalizePatches(patches: np.ndarray, mean: float, std: float) -> np.ndarray:
    '''逐元素 (x - mean) / std。'''
    return (patches - mean) / std


def preprocessPatches(
    patches: np.ndarray,
    mean: float,
    std: float,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    '''np [N, C, h, w] in [0, 1] -> 归一化后的 torch 张量（指定 device/dtype）。'''
    patches = normalizePatches(patches.astype(np.float32), mean, std)
    return torch.from_numpy(patches).to(device=device, dtype=dtype)


@torch.no_grad()
def scorePatches(model, patches_tensor: torch.Tensor) -> torch.Tensor:
    '''一次前向给所有 patch 打分 -> [N]。'''
    return model(patches_tensor)


@torch.no_grad()
def predictFile(predictor, image_file_path: str) -> Union[float, None]:
    '''完整链路：图片路径 -> 质量分（float）；失败返回 None。

    对应 dino_detect 里的 ``detectFile(detector, path)``。
    '''
    image_chw = loadImageCHW(image_file_path)
    if image_chw is None:
        return None
    return predictor.predict(image_chw)
