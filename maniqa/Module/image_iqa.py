from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Union

from tqdm import tqdm

from maniqa.Module.predictor import Predictor
from maniqa.Module.keypoint_detector import KeypointDetector

if TYPE_CHECKING:
    # camera_control 是自研库（github.com/565353780/camera-control），非 pip 包，
    # 这里仅用于类型标注。运行期对 camera 只依赖其 ``.toImage()`` 接口（鸭子类型），
    # 故未安装 camera-control 也能 import 本模块。
    from camera_control.Module.camera import Camera


class ImageIQA(object):
    '''统一的图像质量评估器：对每个 camera 同时给出 MANIQA 画质分与 SuperPoint 关键点数量。

    组合 ``Predictor``（MANIQA）与 ``KeypointDetector``（SuperPoint）。接口风格对齐
    pixel-align-deform 用到的各库（``Module`` 类持 ``is_offload_cpu`` + ``toGPU/toCPU``，
    ``self.device`` 恒为目标 GPU、仅模型按需在 CPU/GPU 间搬运）。

    ``is_offload_cpu=True``（默认）：两个模型平时常驻 CPU，``query_cameras_quality``
    在推理窗口前把它们钉到 ``device``、整批跑完立刻卸回 CPU，省显存以便与其它模块共存。
    '''

    def __init__(
        self,
        maniqa_model_file_path: Union[str, None] = None,
        superpoint_model_file_path: Union[str, None] = None,
        maniqa_dtype=torch.float16,
        superpoint_dtype=torch.float32,
        device: str = 'cuda:0',
        is_offload_cpu: bool = True,
        allow_tf32: bool = True,
    ) -> None:
        self.device = device
        self.is_offload_cpu = bool(is_offload_cpu)

        # MANIQA 是 compute-bound（ViT/8 + TAB 通道注意力），fp32 下 A800 张量核几乎闲置，
        # 打 batch 也不提速；真正提速靠降精度。默认 MANIQA 跑 **fp16**（约 6x，分数较 fp32
        # 仅 ~1e-3 漂移：kunkun 0.3398->0.3396）。SuperPoint 只占 ~9ms，且 fp16 会让 NMS/
        # 阈值轻微改动关键点计数，故保持 **fp32** 计数稳定。需严格 fp32 画质分时把
        # maniqa_dtype 设回 torch.float32。
        self.allow_tf32 = bool(allow_tf32)
        if self.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.predictor = Predictor(
            model_file_path=maniqa_model_file_path,
            dtype=maniqa_dtype,
            device=device,
            is_offload_cpu=self.is_offload_cpu,
        )
        self.keypoint_detector = KeypointDetector(
            model_file_path=superpoint_model_file_path,
            dtype=superpoint_dtype,
            device=device,
            is_offload_cpu=self.is_offload_cpu,
        )

        self.is_valid = self.predictor.is_valid and self.keypoint_detector.is_valid
        return

    def toGPU(self) -> 'ImageIQA':
        '''把两个模型钉到 self.device（批量推理前调用，避免逐图来回搬运）。'''
        self.predictor.toGPU()
        self.keypoint_detector.toGPU()
        return self

    def toCPU(self) -> 'ImageIQA':
        '''把两个模型卸回 CPU。'''
        self.predictor.toCPU()
        self.keypoint_detector.toCPU()
        return self

    @staticmethod
    def _cameraToImageCHW(camera: Camera) -> np.ndarray:
        '''camera.toImage() -> (H,W,3) RGB[0,1] tensor -> (3,H,W) float32 numpy。'''
        image_hwc = camera.toImage()
        return image_hwc.permute(2, 0, 1).contiguous().detach().cpu().float().numpy()

    @torch.no_grad()
    def query_cameras_quality(self, camera_list: List[Camera], batch_size: int = 8) -> torch.Tensor:
        '''对每个 camera 返回 [maniqa 画质分, superpoint 关键点数量]。

        ``batch_size`` 张相机一组打包前向（MANIQA 把各图的裁块拼成一个大 batch；
        SuperPoint 把同尺寸图 stack 成一个 batch），打满 GPU 提速；结果与逐张完全一致。
        ``batch_size`` 越大越快、越吃显存，可据显存预算调大。

        Returns:
            torch.Tensor，形状 [N, 2]，float32，CPU；列 0 = MANIQA 分，列 1 = 关键点数。
        '''
        n = len(camera_list)
        if n == 0:
            return torch.zeros((0, 2), dtype=torch.float32)

        # offload 模式：整批推理只搬一次（钉到 GPU -> 分批跑 -> 卸回 CPU）。
        if self.is_offload_cpu:
            self.toGPU()
        try:
            maniqa_scores = [0.0] * n
            keypoint_counts = [0] * n
            for start in tqdm(range(0, n, batch_size), desc='ImageIQA'):
                idxs = list(range(start, min(start + batch_size, n)))
                images = [self._cameraToImageCHW(camera_list[i]) for i in idxs]

                # MANIQA：裁块统一 224x224，可直接整批
                for i, score in zip(idxs, self.predictor.predictBatch(images)):
                    maniqa_scores[i] = score

                # SuperPoint：要求同尺寸，按 (C,H,W) 分组后各自整批
                for local_idxs in self._groupBySize(images).values():
                    counts = self.keypoint_detector.countBatch([images[j] for j in local_idxs])
                    for j, count in zip(local_idxs, counts):
                        keypoint_counts[idxs[j]] = count
        finally:
            if self.is_offload_cpu:
                self.toCPU()

        rows = [[maniqa_scores[i], float(keypoint_counts[i])] for i in range(n)]
        return torch.tensor(rows, dtype=torch.float32)

    @staticmethod
    def _groupBySize(images_chw: List[np.ndarray]) -> dict:
        '''把一批图按 shape 分组 -> {shape: [局部下标...]}，供 SuperPoint 同尺寸 stack。'''
        groups: dict = {}
        for j, image in enumerate(images_chw):
            groups.setdefault(image.shape, []).append(j)
        return groups

    @staticmethod
    def _cameraToMaskedInputs(camera: Camera):
        '''取相机的原图 (3,H,W) 与对齐的前景 mask (H,W)。

        用原图（``use_mask=False``）而非抠白底图，避免物体/白底边界产生人工伪影；
        mask 仅用来约束裁块位置 / 过滤关键点。相机无 mask 时 mask 返回 None。
        '''
        image_chw = camera.toImage(use_mask=False).permute(2, 0, 1).contiguous().detach().cpu().float().numpy()
        if getattr(camera, 'mask', None) is None:
            return image_chw, None
        mask_hw = camera.sampleMaskWithSize(camera.width, camera.height).detach().cpu().numpy().astype(bool)
        return image_chw, mask_hw

    @torch.no_grad()
    def query_cameras_masked_quality(
        self,
        camera_list: List[Camera],
        batch_size: int = 8,
        min_mask_ratio: float = 0.5,
    ) -> torch.Tensor:
        '''带 mask 的版本：只评估每个相机的前景（物体）区域。

        与 ``query_cameras_quality`` 同样返回 [N, 2]（MANIQA 画质分 / SuperPoint 关键点数），
        但 MANIQA 的随机裁块只落在 mask 前景内（``min_mask_ratio`` 控制块的前景占比下限），
        SuperPoint 也只计落在 mask 内的关键点。相机无 mask 的退化为整图（与非 mask 版一致）。
        '''
        n = len(camera_list)
        if n == 0:
            return torch.zeros((0, 2), dtype=torch.float32)

        if self.is_offload_cpu:
            self.toGPU()
        try:
            maniqa_scores = [0.0] * n
            keypoint_counts = [0] * n
            for start in tqdm(range(0, n, batch_size), desc='ImageIQA(mask)'):
                idxs = list(range(start, min(start + batch_size, n)))
                images, masks = [], []
                for i in idxs:
                    image_chw, mask_hw = self._cameraToMaskedInputs(camera_list[i])
                    images.append(image_chw)
                    masks.append(mask_hw)

                scores = self.predictor.predictMaskedBatch(images, masks, min_mask_ratio)
                for i, score in zip(idxs, scores):
                    maniqa_scores[i] = score

                for local_idxs in self._groupBySize(images).values():
                    counts = self.keypoint_detector.countInMaskBatch(
                        [images[j] for j in local_idxs],
                        [masks[j] for j in local_idxs],
                    )
                    for j, count in zip(local_idxs, counts):
                        keypoint_counts[idxs[j]] = count
        finally:
            if self.is_offload_cpu:
                self.toCPU()

        rows = [[maniqa_scores[i], float(keypoint_counts[i])] for i in range(n)]
        return torch.tensor(rows, dtype=torch.float32)
