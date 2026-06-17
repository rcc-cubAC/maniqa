import torch
import numpy as np
from typing import List, Union

from tqdm import tqdm

from camera_control.Module.camera import Camera

from maniqa.Module.predictor import Predictor
from maniqa.Module.keypoint_detector import KeypointDetector


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
        dtype='auto',
        device: str = 'cuda:0',
        is_offload_cpu: bool = True,
    ) -> None:
        self.device = device
        self.is_offload_cpu = bool(is_offload_cpu)

        self.predictor = Predictor(
            model_file_path=maniqa_model_file_path,
            dtype=dtype,
            device=device,
            is_offload_cpu=self.is_offload_cpu,
        )
        self.keypoint_detector = KeypointDetector(
            model_file_path=superpoint_model_file_path,
            dtype=dtype,
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
    def query_cameras_quality(self, camera_list: List[Camera]) -> torch.Tensor:
        '''对每个 camera 返回 [maniqa 画质分, superpoint 关键点数量]。

        Returns:
            torch.Tensor，形状 [N, 2]，float32，CPU；列 0 = MANIQA 分，列 1 = 关键点数。
        '''
        if len(camera_list) == 0:
            return torch.zeros((0, 2), dtype=torch.float32)

        # offload 模式：整批推理只搬一次（钉到 GPU -> 逐图跑 -> 卸回 CPU）。
        if self.is_offload_cpu:
            self.toGPU()
        try:
            rows = []
            for camera in tqdm(camera_list, desc='ImageIQA'):
                image_chw = self._cameraToImageCHW(camera)
                maniqa_score = self.predictor.predict(image_chw)
                n_keypoints = self.keypoint_detector.count(image_chw)
                rows.append([maniqa_score, float(n_keypoints)])
        finally:
            if self.is_offload_cpu:
                self.toCPU()

        return torch.tensor(rows, dtype=torch.float32)
