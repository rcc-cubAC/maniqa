import os

from maniqa.Module.predictor import Predictor
from maniqa.Module.keypoint_detector import KeypointDetector
from maniqa.Module.frame_filter import FrameQualityFilter
from maniqa.Method.predict import setupSeed


def demo():
    home = os.environ['HOME']

    maniqa_ckpt = f'{home}/chLi/Model/MANIQA/ckpt_koniq10k.pt'
    superpoint_ckpt = f'{home}/chLi/Model/SuperPoint/superpoint_v6_from_tf.pth'
    video_file_path = f'{home}/tmp/test_foriqa.mp4'
    output_dir = f'{home}/tmp/test_foriqa_top70'
    device = 'cuda:0'
    target_fps = 10.0
    keep_ratio = 0.7
    quality_weight = 0.5  # fused = 0.5*画质 + 0.5*特征点数量

    setupSeed(20)

    predictor = Predictor(model_file_path=maniqa_ckpt, device=device)
    keypoint_detector = KeypointDetector(model_file_path=superpoint_ckpt, device=device)
    if not predictor.is_valid or not keypoint_detector.is_valid:
        print('[ERROR][demo] model not valid, check checkpoint paths.')
        return False

    frame_filter = FrameQualityFilter(
        predictor,
        keypoint_detector=keypoint_detector,
        quality_weight=quality_weight,
    )
    kept = frame_filter.filterVideo(
        video_file_path,
        output_dir,
        target_fps=target_fps,
        keep_ratio=keep_ratio,
    )

    print('[INFO][demo]')
    print(f'\t kept {len(kept)} frames -> {output_dir}')
    return True
