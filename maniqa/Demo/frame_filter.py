import os

from maniqa.Module.predictor import Predictor
from maniqa.Module.frame_filter import FrameQualityFilter
from maniqa.Method.predict import setupSeed


def demo():
    home = os.environ['HOME']

    model_file_path = f'{home}/chLi/Model/MANIQA/ckpt_koniq10k.pt'
    video_file_path = f'{home}/tmp/test_foriqa.mp4'
    output_dir = f'{home}/tmp/test_foriqa_top70'
    device = 'cuda:0'
    target_fps = 10.0
    keep_ratio = 0.7

    setupSeed(20)

    predictor = Predictor(model_file_path=model_file_path, device=device)
    if not predictor.is_valid:
        print('[ERROR][demo] predictor not valid, check checkpoint path.')
        return False

    frame_filter = FrameQualityFilter(predictor)
    kept = frame_filter.filterVideo(
        video_file_path,
        output_dir,
        target_fps=target_fps,
        keep_ratio=keep_ratio,
    )

    print('[INFO][demo]')
    print(f'\t kept {len(kept)} frames -> {output_dir}')
    return True
