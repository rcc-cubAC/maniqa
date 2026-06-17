import os
import argparse

from maniqa.Module.predictor import Predictor
from maniqa.Module.frame_filter import FrameQualityFilter
from maniqa.Method.predict import setupSeed

# 与 dino-detect 统一的权重约定位置：~/chLi/Model/<MODEL>/<weight>
DEFAULT_CKPT = os.path.expanduser('~/chLi/Model/MANIQA/ckpt_koniq10k.pt')


def main():
    parser = argparse.ArgumentParser(
        description='抽帧 + MANIQA 打分，保留质量最高的前若干比例帧。')
    parser.add_argument('video', help='输入视频文件')
    parser.add_argument('output_dir', help='保留帧 + scores.txt 的输出目录')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help=f'MANIQA checkpoint (.pt)，默认 {DEFAULT_CKPT}')
    parser.add_argument('--fps', type=float, default=10.0, help='抽帧帧率 (default 10)')
    parser.add_argument('--keep', type=float, default=0.7, help='按质量保留的比例 (default 0.7)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-crops', type=int, default=20, help='每帧随机裁块数 (default 20)')
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    setupSeed(args.seed)

    predictor = Predictor(
        model_file_path=args.ckpt,
        num_crops=args.num_crops,
        device=args.device,
    )
    frame_filter = FrameQualityFilter(predictor)
    frame_filter.filterVideo(
        args.video,
        args.output_dir,
        target_fps=args.fps,
        keep_ratio=args.keep,
    )


if __name__ == '__main__':
    main()
