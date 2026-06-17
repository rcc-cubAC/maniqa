import os

from maniqa.Module.predictor import Predictor
from maniqa.Method.predict import setupSeed


def demo():
    home = os.environ['HOME']

    model_file_path = f'{home}/chLi/Model/MANIQA/ckpt_koniq10k.pt'
    image_file_path = './image/kunkun.png'
    device = 'cuda:0'

    setupSeed(20)  # 复现原版随机裁块序列（kunkun.png 应得 0.3398）

    predictor = Predictor(model_file_path=model_file_path, device=device)
    if not predictor.is_valid:
        print('[ERROR][demo] predictor not valid, check checkpoint path.')
        return False

    score = predictor.predictFile(image_file_path)

    print('[INFO][demo]')
    print(f'\t image: {image_file_path}')
    print(f'\t quality score: {score:.4f}')
    return True
