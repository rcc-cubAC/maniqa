import os

from maniqa.Module.keypoint_detector import KeypointDetector
from maniqa.Method.predict import loadImageCHW


def demo():
    home = os.environ['HOME']

    model_file_path = f'{home}/chLi/Model/SuperPoint/superpoint_v6_from_tf.pth'
    image_file_path = './image/kunkun.png'
    device = 'cuda:0'

    detector = KeypointDetector(model_file_path=model_file_path, device=device)
    if not detector.is_valid:
        print('[ERROR][demo] detector not valid, check checkpoint path.')
        return False

    output = detector.detect(loadImageCHW(image_file_path))

    print('[INFO][demo]')
    print(f'\t image: {image_file_path}')
    print(f'\t num keypoints: {output["keypoints"][0].shape[0]}')
    print(f'\t descriptor dim: {output["descriptors"][0].shape}')
    return True
