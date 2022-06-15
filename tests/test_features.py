import torch

from brever.features import FeatureExtractor
from brever.filters import MelFB


def test_feature_extractor():
    mag = torch.randn(2, 257, 30)
    phase = torch.randn(2, 257, 30)
    features = [
        'ild',
        'ipd',
        'ic',
        'fbe',
        'logfbe',
        'cubicfbe',
        'pdf',
        'logpdf',
        'cubicpdf',
        'mfcc',
        'cubicmfcc',
        'pdfcc',
    ]
    mel_fb = MelFB()
    extractor = FeatureExtractor(features=features, mel_fb=mel_fb)
    extractor((mag, phase))
