import torch

from brever.features import FeatureExtractor


def test_feature_extractor():
    mag = torch.randn(2, 514, 30)
    phase = torch.randn(2, 514, 30)
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
    extractor = FeatureExtractor(features=features)
    extractor((mag, phase))
