import numpy as np

from brever.features import FeatureExtractor


def test_feature_extractor():
    x = np.random.randn(64, 3, 2, 60, 512)
    features = set(FeatureExtractor.feature_map.keys())
    extractor = FeatureExtractor(features=features)
    extractor(x)
