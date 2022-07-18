from ..criterion import get_criterion

import torch.nn as nn


class BreverBaseModel(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = get_criterion(criterion)

    def pre_proc(self, *args, **kwargs):
        raise NotImplementedError

    def segment_to_item_length(self, *args, **kwargs):
        raise NotImplementedError

    def enhance(self, *args, **kwargs):
        raise NotImplementedError
