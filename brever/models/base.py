import torch.nn as nn


class BreverBaseModel(nn.Module):
    def pre_proc(self, *args, **kwargs):
        raise NotImplementedError

    def segment_to_item_length(self, *args, **kwargs):
        raise NotImplementedError

    def enhance(self, *args, **kwargs):
        raise NotImplementedError
