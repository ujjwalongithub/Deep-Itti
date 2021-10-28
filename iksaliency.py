import torch
import kornia as K


class IKSaliency(torch.nn.Module):
    def __init__(self):
        super(IKSaliency, self).__init__()
        