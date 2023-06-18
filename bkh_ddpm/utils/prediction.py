import torch

def to_uint8(x):
    return (127.5 * (torch.clip(x, -1, 1) + 1)).to(torch.uint8)