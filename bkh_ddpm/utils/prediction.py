import numpy as np

def to_uint8(x):
    return (127.5 * np.clip(x, -1, 1) + 1).astype(np.uint8)