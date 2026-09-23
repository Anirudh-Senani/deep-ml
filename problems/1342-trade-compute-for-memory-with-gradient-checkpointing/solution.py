import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def run_blocks(blocks, x, use_checkpoint=False):
    # TODO: apply each block in sequence to x
    # TODO: when use_checkpoint is True, run each block through
    #       checkpoint(...) with use_reentrant explicitly set to False
    if use_checkpoint:
        for block in blocks:
            x = checkpoint(block, x, use_reentrant=False)

    else:
        for block in blocks:
            x = block(x)

    return x
