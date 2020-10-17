import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_grid(source_size, target_size):
    k = float(target_size)/float(source_size)
    direct = torch.linspace(0, k, target_size).unsqueeze(
        0).repeat(target_size, 1).unsqueeze(-1)
    full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
    return full.cuda()


def random_crop_grid(x, grid):
    delta = x.size(2)-grid.size(1)
    grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
    # Add random shifts by x
    grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).cuda().random_(
        0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    # Add random shifts by y
    grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).cuda().random_(
        0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    return grid


# We want to crop a 80x80 image randomly for our batch
# Building central crop of 80 pixel size
grid_source = build_grid(batch.size(2), 80)
# Make radom shift for each batch
grid_shifted = random_crop_grid(batch, grid_source)
# Sample using grid sample
sampled_batch = F.grid_sample(batch, grid_shifted)
