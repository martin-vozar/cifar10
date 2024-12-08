import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def generate_masked_tensor(input, mask, fill=0):
    masked_tensor = torch.zeros(input.size()) + fill
    masked_tensor[mask] = input[mask]
    return masked_tensor

def get_random_mask(steps=8, p=0.5):
    mask = torch.zeros(3, steps, steps)
    for it in range(mask.shape[-2]):
        for jt in range(mask.shape[-1]):
            if np.random.rand()>p:
                mask[:,it,jt] = 1
    mask = F.interpolate(mask.clone().unsqueeze(0), size=32, mode='nearest')[0]
    mask = mask > 0
    return mask

def get_grid_mask(steps=3, inverted=False):
    mask = torch.zeros(3, steps, steps)
    for it in range(mask.shape[-2]):
        for jt in range(mask.shape[-1]):
            if (it+jt)%2 == (0+int(inverted)):
                mask[:,it,jt] = 1
    mask = F.interpolate(mask.clone().unsqueeze(0), size=32, mode='nearest')[0]
    mask = mask > 0
    return mask

def get_center_mask(border=200, inverted=False):   
    if not inverted:
        mask = torch.ones(3, 32, 32)
        mask[:, border:-border, border:-border] = torch.zeros(3, 32-2*border, 32-2*border)
    else:
        mask = torch.zeros(3, 32, 32)
        mask[:, border:-border, border:-border] = torch.ones(3, 32-2*border, 32-2*border)   
    mask = mask > 0
    return mask

MASK_CENTER_8 = get_center_mask(border=8)
MASK_EDGES_8 = get_center_mask(border=8, inverted=True)

MASK_CENTER_12 = get_center_mask(border=12)
MASK_EDGES_12 = get_center_mask(border=12, inverted=True)

MASK_GRID_4 = get_grid_mask(steps=4)
MASK_GRID_4I = get_grid_mask(steps=4, inverted=True)

MASK_GRID_8 = get_grid_mask(steps=8)
MASK_GRID_8I = get_grid_mask(steps=8, inverted=True)

def mask_center_8(xi):
    xi = generate_masked_tensor(xi, MASK_CENTER_8)
    return xi

def mask_center_12(xi):
    xi = generate_masked_tensor(xi, MASK_CENTER_12)
    return xi

def mask_edge_8(xi):
    xi = generate_masked_tensor(xi, MASK_EDGES_8)
    return xi

def mask_edge_12(xi):
    xi = generate_masked_tensor(xi, MASK_EDGES_12)
    return xi

def mask_grid_4(xi):
    xi = generate_masked_tensor(xi, MASK_GRID_4)
    return xi

def mask_grid_4i(xi):
    xi = generate_masked_tensor(xi, MASK_GRID_4I)
    return xi

def mask_grid_8(xi):
    xi = generate_masked_tensor(xi, MASK_GRID_8)
    return xi

def mask_grid_8i(xi):
    xi = generate_masked_tensor(xi, MASK_GRID_8I)
    return xi

def mask_rnd_8_05(xi):
    mask = get_random_mask(steps=8, p=0.5)
    xi = generate_masked_tensor(xi, mask)
    return xi

def mask_rnd_8_07(xi):
    mask = get_random_mask(steps=8, p=0.7)
    xi = generate_masked_tensor(xi, mask)
    return xi

def mask_rnd_8_09(xi):
    mask = get_random_mask(steps=8, p=0.9)
    xi = generate_masked_tensor(xi, mask)
    return xi

def mask_rnd_16_09(xi):
    mask = get_random_mask(steps=16, p=0.9)
    xi = generate_masked_tensor(xi, mask)
    return xi

def mask_rnd_32_09(xi):
    mask = get_random_mask(steps=32, p=0.9)
    xi = generate_masked_tensor(xi, mask)
    return xi

def get_augmentations():
    return([
        # mask_center_8,
        # mask_center_12,
        # mask_edge_8,
        # mask_edge_12,
        # mask_grid_4,
        # mask_grid_4i,
        # mask_grid_8,
        # mask_grid_8i,
        # mask_rnd_8_05,
        # mask_rnd_8_07,
        mask_rnd_8_09,
        mask_rnd_16_09,
        mask_rnd_32_09,
    ])