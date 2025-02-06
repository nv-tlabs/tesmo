from typing import Dict, Union, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision.models.feature_extraction import create_feature_extractor

import os
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.base_models import MapGridDecoder,RasterizedMapEncoder

import torchshow as ts
import matplotlib.pyplot as plt
from  PIL import Image

def visualize_feature_grid(pos, feat_grid, save_dir='debug_feature_grid', step=0):

    # visualize feature grid
    # for c in feature_grid.shape[1]:
    #     torch.save(feature_grid[:,c,:,:], os.path.join(save_dir, f'feature_grid_{c}.jpg'))
    
    ts.save(feat_grid[0][np.arange(0, 512, 50), None], os.path.join(save_dir, f'feature_grid.jpg'))
    ts.save(feat_grid[0,0,:,:], os.path.join(save_dir, f'feature_grid_0.jpg'))
    # visualize the pos on the grid
    
    B, T, _ = pos.size()
    x, y = pos[0,:, 0], pos[0, :, 1] # feature grid is indexed by (y, x), [H, 0], [W, 0]
    x = x.reshape((-1)).cpu().detach().numpy() * 512/64
    y = y.reshape((-1)).cpu().detach().numpy() * 512/64

    ori_img = Image.open(os.path.join(save_dir, f'feature_grid_0.jpg')).resize((512, 512))

    plt.figure()
    plt.scatter(x, y, s=1, c='r')
    plt.imshow(ori_img)
    plt.savefig(os.path.join(save_dir, f'feature_grid_pos_ori_{step}.jpg'))
    plt.close('all')

    eps = 1e-3
    x = np.clip(x, 0, 512 - 1 - eps) # clamp to avoid out of bounds
    y = np.clip(y, 0, 512 - 1 - eps)
    

    plt.figure()
    plt.imshow(ori_img)
    plt.scatter(x, y, s=1, c='r')
    plt.savefig(os.path.join(save_dir, f'feature_grid_pos_{step}.jpg'))
    plt.close('all')

def query_feature_grid(pos, feat_grid):
    '''
    Bilinearly interpolates given positions in feature grid.
    - pos : (B x T x 2) float
    - feat_grid : (B x C x H x W)

    Returns:
    - (B x T x C) feature interpolated to each position
    '''
    B, T, _ = pos.size()
    x, y = pos[...,0], pos[...,1] # feature grid is indexed by (y, x), [H, 0], [W, 0]
    x = x.reshape((-1))
    y = y.reshape((-1))

    eps = 1e-3
    x = torch.clamp(x, 0, feat_grid.shape[-1] - 1 - eps) # clamp to avoid out of bounds
    y = torch.clamp(y, 0, feat_grid.shape[-2] - 1 - eps)

    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    feat_grid = torch.permute(feat_grid, (0, 2, 3, 1)) # B x H x W x C
    bdim = torch.arange(feat_grid.size(0))[:,None].expand((B, T)).reshape((-1))

    Ia = feat_grid[bdim, y0, x0]
    Ib = feat_grid[bdim, y1, x0]
    Ic = feat_grid[bdim, y0, x1]
    Id = feat_grid[bdim, y1, x1]

    x0 = x0.to(torch.float)
    x1 = x1.to(torch.float)
    y0 = y0.to(torch.float)
    y1 = y1.to(torch.float)

    norm_const = 1. / ((x1 - x0) * (y1 - y0)) # always dealing with discrete pixels so no numerical issues (should always be 1)
    wa = (x1 - x) * (y1 - y) * norm_const
    wb = (x1 - x) * (y - y0) * norm_const
    wc = (x - x0) * (y1 - y) * norm_const
    wd = (x - x0) * (y - y0) * norm_const

    interp_feats = wa.unsqueeze(1) * Ia + \
                   wb.unsqueeze(1) * Ib + \
                   wc.unsqueeze(1) * Ic + \
                   wd.unsqueeze(1) * Id

    interp_feats = interp_feats.reshape((B, T, -1))

    return interp_feats


class MapEncoder(nn.Module):
    """Encodes map, may output a global feature, feature map, or both."""
    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            global_feature_dim=None,
            grid_feature_dim=None,
    ) -> None:
        super(MapEncoder, self).__init__()
        self.return_global_feat = global_feature_dim is not None
        self.return_grid_feat = grid_feature_dim is not None
        encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim
        )
        self.input_image_shape = input_image_shape
        # build graph for extracting intermediate features
        feat_nodes = {
            'map_model.layer1': 'layer1',
            'map_model.layer2': 'layer2',
            'map_model.layer3': 'layer3',
            'map_model.layer4': 'layer4',
            'map_model.fc' : 'fc',
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)
        if self.return_grid_feat:
            encoder_channels = list(encoder.feature_channels().values())
            input_shape_scale = encoder.feature_scales()["layer4"]
            self.decoder = MapGridDecoder( # downsampled / 4
                input_shape=(encoder_channels[-1], 
                             input_image_shape[1]*input_shape_scale, 
                             input_image_shape[2]*input_shape_scale),
                encoder_channels=encoder_channels[:-1],
                output_channel=grid_feature_dim,
                batchnorm=True,
            )
        self.encoder_feat_scales = list(encoder.feature_scales().values())

    def feat_map_out_dim(self, H, W):
        dim_scale = self.encoder_feat_scales[-4] # decoder has 3 upsampling
        return (H * dim_scale, W * dim_scale )

    def forward(self, map_inputs, encoder_feats=None):
        if encoder_feats is None:   
            encoder_feats = self.encoder_heads(map_inputs)
        fc_out = encoder_feats['fc'] if self.return_global_feat else None
        encoder_feats = [encoder_feats[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
        feat_map_out = None
        if self.return_grid_feat:
            feat_map_out = self.decoder.forward(feat_to_decode=encoder_feats[-1],
                                                encoder_feats=encoder_feats[:-1])
        return fc_out, feat_map_out

if __name__ == "__main__":
    map_encoder = MapEncoder(model_arch='resnet18', 
                             input_image_shape=(1, 64, 64), 
                             global_feature_dim=8, 
                             grid_feature_dim=32)
    input_data = torch.randn((1, 1, 64, 64))
    feat, grid = map_encoder(input_data)