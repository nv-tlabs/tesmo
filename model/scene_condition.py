import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectEncoder(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        if self.data_rep in 'bps':
            self.merge_object = nn.Sequential(
                        # nn.Linear(in_features=699 * 3 + 1024 * 3, out_features=512),
                        nn.Linear(in_features=self.input_feats, out_features=512),
                        # nn.Linear(in_features=168451*3, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=self.latent_dim),
                    )
        elif self.data_rep in 'sdf':
            self.merge_object = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['sdf', 'bps']:
            x = self.merge_object(x)  # [seqlen, bs, d]
            return x
        else:
            raise ValueError

# do the canonicalization.
def query_feature_grid_3D(vertices, obj_sdf, sdf_centroid, sdf_scale):
    # this is 3D volume version.
    '''
    Bilinearly interpolates given positions in feature grid.
    - pos : (B x T x 3) float (B, frames, joints, 3)
    - obj_sdf : (B x C x D x H x W)
    - sdf volume center: B x 3
    - sdf volume scale: B x 1

    Returns:
    - (B x T x C) feature interpolated to each position
    '''

    # TODO: append with sdf gradient.
    
    # import pdb;pdb.set_trace()
    
    # sdf_centroid = torch.tensor(self.sdf_dict['centroid']).reshape(1, 1, 3).to(device=vertices.device, dtype=torch.float32)
    batch_size = vertices.shape[0]
    feat_dim = obj_sdf.shape[1]

    # import pdb;pdb.set_trace()
    vertices = vertices.reshape(batch_size, -1, 3)
    num_vertices = vertices.shape[1]
    
    vertices = ((vertices - sdf_centroid[:, None].repeat((1, num_vertices, 1))) / sdf_scale[:, None].repeat((1, num_vertices, 1))) # scale to [-1,1]

    sdf_values = F.grid_sample(obj_sdf, vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3).float(), padding_mode='border', align_corners=True)
                    # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                    

    # import pdb;pdb.set_trace()
    '''
    # illustration of grid_sample dimension order, assume first dimension to be innermost
    # import torch
    # import torch.nn.functional as F
    # import numpy as np
    # sz = 5
    # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
    # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
    # out = F.grid_sample(input_arr, indices, align_corners=True)
    # print(input_arr)
    # print(out)
    '''
    return sdf_values.reshape(batch_size, feat_dim, num_vertices)


if __name__ == "__main__":
    pass