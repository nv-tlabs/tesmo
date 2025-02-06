from model.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

def save_sample_poses(input_poses, save_dir, post_fix):
    for i in range(input_poses.shape[0]):
        if i % 10 == 0 or i == input_poses.shape[0]-1:
            poses_mesh = Trimesh(vertices=input_poses[i], process=False)
            poses_mesh.export(os.path.join(save_dir, f'poses_{i}_{post_fix}.obj'))

# TODO: set device id.
class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True, input_poses=False, flip_z=False):

        if not input_poses:
            self.npy_path = npy_path
            self.motions = np.load(self.npy_path, allow_pickle=True)
            if self.npy_path.endswith('.npz'):
                self.motions = self.motions['arr_0']
            self.motions = self.motions[None][0]
        
        else:
            self.motions = {
                'motion':  npy_path[None].transpose(0, 2, 3, 1), # [frames, joints_num, 3]
                'lengths': [npy_path.shape[0]],
                'num_samples': 1,
            }

        self.rot2xyz = Rotation2xyz(device='cuda')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        
        # import pdb;pdb.set_trace()
        # other_motion
        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            # import pdb;pdb.set_trace()
            if flip_z:
                print('flip z in joint2mesh!')
                tmp_motion_input = self.motions['motion'][self.absl_idx].transpose(2, 0, 1)
                tmp_motion_input[..., 2] *= -1
                motion_tensor, opt_dict = self.j2s.joint2smpl(tmp_motion_input)  # [nframes, njoints, 3]   

            else:
                motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            # motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3] 
            self.motions['motion'] = motion_tensor.cpu().numpy()

        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
            
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']).cuda(), mask=None, pose_rep='rot6d', translation=True, glob=True, jointstype='vertices',vertstrans=True).cpu()
        
        # ! old one: double add translation. [check the original 3D joints positions]
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        # self.vertices[:, :, 1, :] += self.root_loc[:, :, 1, :]

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)
