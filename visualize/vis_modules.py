import os
import pickle
import trimesh
import numpy as np

def merg_scene_obj(scene_list):
    return trimesh.util.concatenate(scene_list)

# TODO: add interactive motion
def get_transform_mat(input_dir, sample_idx, flip_z=False):
    # import pdb;pdb.set_trace()
    with open(os.path.join(input_dir, 'input_y.pickle'), 'rb') as f:
        y_dict = pickle.load(f)
    
    # import pdb;pdb.set_trace()
    if 'transform_mat' in y_dict.keys():
        transform_mat = y_dict['transform_mat'][sample_idx].cpu().numpy()
        if flip_z: # for the transform_mat from locomotion.
            transform_mat[1] *= -1
    else:
        transform_mat = y_dict['transform_vector'][sample_idx].cpu().numpy()

    return transform_mat

def get_transform_mat_length(input_dir):
    with open(os.path.join(input_dir, 'input_y.pickle'), 'rb') as f:
        y_dict = pickle.load(f)

    if 'transform_mat' in y_dict.keys():
        length = y_dict['transform_mat'].shape[0]
    else:
        import pdb;pdb.set_trace()
        length = y_dict['transform_vector'].shape[0]

    return length

# TODO: put all motions into world coordinate system.
def canonicalize_motion_to_world(motion, transform_mat):
    # motion: 22, 3, frames
    if transform_mat.size == 4:
        x, z, cos, sin = transform_mat
        R = np.zeros((3, 3))
        R[0, 0] = cos
        R[0, 2] = -sin
        R[2, 0] = sin
        R[2, 2] = cos
        R[1, 1] = 1.
    else:
        R = transform_mat[:3, :3]
        x, z = transform_mat[[0, 2], 3]
    
    # import pdb;pdb.set_trace()
    new_motion = np.matmul(R.T[None], motion.transpose(2, 1, 0)).transpose(0, 2, 1) # frames, 22, 3
    # motion: 22, 3, frames
    # transform_mat: 4
    # return: 22, 3, frames
    # import pdb;pdb.set_trace()
    transl = np.array([x, 0, z])
    new_motion = new_motion + transl[None, None, :]
    return new_motion