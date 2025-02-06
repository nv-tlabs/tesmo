import numpy as np
import torch

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

HML_USE_LOCAL_JOINT_MASK= np.concatenate(([False]*(1+2+1),
                                [True] * (NUM_HML_JOINTS-1)*3,
                                [True] * (NUM_HML_JOINTS-1)*6,
                                [False]*3,
                                [True] * (NUM_HML_JOINTS-1)*3,
                                [True] * 4)) # 263-7 = 256

HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_ROOT_POSITION_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))

HML_ROOT_HORIZONTAL_MASK = np.concatenate(([True]*(1+2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))
HML_ROOT_HORIZONTAL_MASK_WOROT = np.concatenate(([False] + [True]*(2) + [False],
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(3)),
                                np.zeros_like(HML_ROOT_BINARY[1:].repeat(6)),
                                np.zeros_like(HML_ROOT_BINARY.repeat(3)),
                                [False] * 4))

HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

HML_TRAJ_MASK = np.zeros_like(HML_ROOT_MASK)
HML_TRAJ_MASK[1:3] = True

NUM_HML_FEATS = 263


def expand_mask(mask, shape):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_feat, 1, seq_len))
    """
    _, num_feat, _, _ = shape
    return np.ones(shape) * mask.reshape((1, num_feat, 1, -1))

def get_joints_mask(join_names):
    joins_mask = np.array([joint_name in join_names for joint_name in HML_JOINT_NAMES])
    mask = np.concatenate(([False]*(1+2+1),
                                joins_mask[1:].repeat(3),
                                np.zeros_like(joins_mask[1:].repeat(6)),
                                np.zeros_like(joins_mask.repeat(3)),
                                [False] * 4))
    return mask

def get_batch_joint_mask(shape, joint_names):
    return expand_mask(get_joints_mask(joint_names), shape)

def get_in_between_mask(shape, lengths, prefix_end, suffix_end):
    mask = np.ones(shape)  # True means use gt motion
    for i, length in enumerate(lengths):
        start_idx, end_idx = int(prefix_end * length), int(suffix_end * length)
        mask[i, :, :, start_idx: end_idx] = 0  # do inpainting in those frames
    return mask

def get_prefix_mask(shape, prefix_length=20):
    _, num_feat, _, seq_len = shape
    prefix_mask = np.concatenate((np.ones((num_feat, prefix_length)), np.zeros((num_feat, seq_len - prefix_length))), axis=-1)
    return expand_mask(prefix_mask, shape)

def get_inpainting_mask(mask_name, shape, **kwargs):
    mask_names = mask_name.split(',')

    if 'no_mask' in mask_names:
        print('no mask at all !!!!!')
        mask = np.zeros(shape)
        return mask    

    if shape[1] != 263:
        original_shape = torch.Size((shape[0], shape[1], shape[2], shape[3]))
        shape = torch.Size((shape[0], 263, shape[2], shape[3]))
    else:
        original_shape = shape
  
    mask = np.zeros(shape)
    
    if 'in_between' in mask_names:
        mask = np.maximum(mask, get_in_between_mask(shape, **kwargs))
    
    if 'root' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_MASK, shape))
    
    if 'root_horizontal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))

    if 'root_horizontal_worot' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK_WOROT, shape))

    if 'prefix' in mask_names:
        mask = np.maximum(mask, get_prefix_mask(shape, **kwargs))

    if 'upper_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_UPPER_BODY_MASK, shape))
    
    if 'lower_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_LOWER_BODY_MASK, shape))

    mask = np.maximum(mask, get_batch_joint_mask(shape, mask_names))

    # import pdb;pdb.set_trace()

    ## batch
    if 'root_posRot' in mask_names: # append extra 3+6 dimension during training.
        # append the last three dimension
        root_mask = np.ones((mask.shape[0], 3+6, mask.shape[2], mask.shape[3]))
        mask = np.concatenate((mask, root_mask), 1)
    # import pdb;pdb.set_trace()
    if 'root_posRot_replace' in mask_names or 'root_posRot_replace_norm' in mask_names or 'root_posRot_only_norm' in mask_names: 
        
        if 'end_pose' in mask_names: # ! this is what we are using.
            mask = np.ones((mask.shape[0], original_shape[1], mask.shape[2], mask.shape[3]))
        else:
            # replace the last three dimension
            root_mask = np.ones((mask.shape[0], 3+6, mask.shape[2], mask.shape[3]))
            mask = np.concatenate((root_mask, mask[:, HML_USE_LOCAL_JOINT_MASK, :, :], ), 1)

    if 'root_pos_y_abs_norm' in mask_names: # default: horizontal root postion and rot-y.
        # replace the last three dimension
        root1 = np.ones((mask.shape[0], 2, mask.shape[2], mask.shape[3]))
        root2 = np.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]))
        root3 = np.ones((mask.shape[0], 1, mask.shape[2], mask.shape[3]))
        mask = np.concatenate((root1, root2, root3, mask[:, HML_USE_LOCAL_JOINT_MASK, :, :], ), 1)
    
    # given start poses.
    if 'start_pose' in mask_names:
        mask[..., 0] = 1
    
    
    if original_shape[1] > mask.shape[1] and 'root_posRot_replace' not in mask_names and 'root_posRot_replace_norm' not in mask_names and \
        'root_pos_y_abs_norm' not in mask_names:
            
        print('append zeros for extra dimension')
        append_mask = np.ones((mask.shape[0], original_shape[1] - mask.shape[1], mask.shape[2], mask.shape[3]))
        mask = np.concatenate((mask, append_mask), 1)

    # import pdb;pdb.set_trace()
    return mask