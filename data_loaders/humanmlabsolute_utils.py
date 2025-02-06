import numpy as np

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
HML_ROOT_BINARY = np.array([True]*9 + [False] * (NUM_HML_JOINTS-1)*6)
HML_ROOT_MASK = HML_ROOT_BINARY
HML_ROOT_POSITION_MASK = HML_ROOT_BINARY

HML_ROOT_POSITION_ONLY = np.array([True]*3 + [False] * (NUM_HML_JOINTS)*6)

HML_ROOT_HORIZONTAL_MASK = np.array([True, True, False] + [True]*6 + [False] * (NUM_HML_JOINTS-1)*6)

print('mask length', len(HML_ROOT_BINARY), len(HML_ROOT_POSITION_MASK), len(HML_ROOT_HORIZONTAL_MASK))

NUM_HML_FEATS = 135


def expand_mask(mask, shape):
    """
    expands a mask of shape (num_feat, seq_len) to the requested shape (usually, (batch_size, num_feat, 1, seq_len))
    """
    _, num_feat, _, _ = shape
    return np.ones(shape) * mask.reshape((1, num_feat, 1, -1))

def get_joints_mask(join_names):
    joins_mask = np.array([joint_name in join_names for joint_name in HML_JOINT_NAMES])
    mask = np.concatenate(([False]*(3+6),
                                joins_mask[1:].repeat(6),
    ))
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

    if 'root_position_no_mask' in mask_names or 'no_mask' in mask_names:
        mask = np.zeros(shape)
        return mask

    import pdb;pdb.set_trace()
    mask = np.zeros(shape)
    
    if 'in_between' in mask_names:
        mask = np.maximum(mask, get_in_between_mask(shape, **kwargs))
    
    if 'root' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_MASK, shape))
    
    if 'root_horizontal' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_HORIZONTAL_MASK, shape))
    
    if 'root_position_only' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_ROOT_POSITION_ONLY, shape))

    if 'prefix' in mask_names:
        mask = np.maximum(mask, get_prefix_mask(shape, **kwargs))

    if 'upper_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_UPPER_BODY_MASK, shape))
    
    if 'lower_body' in mask_names:
        mask = np.maximum(mask, expand_mask(HML_LOWER_BODY_MASK, shape))


    mask = np.maximum(mask, get_batch_joint_mask(shape, mask_names))

    # import pdb;pdb.set_trace()

    ## batch

    # only given the goal position. set controlled signal as 1.
    if 'goal_only' in mask_names:
        mask[..., :-5] = 0

    # given start poses.
    if 'start_pose' in mask_names:
        mask[..., 0] = 1
    
    # import pdb;pdb.set_trace()

    return mask