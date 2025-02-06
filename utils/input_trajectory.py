import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from scipy.ndimage import gaussian_filter1d, gaussian_filter
from PIL import Image
import os
import math

GAUSSIAN_FILTER = True
def plot_2d(array_2d, save_path='debug.png'):
    import numpy as np

    # Plot the 2D array
    # plt.imshow(array_2d, cmap='viridis')
    plt.figure()
    plt.scatter(array_2d[:,0], array_2d[:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.title('2D Point Plot')
    # Show the plot
    plt.show()
    plt.savefig(save_path)

def load_trajectory(input_path):
    with open(input_path, 'rb') as f:
        input_trajectory = pickle.load(f)
        # dict_keys(['s_start', 's_goal', 'path', 'visited'])
    return input_trajectory

def canonical(positions):
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    rot_r = positions[1:] - positions[:-1]
    angle_rad = np.arctan2(rot_r[:,:, 2], rot_r[:,:, 0])

    theta = angle_rad[0, 0]
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot_mat = np.array([[cos_theta, 0, sin_theta],[0, 1, 0], [-sin_theta, 0, cos_theta]])
    new_positions = ((rot_mat[None] @ positions.transpose(1, 2, 0))).transpose(2, 0, 1)
    
    new_positions = new_positions[:, :, [2,1,0]] # z - x transpose.
    # new_positions[..., 0] *= -1
    return new_positions

def get_new_center(width, height, center, theta):
    cx = width/ 2
    cy = height /2
    x = center[0]
    y = center[1]
    new_x = (x - cx) * np.cos(theta) - (y - cy) * np.sin(theta) + cx
    new_y = (x - cx) * np.sin(theta) + (y - cy) * np.cos(theta) + cy
    return new_x, new_y

def canonical_maps(input_m, start_data, rot, out_dir=None):
    pass
    # import pdb;pdb.set_trace()
    all_round = Image.fromarray((np.ones((256*2, 256*2))*255).astype(np.uint8))
    rot_m = input_m.rotate(rot*180/np.pi, Image.NEAREST, center=(start_data[0], start_data[1]), expand = 1)
    new_x, new_y = get_new_center(256, 256, (start_data[0], start_data[1]), rot)
    
    Image.Image.paste(all_round, rot_m, (256-new_x.astype(np.int), 256-new_y.astype(np.int))) # paste with the center.
    if out_dir is not None:
        rot_m.save(f'{out_dir}/rot_m.png')
        all_round.save(f'{out_dir}/all_rot_m.png')
        
    all_round = 1.0 - (np.array(all_round)==255) * 1.0
    return all_round
    
def get_cont6d_params(positions, kind=0): # this is canicalized body motion.
    
    # (seq_len, joints_num, 4)
    rot_r = positions[1:] - positions[:-1] 
    
    
    if kind == 0:
        angle_rad = -np.arctan2(rot_r[:, 0, 0], rot_r[:, 0, 2])  # this is walking along with the trajectory
    elif kind == 1:
        angle_rad = -np.arctan2(rot_r[:, 0, 0], rot_r[:, 0, 2]) + np.pi # this is walking backside.
    else:
        angle_rad = -np.arctan2(rot_r[:, 0, 0], rot_r[:, 0, 2]) + np.pi / 2 # this is walking laterally.
    
    
    angle_rad_3d = np.concatenate([np.zeros_like(angle_rad[:, None]), \
                angle_rad[:, None], np.zeros_like(angle_rad[:, None])], axis=-1)
    
    # Convert the angle from radians to quaternion
    quat_params = euler_to_quaternion(angle_rad_3d[:, None, :], 'xyz')
    #     
    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot, velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    
    r_rot = np.concatenate((r_rot[0:1, ...], r_rot), axis=0)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot

def insert_body_trajectory(input_motions, input_trajectory, body_part='root_horizontal', dataset=None, fps=2):

    import pdb;pdb.set_trace()
    # xz is the ground plane, y is the height.

    if body_part == 'root_horizontal':
        index = [0,1,2] # relative orientation velocity, root linear velocity;
        # ! we need to define the velocity of the root horizontal 
        input_path_list = np.array(input_trajectory['path'][0:-1:fps])
        input_path_list = input_path_list / 256 * 6.2
        input_path_list = input_path_list[::-1]

        if GAUSSIAN_FILTER and False:
            input_path_list = gaussian_filter(input_path_list, sigma=3)

        plot_2d(input_path_list)

        # canicalize the input trajectory
        input_list = np.concatenate((input_path_list[:, 0:1], np.zeros((input_path_list.shape[0],1)), input_path_list[:, 1:2]), -1)[:, None, :]
        input_list_can = canonical(input_list)
        plot_2d(input_list[:,0,[0,2]], 'debug_canonical.png')
        # import pdb;pdb.set_trace()
        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(input_list_can)
        
        '''Root rotation and linear velocity'''
        # (seq_len-1, 1) rotation velocity along y-axis
        # (seq_len-1, 2) linear velovity on xz plane

        # import pdb;pdb.set_trace()
        r_velocity = np.arcsin(r_velocity[:, 2:3])

        if GAUSSIAN_FILTER: 
            r_velocity = gaussian_filter1d(r_velocity, sigma=3)

        l_velocity = velocity[:, [0, 2]]

        data = np.concatenate([r_velocity, l_velocity], axis=-1)

        # import pdb;pdb.set_trace()
        # calculate the delta
        if dataset is not None:
            data = (data - dataset.mean[:3]) / dataset.std[:3]
        
        if data.shape[0] <= input_motions.shape[-1]:
            input_motions[0, index, 0, :data.shape[0]] = torch.from_numpy(data[:, :]).to(input_motions.device).T.float()
        else:
            input_motions[0, index, 0, :data.shape[0]] = torch.from_numpy(data[:input_motions.shape[-1], :]).to(input_motions.device).T.float()

        return input_motions

def insert_start_end_trajectory(input_motions, input_lengths, start_data, end_data, body_part='abs_pos_rotcossin', \
    dataset=None, final_frame=None, model_kwargs=None):
    # import pdb;pdb.set_trace()
    # xz is the ground plane, y is the height.

    if body_part == 'abs_pos_rotcossin':
        index = [0,1] # relative orientation velocity, root linear velocity;
        # ! we need to define the velocity of the root horizontal 
        start_data[index] = start_data[index] / 256 * 6.4
        end_data[index] = end_data[index] / 256 * 6.4
            
        # canocalize the input trajectory
        cos_theta = start_data[2]
        sin_theta = start_data[3]    
        rot_mat = np.array([[cos_theta, 0, sin_theta],[0, 1, 0], [-sin_theta, 0, cos_theta]])
        
        end_positions = np.array([end_data[0]-start_data[0], 0, end_data[1]-start_data[1]]).reshape(3, 1)
        new_positions = (rot_mat @ end_positions).T
        cos_theta_end = end_data[2]
        sin_theta_end = end_data[3]    
        rot_mat_end = np.array([[cos_theta_end, 0, sin_theta_end],[0, 1, 0], [-sin_theta_end, 0, cos_theta_end]])
        
        # import pdb;pdb.set_trace()
        new_rot_mat = rot_mat @ rot_mat_end
        
        # import pdb;pdb.set_trace()
        end_data[index] = new_positions[0, [0,2]]
        end_data[2] = new_rot_mat[0, 0]
        end_data[3] = new_rot_mat[0, 2]
        
        start_data[index] *= 0.0
        start_data[2] = np.cos(0.0)
        start_data[3] = np.sin(0.0)
        
        if dataset is not None:
            all_index = [0,1,2,3] # the height is not replaced.
            start_data[all_index] = (start_data[all_index] - dataset.mean[all_index]) / dataset.std[all_index]
            end_data[all_index] = (end_data[all_index] - dataset.mean[all_index]) / dataset.std[all_index]
        
        if final_frame is None:
            final_frame = input_lengths[0]
            
        # import pdb;pdb.set_trace()
        # new_input_motions = input_motions.clone()
        new_input_motions = input_motions
        new_input_motions[0, all_index, 0, 0] = torch.from_numpy(start_data[all_index]).to(input_motions.device).T.float()
        new_input_motions[0, all_index, 0, final_frame-1] = torch.from_numpy(end_data[all_index]).to(input_motions.device).T.float()  
        new_input_motions[0, [4], 0, final_frame-1] = new_input_motions[0, [4], 0, input_lengths[0]-1] # use the last frame height.


        # renew the input information.
        # all left input is zero padding.
        new_input_motions[0, :, 0, final_frame:] *= 0.0        
        model_kwargs['y']['lengths'][0] = final_frame
        model_kwargs['y']['mask'] = torch.ones_like(model_kwargs['y']['mask'])
        model_kwargs['y']['mask'][..., final_frame:] = False
        return new_input_motions
    
    else:
        assert False, 'not implemented yet.'

## absolute motion to relative motion.
def absolute_to_relative(input_motions, motion, input_lengths, dataset=None, abs_model_output=None): 
    # input motion: (B, frames, joints, 3)

    index = [0,1,2]
    for i in range(len(input_lengths)):
        
        import pdb;pdb.set_trace()
        
        # get the relative translation
        positions = motion[i, :, :, :input_lengths[i]].transpose(2, 0, 1)
        
        
        # canocalize the input trajectory
        root_pos_init = positions[0]
        root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
        positions = positions - root_pose_init_xz
        
        ori_data = abs_model_output[i][0][:input_lengths[i]]
        r_rot_matrix = np.zeros(ori_data.shape[:-1] + (3,3)) # this should be quaternion.
        r_rot_matrix[:, 0,  0] = ori_data[..., 2] # cosine
        r_rot_matrix[:, 0,  2] = ori_data[..., 3] # sin
        r_rot_matrix[:, 1,  1] = 1
        r_rot_matrix[:, 2,  0] = -ori_data[..., 3]
        r_rot_matrix[:, 2,  2] = ori_data[..., 2] #cos
        
        # r_rot[ 0] = ori_data[..., 2]
        # r_rot[..., 2] = ori_data[..., 3]
        # from data_loaders.amass.tools_teach import matrix_to_quaternion
        from utils.rotation_conversions import matrix_to_quaternion
        r_rot_torch = matrix_to_quaternion(r_rot_matrix)
        r_rot = r_rot_torch.cpu().numpy()
            
            
        # position is correct.
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        data = np.concatenate([r_velocity, l_velocity], axis=-1)

        
        # import pdb;pdb.set_trace()
        # calculate the delta
        if dataset is not None:
            data = (data - dataset.mean[:3]) / dataset.std[:3]
        
        if data.shape[0] <= input_motions.shape[-1]: # input frame is shorter than the target frame.
            input_motions[i, index, 0, :data.shape[0]] = torch.from_numpy(data[:, :]).to(input_motions.device).T.float()
            input_motions[i, :, 0, data.shape[0]:] *= 0
            
    return input_motions    

def convert_root_global_to_local(input_motions, input_lengths, root_motion, dataset=None): # input original modle input. -> 1.3 * to final motion.
    root_motion = torch.from_numpy(root_motion) # now permuted to: [batch, 1, seq_len, nfeat]
    r_pos, rot_cos, rot_sin = root_motion[..., :2], root_motion[..., 2], root_motion[..., 3]
    # import pdb;pdb.set_trace()
    r_pos = torch.stack([r_pos[..., 0], torch.zeros_like(r_pos[..., 0]), r_pos[..., 1]], -1)
        
    # return input_motions    

    # import pdb;pdb.set_trace()
    
    r_rot_quat = torch.stack([rot_cos, torch.zeros_like(rot_cos), rot_sin, torch.zeros_like(rot_cos)], dim=-1) # quaterions
    # r_pos_y = r_pos[..., [1]].clone()
    r_pos = r_pos[:, :, 1:] - r_pos[:, :, :-1]
    r_pos = torch.cat([torch.zeros_like(r_pos[:, :, [0]]), r_pos], dim=2)
    r_pos = qrot(r_rot_quat, r_pos)
    
    # r_rot_ang = torch.atan2(rot_sin, rot_cos).unsqueeze(-1)
    # ang_v = r_rot_ang[:, :, 1:] - r_rot_ang[:, :, :-1]
    
    # ang_v[ang_v > np.pi] -= 2 * np.pi
    # ang_v[ang_v < -np.pi] += 2 * np.pi    
    # ang_v = torch.cat([ang_v, ang_v[:, :, [-1]]], dim=2)
    
    # import pdb;pdb.set_trace()
    r_velocity = qmul_np(r_rot_quat[:, 0, 1:].cpu().numpy(), qinv_np(r_rot_quat[:, 0, :-1].cpu().numpy()))
    r_velocity = np.concatenate([r_rot_quat[:, 0, :1].cpu().numpy(), r_velocity], 1)
        # (seq_len, joints_num, 4)
    anv_v = np.arcsin(r_velocity[:, :, 2:3])
    anv_v = torch.from_numpy(anv_v[:, None])
    
    anv_v[:, 0, 0] = 0 # set start frame as zeros. 
    local_motion = torch.cat([anv_v, r_pos[..., [0, 2]]], dim=-1)
    if dataset is not None:
        local_motion_norm = (local_motion - torch.from_numpy(dataset.mean[:3]).to(local_motion.device)) / torch.from_numpy(dataset.std[:3]).to(local_motion.device)
        local_motion_norm = local_motion_norm.permute(0, 3, 1, 2) # batch, feat, 1, frames
    
    for i in range(len(input_lengths)):
       input_motions[i, :3, 0, :input_lengths[i]] = local_motion_norm[i, :, 0, :input_lengths[i]]
       input_motions[i, :, 0, input_lengths[i]:] *= 0

    return input_motions


def get_sample_input_motion(start_position, end_position, final_frame=196, dataset=None, floor_map_2d=None, model_kwargs=None): #
    
    # TODO:  load Robotics Path Planning Methods.
    
    scale_x = 1 / 256 * 6.4
    scale_z = 1 / 256 * 6.4
    
    middle_position = np.array([200, 20])
    start_position = np.array(start_position, dtype=float)
    end_position = np.array(end_position, dtype=float)
    
    first_part = np.linalg.norm(middle_position-start_position) / np.linalg.norm(end_position-middle_position)
    first_frame = int((final_frame -1 )/ (1+1/first_part))
    
    positions = [start_position]
    current_position = start_position

    # import pdb;pdb.set_trace()
    step_size = (middle_position - start_position) / float(first_frame)
    for i in range(first_frame):
        current_position = current_position + step_size
        positions.append(current_position)
    
    step_size = (end_position - middle_position) / (final_frame - first_frame - 1)
    for i in range(first_frame+1, final_frame):
        current_position = current_position + step_size
        positions.append(current_position)
    
    positions = np.stack(positions)
    positions[:, 0] *= scale_x
    positions[:, 1] *= scale_z
    
    positions = positions - positions[0, :] # normalize 0,0
    # import pdb;pdb.set_trace()
    debug = True 
    if debug:
        plot_2d(positions, 'extra_sample.png')
    if dataset is not None:
        positions = (positions - dataset.mean[:2]) / dataset.std[:2]
    if debug:
        plot_2d(positions, 'extra_sample_norm.png')
        
    return torch.from_numpy(positions).float().T[None, None, :, :] # B, 1, 2, frames


def get_input_motion(ori_input_motions, model_kwargs, input_motion_kind, out_path,  \
                     input_trajectory=None, data=None, input_distance=None, input_length=None, \
                    final_frame=None, floor_map_2d=None, root_representation=None, ):
    
    extra_sample_motions = None # extra input generated trajectory from Robotics Fields.
    # import pdb;pdb.set_trace()
    # get the 2D floor maps.

    can_maps = None
    if floor_map_2d is not None:
        floor_map_2d = Image.open(floor_map_2d)
        can_maps = canonical_maps(floor_map_2d, start_data, start_ori, out_dir=out_path)
    elif 'scene' in model_kwargs['y'].keys(): 
        can_maps = model_kwargs['y']['scene']

    # load specific trajectory.
    # change specific parts of the input motion: such as input_trajectory, input_distance, input_length.
    if input_trajectory is not None: # the a* generated trajectory.

        print(f'input motion kind: {input_motion_kind}')
        
        if input_motion_kind == 'a_star':
            # import pdb;pdb.set_trace()
            input_trajectory = load_trajectory(input_trajectory)
            # TODO: change it with our designed motion trajectory.
            input_motions = insert_body_trajectory(ori_input_motions, input_trajectory, body_part='root_horizontal', dataset=data.dataset)
            
        elif input_motion_kind == 'absolute_xz_rot':
            # import pdb;pdb.set_trace()
            # from utils.input_trajectory import absolute_to_relative, convert_root_global_to_local
            input_results = np.load(input_trajectory, allow_pickle=True).tolist() 
            
            if os.path.exists(os.path.join(os.path.dirname(input_trajectory), 'sorted_idx.npy')):
                idx = np.load(os.path.join(os.path.dirname(input_trajectory), 'sorted_idx.npy'))
                print('redo the idx')
            else:
                idx = np.arange(input_results['motion'].shape[0])
            all_motions = input_results['motion'][idx] #
            input_lengths = input_results['lengths'][idx]
            model_kwargs['y']['lengths'] = torch.tensor(input_lengths).int() # load original dataset.
            
            abs_model_output = input_results['model_output'][0][idx]
            # import pdb;pdb.set_trace()
            if False:
                # TODO: bugs to fix, global translation to local is not correct.
                input_motions = absolute_to_relative(ori_input_motions, all_motions, input_lengths, dataset=data.dataset, abs_model_output=abs_model_output)
                # {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                #  'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
            else:
                input_motions = convert_root_global_to_local(ori_input_motions, input_lengths, abs_model_output, dataset=data.dataset)    
            
            if False:
                if model.data_rep == 'hml_vec':
                    n_joints = 22 if args.dataset == 'humanml' else 21
                root_represent = args.root_representation
                gt_frames_per_sample = {}
                abs_input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
                abs_input_motions = recover_from_ric(abs_input_motions, n_joints, root_rep=root_represent)
                abs_input_motions = abs_input_motions.view(-1, *abs_input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
                
                for ori_sample_i in range(7):
                    length = input_lengths[ori_sample_i]
                    save_file = 'debug_input_motion{:02d}.mp4'.format(ori_sample_i)
                    animation_save_path = os.path.join(out_path, save_file)
                    print(f'[({ori_sample_i}) "" | -> {save_file}]')
                    print('save to', animation_save_path)
                    plot_3d_motion(animation_save_path, skeleton, abs_input_motions[ori_sample_i].transpose(2, 0, 1)[:length], title='',
                                dataset=args.dataset, fps=fps, vis_mode='gt',
                                gt_frames=gt_frames_per_sample.get(ori_sample_i, []))
                    import pdb;pdb.set_trace()
        else:
            assert False, 'not implemented'
        
    elif input_distance is not None:
        # import pdb;pdb.set_trace()
        radius = input_distance
        num_points = ori_input_motions.shape[0]
        theta = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        coordinates = torch.Tensor(list(zip(x, z)))
        # to normalize
        coordinates = (coordinates - data.dataset.t2m_dataset.mean[[0,1]]) / data.dataset.t2m_dataset.std[[0,1]]
        input_motions = ori_input_motions.clone()
        
        for idx, length in enumerate(model_kwargs['y']['lengths']):
            input_motions[idx, [0,1], :, length-1] = coordinates[idx][:, None].float()
        
    elif input_length is not None:
        import pdb;pdb.set_trace()
        input_motions = ori_input_motions.clone()
        for idx, length in enumerate(model_kwargs['y']['lengths']):
            input_motions[idx, :, :, input_length-1] = input_motions[idx, :, :, length-1]
            input_motions[idx, :, :, input_length:] *= 0.0
        model_kwargs['y']['lengths'] = torch.ones(len(model_kwargs['y']['lengths'])).int() * input_length
        
    else:
        input_motions = ori_input_motions
    
    return input_motions, can_maps, extra_sample_motions

if __name__ == '__main__':
    input_path = ''
