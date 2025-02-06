import os
import numpy as np
from data_loaders.humanml.scripts.motion_process import Skeleton, face_joint_indx, n_raw_offsets, kinematic_chain, uniform_skeleton, tgt_offsets
from visualize.vis_modules import get_transform_mat
from PIL import Image, ImageChops
from data_loaders.humanml.data.dataset import load_motion
from data_loaders.humanml.scripts.motion_process import process_file
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.data.dataset import rotate_2D_points_along_z_axis

code_dir = os.path.dirname(__file__)+'/../../..'
FLOOR_PLANE_GRID_SIZE = 256 # size of floor plane grids (square)
FLOOR_PLANE_GRID_SCALE = 6.2*2/FLOOR_PLANE_GRID_SIZE # scale of floor plane grid (metre)   
ROOT_DATA_DIR= f'{code_dir}/dataset/SAMP_interaction/'
body_data_dir = f'{ROOT_DATA_DIR}/body_meshes_smpl'

def get_cont6d_params_only(positions):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    # velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    # velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    # r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_rot
    
# TODO: merge the data loader with bps and sdf;

def get_generated_loco_motion(input_trajectory, sample_idx, return_joint=False, root_z_flip=True):
    # import pdb;pdb.set_trace()

    input_results =  np.load(os.path.join(input_trajectory, 'results.npy'), allow_pickle=True).tolist() # prior_mdm generated results.
    # input_results = input_results['arr_0']

    # batch, 1, frames, dim
    input_model_outputs = input_results['model_output'][0] # walk end pose | after re-normalization, need to use normalization from dataset.
    input_model_lenghts = input_results['lengths']
    
    # only use one sample poses.
    # sample_idx = 3
    input_model_output_sample = input_model_outputs[sample_idx] # 1, frames, dim
    input_model_lengths_sample = input_model_lenghts[sample_idx]
    
    # import pdb;pdb.set_trace()
    if return_joint: 
        input_model_joints = input_results['motion'][sample_idx] # 22, 3, frames (196)
        # if root_z_flip:
        #     import pdb;pdb.set_trace()
        #     start_z = input_model_joints[0, 2, 0]
        #     input_model_joints[0, 2] = - 2 * start_z

        return input_model_output_sample, input_model_lengths_sample, input_model_joints

    return input_model_output_sample, input_model_lengths_sample


def get_generated_interactive_motion(input_trajectory, sample_idx, return_joint=False, load_tranform_mat=True, transform_mat_dir=None):
    # import pdb;pdb.set_trace()

    # TODO: add transform_mat.
    if load_tranform_mat:
        # load input k.
        if transform_mat_dir is not None:
            transform_mat = get_transform_mat(transform_mat_dir, sample_idx)
        else:
            transform_mat = get_transform_mat(input_trajectory, sample_idx)

    input_results =  np.load(os.path.join(input_trajectory, 'results.npy'), allow_pickle=True).tolist() # prior_mdm generated results.
    # input_results = input_results['arr_0']

    # batch, 1, frames, dim
    input_model_outputs = input_results['model_output'][0] # walk end pose | after re-normalization, need to use normalization from dataset.
    input_model_lenghts = input_results['lengths']
    
    # only use one sample poses.
    # sample_idx = 3
    input_model_output_sample = input_model_outputs[sample_idx] # 1, frames, dim
    input_model_lengths_sample = input_model_lenghts[sample_idx]
    
    # import pdb;pdb.set_trace()
    if return_joint: 
        input_model_joints = input_results['motion'][sample_idx] # 22, 3, frames (196)

        if load_tranform_mat:
            if transform_mat.size == 4:
                x, z, cos, sin = transform_mat
                R = np.zeros((3, 3))
                R[0, 0] = cos
                R[0, 2] = -sin
                R[2, 0] = sin
                R[2, 2] = cos
                R[1, 1] = 1.
                T = np.array([x, 0, z]).reshape(-1)
            else:
                R = transform_mat[:3, :3]
                T = transform_mat[:3, -1]
            input_model_joints_world = np.matmul(R.T[None], input_model_joints.transpose(2, 1, 0)).transpose(2, 1, 0) \
                + np.tile(T[None, :, None], (22, 1, 196))
            return input_model_output_sample, input_model_lengths_sample, input_model_joints, input_model_joints_world

        return input_model_output_sample, input_model_lengths_sample, input_model_joints

    return input_model_output_sample, input_model_lengths_sample



def replace_start_pose(input_motion, input_model_output_sample, input_model_lengths_sample, data):
    input_motions = input_motion.clone()

    batch, dim, _, frames = input_motions.shape
    insert_start_pose = input_model_output_sample[:, input_model_lengths_sample-1:input_model_lengths_sample, :] # end pose with 263;

    # 263: first 4 is for root;
    insert_start_pose_norm = (insert_start_pose - data.dataset.t2m_dataset.mean[5:]) / data.dataset.t2m_dataset.std[5:]

    input_motions[:, 5:, :, 0] = torch.from_numpy(insert_start_pose_norm).to(input_motions.device).float().repeat((batch, 1, 1)).permute(0, 2, 1) # not replace the local positions; 

    return input_motions

def canonical_object_start_pose(): # the object is center-origin and z-up.
    pass


# TODO: change the 2D floor maps  as scene_maps; let scene only works for objects (BPS or SDF)
def transform_scene(scene_data_input, trans_xz, rot_y, scale_rt, center=False):
    
    # import pdb;pdb.set_trace()
    if scene_data_input.max() == 255.0:
        image = Image.fromarray((scene_data_input).astype(np.uint8))
    else: # ! this is used in 2D floor map originally
        image = Image.fromarray((scene_data_input*255).astype(np.uint8))

    scale_w = int(256*scale_rt)
    img_scale=image.resize((scale_w, scale_w))

    root_pose_init_xz_img = trans_xz / scale_rt /FLOOR_PLANE_GRID_SCALE

    if center:
        transl_x=int((int(root_pose_init_xz_img[0])) * scale_rt)# translation is problem.
        transl_y=int((int( root_pose_init_xz_img[1])) * scale_rt)
    else:
        transl_x=int((int(root_pose_init_xz_img[0])-128) * scale_rt)# translation is problem.
        transl_y=int((int( root_pose_init_xz_img[1])-128) * scale_rt)

    # import pdb;pdb.set_trace()
    translated_image = img_scale.transform(img_scale.size, Image.AFFINE, (1, 0, transl_x, 0, 1, transl_y))
    
    # * always top_left as original point;
    rot_img = translated_image.rotate(rot_y, center=(int(128*scale_rt),int(128*scale_rt)))

    def process_image(image):
        width, height = image.size          

        # Check if image is smaller than (256, 256)
        if width < 256 or height < 256:
            # Create a new blank image with size (256, 256) and black background
            # new_image = Image.new("RGB", (256, 256), (0, 0, 0))
            new_image = Image.new("L", (256, 256), (0))

            # Calculate the position to paste the original image
            x = (256 - width) // 2
            y = (256 - height) // 2

            # Paste the original image onto the new image
            new_image.paste(image, (x, y))

            return new_image

        # Check if image is larger than (256, 256)
        if width > 256 or height > 256:
            # Calculate the coordinates for center cropping
            left = (width - 256) // 2
            top = (height - 256) // 2
            right = left + 256
            bottom = top + 256

            # Crop the image using the calculated coordinates
            cropped_image = image.crop((left, top, right, bottom))

            return cropped_image

        # Image is already (256, 256)
        return image

    scene_data_process = np.array(process_image(rot_img))

    # scene_data = scene_data[:256,:256]
    scene_data_process = scene_data_process[None,...]
    
    return scene_data_process

def canonicalize_motion_and_scene(pose_seq_np, scene_data_input=None):                                                      
    joints_num = 22
    trans_matrix = np.array([[1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]])

    # in xz plane coord /
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)#(68, 55, 3) # face orientation is wrong.

    # ! this is the key problem.
    pose_seq_np_n[..., 2] = pose_seq_np_n[..., 2] # * -1

    source_data = pose_seq_np_n[:, :joints_num]#(68, 22, 3)
    motion, ground_positions, positions, l_velocity = process_file(source_data, 0.002)

    # 获取AMASS到HumanML3D的偏移和旋转
    positions=source_data
    positions, scale_rt = uniform_skeleton(positions, tgt_offsets)
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target) # rotation;
    root_init_matrix = quaternion_to_matrix_np(root_quat_init)
    
    root_init_euler = qeuler(torch.tensor(root_quat_init), 'xzy', epsilon=0, deg=True)
        
    # import pdb;pdb.set_trace()
    # record the translation and rotation info
    transform_vector = np.concatenate([root_pose_init_xz, root_init_euler[:, 1], root_init_matrix.reshape(-1)])

    if scene_data_input is None:
        return motion, transform_vector, scale_rt
    else:
        image = Image.fromarray((scene_data_input*255).astype(np.uint8))
        scale_w = int(256*scale_rt)
        img_scale=image.resize((scale_w, scale_w))
    
        # new_width = scale_w
        # new_height = scale_w
        # scaled_image = image.resize((new_width, new_height))
        # new_image = Image.new('1', image.size)
        # x_offset = int((image.width - new_width) / 2)
        # y_offset = int((image.height - new_height) / 2)
        # new_image.paste(scaled_image, (x_offset, y_offset))
        # img_scale = new_image
        
        root_pose_init_xz_img = root_pose_init_xz / scale_rt /FLOOR_PLANE_GRID_SCALE

        transl_x=int((int(root_pose_init_xz_img[0])-128) * scale_rt)# translation is problem.
        transl_y=int((int( root_pose_init_xz_img[2])-128) * scale_rt)

        translated_image = img_scale.transform(img_scale.size, Image.AFFINE, (1, 0, transl_x, 0, 1, transl_y))
        
        # TODO rotation;
        rot_img = translated_image.rotate(root_init_euler[0][1], center=(int(128*scale_rt),int(128*scale_rt))) # * always top_left as original point;

        def process_image(image):
            width, height = image.size          

            # Check if image is smaller than (256, 256)
            if width < 256 or height < 256:
                # Create a new blank image with size (256, 256) and black background
                # new_image = Image.new("RGB", (256, 256), (0, 0, 0))
                new_image = Image.new("L", (256, 256), (0))

                # Calculate the position to paste the original image
                x = (256 - width) // 2
                y = (256 - height) // 2

                # Paste the original image onto the new image
                new_image.paste(image, (x, y))

                return new_image

            # Check if image is larger than (256, 256)
            if width > 256 or height > 256:
                # Calculate the coordinates for center cropping
                left = (width - 256) // 2
                top = (height - 256) // 2
                right = left + 256
                bottom = top + 256

                # Crop the image using the calculated coordinates
                cropped_image = image.crop((left, top, right, bottom))

                return cropped_image

            # Image is already (256, 256)
            return image

        scene_data_process = np.array(process_image(rot_img))

        # scene_data = scene_data[:256,:256]
        scene_data_process = scene_data_process[None,...]
        # utterance = 'walk'

        return motion, scene_data_process, transform_vector
    

def load_one_motion_example(test_split, kind='loco_motion'): 
    if kind == 'loco_motion':
        with open(test_split, 'r') as f:
            test_list = f.readlines()
            test_list = [one[:-1] for one in test_list]

        # loco motion from test split of Humanml3D
        cnt = 0
        id_name = test_list[cnt]

        if not os.path.exists(id_name):
            # cluster_folder = ''
            id_name = id_name.replace('/home/hyi/data/dataset/humanise_bundle/scene_aware_motion_gen', code_dir)

        motion_npy = np.load(id_name, allow_pickle=True) 
        motion_idx = 0 # only one 

        motion_npz_path = motion_npy[0]['motion']

        print(motion_npz_path)
        if '/home/hyi/data/workspace/motion_generation/priorMDM' in motion_npz_path:
            motion_npz_path = motion_npz_path.replace('/home/hyi/data/workspace/motion_generation/priorMDM', code_dir)
            print('load ', motion_npz_path)
        
        start_f, end_f = motion_npy[0]['start_f'], motion_npy[0]['end_f']   

        # import pdb;pdb.set_trace()
        
        all_verts, pose_seq, body_face = load_motion(motion_npz_path, start_frame=start_f, end_frame=end_f, viz=False, return_all=True)
        ori_all_points = pose_seq.detach().cpu().numpy()

        print('starf_f, end_f:', start_f, end_f)

        align_info = motion_npy[motion_idx]
        all_points = ori_all_points.copy()
        len_traj = len(all_points)

        pelvis_xy_rotate = align_info['pelvis_xy_rotate']
        all_points -= np.array([*pelvis_xy_rotate, 0])
        all_points[:, :, 0:2] = rotate_2D_points_along_z_axis(all_points[:, :, 0:2].reshape(-1, 2), align_info['rotation'][0]).reshape(len_traj, -1, 2)
        all_points = (all_points.reshape(len_traj, -1, 3) + align_info['translation'][0][None, None]).reshape(len_traj, -1, 3)
        pose_seq_np_ori = all_points

        motion, transform_vector, scale_rt = canonicalize_motion_and_scene(pose_seq_np_ori)

    else:
        pass

    # interactive motion from test split of SAMP
    return motion, transform_vector, scale_rt

def check_inside_floor(scene_map, point): # y > 0, down;
    x, y = point * FLOOR_PLANE_GRID_SCALE + 128
    if scene_map[y, x] > 0:
        return True # inside the floor;
    else:
        return False