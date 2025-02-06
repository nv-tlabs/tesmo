from os.path import join as pjoin
import codecs as cs
import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import orjson
import torch
import joblib
import sys

from torch import Tensor
import os.path as osp
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from PIL import Image
from pyquaternion import Quaternion as Q
import argparse

base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_root)
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.scripts.motion_process import face_joint_indx,uniform_skeleton,tgt_offsets
from data_loaders.humanml.scripts.motion_process import process_file
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_root_rot_pos_abs
from data_loaders.humanml.scripts.motion_process import *

comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ! flip the motion and scene at the begining.
def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def map_originalFrame_to_20fps(bdata): 

    trans_dict = {
        'gender': bdata['gender'],
        'betas': bdata['betas'],
    }
    if 'mocap_frame_rate' in bdata.files:
        ori_frame_rate = bdata['mocap_frame_rate']
        trans_dict['mocap_frame_rate'] = ori_frame_rate 
    elif 'mocap_framerate' in bdata.files:
        ori_frame_rate = bdata['mocap_framerate']
        trans_dict['mocap_framerate'] = ori_frame_rate
    
    ratio = ori_frame_rate / 20 # ! 20 fps used in HumanML3D 

    sample_idx = np.arange(0, bdata['trans'].shape[0], ratio).astype(np.int32)
    for key in bdata.files:
        if key not in ['gender', 'mocap_framerate', 'mocap_frame_rate', 'betas', 'marker_labels']:
            try:
                trans_dict[key] = bdata[key][sample_idx]
            except:
                import pdb; pdb.set_trace()

    return trans_dict

def load_motion(amass_npz_fname, start_frame=0, end_frame=None, viz=False, return_all=False):
    bdata = np.load(amass_npz_fname)  #  bdata.files
    bdata = map_originalFrame_to_20fps(bdata)
    
    # import pdb;pdb.set_trace()
    # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
    subject_gender = bdata['gender']
    if type(subject_gender) == bytes:
        subject_gender = subject_gender.decode()

    #print('Data keys available:%s'%list(bdata.keys()))
    #print('The subject of the mocap sequence is  {}.'.format(subject_gender))
    support_dir = './'
    if subject_gender == 'male':
        bm_fname = os.path.join(support_dir, 'body_models/smplh/SMPLH_MALE.npz')
    else:
        bm_fname = os.path.join(support_dir, 'body_models/smplh/SMPLH_FEMALE.npz')
    dmpl_fname = osp.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
   

    num_betas = 10 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
    faces = c2c(bm.f)
    
    time_length = len(bdata['trans'])

    # if time_length > 5000:
    #     return False
    
    if end_frame is None:
        end_frame = time_length

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][start_frame:end_frame, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][start_frame:end_frame, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][start_frame:end_frame, 66:]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans'][start_frame:end_frame]).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=end_frame-start_frame, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'dmpls': torch.Tensor(bdata['dmpls'][start_frame:end_frame, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
    }

    #print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    #print('time_length = {}'.format(time_length))

    # visualization
    
    
    body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls',
                                                                   'trans', 'root_orient']})
    if viz:
        vis_body_trans_root(0, body_trans_root, faces) # frame id
    
    if return_all:
        return body_trans_root.v, body_trans_root.Jtr, body_trans_root.f
    return body_trans_root

def rotate_2D_points_along_z_axis(points: np.ndarray, angle: float):
    r = Q(axis=[0, 0, 1], angle=angle).rotation_matrix[0:2, 0:2]
    return (r @ points.T).T

def load_mask(fpath):
    if fpath.split('.')[-1] == 'npy':
        mask = np.load(fpath) > 0
    elif fpath.split('.')[-1] == 'png':
        mask = np.array(Image.open(fpath))[:, :, 0] > 0
    H, W = mask.shape[:2]
    assert FLOOR_PLANE_GRID_SIZE % H == 0 and FLOOR_PLANE_GRID_SIZE % W == 0 and H == W
    mask = mask[:, None, :, None].repeat(FLOOR_PLANE_GRID_SIZE//H, axis=1).repeat(FLOOR_PLANE_GRID_SIZE//W, axis=3)
    mask = mask.reshape(FLOOR_PLANE_GRID_SIZE, FLOOR_PLANE_GRID_SIZE)
    return mask

def get_line_points(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        points.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    points.append((x2, y2))
    return points

def get_easy_hard_kind(scene_map, root_trajectoy, start_point, end_point, file_idx, plot_flag=True):
    # 0 is easy case, with no collision between two points
    # 1 is hard case, with collision between two points
    if scene_map.max() < 255:
        scene_map = scene_map.copy() * 255.0

    points = get_line_points(start_point[0], start_point[1], end_point[0], end_point[1])
    all_value = 0
    for o_i in points:
        x = np.max([0, o_i[0]])
        x = np.min([255, x])

        y = np.max([0, o_i[1]])
        y = np.min([255, y])
        all_value += 255.0 - scene_map[y, x]  # first is y, second is x

    if plot_flag:
        plt.figure(figsize=(10, 10))
        plt.imshow(scene_map)
        plt.scatter(root_trajectoy[:, 0], root_trajectoy[:, 1], marker='.', color='g')
        plt.scatter(np.array(points)[:, 0], np.array(points)[:, 1], marker='.', color='r')
        plt.colorbar()
        plt.savefig(f'debug_results2/easy_hard_{file_idx}.png')
        plt.close('all')
    

    if all_value > 3 * 255:
        # if 1:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(scene_map)
        #     plt.scatter(np.array(points)[:, 0], np.array(points)[:, 1], marker='.', color='r')
        #     plt.colorbar()
        #     plt.savefig(f'debug_results/easy_hard_{file_idx}.png')
        #     plt.close('all')
        #     print('debug_results/easy_hard.png')

        return 1
    else:
        return 0
    

def canonicalize_motion_and_scene(pose_seq_np, scene_data_input):                                                      
    # xzy -> xyz: z is depth, y is height
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
    
    from PIL import Image, ImageChops
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
    
    root_pose_init_xz_img = root_pose_init_xz / scale_rt / FLOOR_PLANE_GRID_SCALE

    transl_x=int((int(root_pose_init_xz_img[0])-128) * scale_rt)# translation is problem.
    transl_y=int((int( root_pose_init_xz_img[2])-128) * scale_rt)

    translated_image = img_scale.transform(img_scale.size, Image.AFFINE, (1, 0, transl_x, 0, 1, transl_y))
    root_init_euler = qeuler(torch.tensor(root_quat_init), 'xzy', epsilon=0, deg=True)
    
    # import pdb;pdb.set_trace()
    # record the translation and rotation info
    transform_vector = np.concatenate([root_pose_init_xz, root_init_euler[:, 1], root_init_matrix.reshape(-1)])
    
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
    
    return motion, scene_data_process, transform_vector, scale_rt


def get_root_pos_angle_from_guofeat(motion):
    r_rot_ang, r_pos = recover_root_rot_pos_abs(torch.from_numpy(motion).cuda().float()) # get root joint rotation and position， here
    r_cos = torch.cos(r_rot_ang)
    r_sin = torch.sin(r_rot_ang)
    
    # remain_list=[] 
    # motion: shape:(199, 5)
    motion = np.concatenate((r_pos.cpu().numpy()[:, [0, 2]], r_cos[:, None].cpu().numpy(), r_sin[:, None].cpu().numpy(), r_pos.cpu().numpy()[:, [1]]), -1)

    # ! only hack the y dimension
    # * why add scene maps, we need to use this hack ???? 
    motion[:, 1] = motion[:, 1] * -1
    angle = np.arctan2(motion[:, 3], motion[:,2])
    motion[:, 2] = np.cos(-angle) # -180
    motion[:, 3] = np.sin(-angle)

    return motion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--split", type=str, default="train") # samp or amass
    parser.add_argument("--fitting_folder", type=str, default="align_data_obj_v2")
    parser.add_argument("--scene_folder", type=str, default="dataset/3d_front_scene_mask/livingroom")
    args = parser.parse_args()

    data_dir = args.fitting_folder
    scene_folder = args.scene_folder
    split = args.split

    FLOOR_PLANE_GRID_SIZE = 256 # size of floor plane grids (square)
    FLOOR_PLANE_GRID_SCALE = 6.2*2/FLOOR_PLANE_GRID_SIZE # scale of floor plane grid (metre) 
    trans_matrix = np.array([[1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]])
    joints_num = 22   
    
    update_data_dir = data_dir + 'store'
    os.makedirs(update_data_dir, exist_ok=True)
    
    new_id_list = glob.glob(os.path.join(data_dir, '*.npy'))
    iter_id_list = np.sort(new_id_list)

    json_path = f'dataset/HumanML3D/annotations.json'
    with open(json_path, "rb") as ff:
        annotations = orjson.loads(ff.read())

    text_folder = f'dataset/HumanML3D/texts/'

    motion_total_num = 0
    motion_text_info = {}
    for cnt, id_name in enumerate(tqdm(iter_id_list)):
        
        motion_id = os.path.basename(id_name).split('.')[0]
        try:
            motion_text = annotations[motion_id]['annotations']
            motion_text = [text['text'] for text in motion_text]
            motion_text_info[motion_id] = motion_text

            motion_text = annotations['M'+motion_id]['annotations']
            motion_text = [text['text'] for text in motion_text]
            motion_text_info['M'+motion_id] = motion_text
        except:
            print(f'Can not find {motion_id} in annotations !!!')
            continue
        
        with open(f'{text_folder}/{motion_id}.txt', 'r') as file:
            motion_text = file.read()
            motion_text_list = motion_text.split('\n')
            motion_text_dict = []
            for _text in motion_text_list:
                if _text == '': continue
                text_dict = {}
                text_dict['caption'] = _text.split('#')[0]
                text_dict['tokens'] = _text.split('#')[1]
                motion_text_dict.append(text_dict)

        with open(f'{text_folder}/M{motion_id}.txt', 'r') as file:
            motion_text = file.read()
            motion_text_list = motion_text.split('\n')
            mirror_motion_text_dict = []
            for _text in motion_text_list:
                if _text == '': continue
                text_dict = {}
                text_dict['caption'] = _text.split('#')[0]
                text_dict['tokens'] = _text.split('#')[1]
                mirror_motion_text_dict.append(text_dict)
        
        save_data_dir = f"{update_data_dir}/{motion_id}"
        os.makedirs(save_data_dir, exist_ok=True)
        
        motion_npy = np.load(id_name, allow_pickle=True) # list
        motion_npz_path = motion_npy[0]['motion']
        print('load ', motion_npz_path)
        
        start_f, end_f = motion_npy[0]['start_f'], motion_npy[0]['end_f']   
        all_verts, pose_seq, body_face = load_motion(motion_npz_path, start_frame=start_f, end_frame=end_f, viz=False, return_all=True)

        ori_all_points = pose_seq.detach().cpu().numpy() # 103, 52, 3
        print('starf_f, end_f:', start_f, end_f)
        
        floor_maps_num = len(motion_npy)
        for motion_idx in tqdm(range(0, floor_maps_num)):
            
            align_info = motion_npy[motion_idx]
            all_points = ori_all_points.copy()
            
            # get rot and translation for the motion.
            len_traj = len(all_points)
            pelvis_xy_rotate = align_info['pelvis_xy_rotate']

            # rotate and translate on the scene maps.
            all_points -= np.array([*pelvis_xy_rotate, 0])
            all_points[:, :, 0:2] = rotate_2D_points_along_z_axis(all_points[:, :, 0:2].reshape(-1, 2), align_info['rotation'][0]).reshape(len_traj, -1, 2)
            all_points = (all_points.reshape(len_traj, -1, 3) + align_info['translation'][0][None, None]).reshape(len_traj, -1, 3)
            pose_seq_np_ori = all_points
            
            # load 2D scene map
            scene_path = align_info['scene']
            # print(scene_path)
            scene_id = os.path.basename(scene_path).split('_')[-1]
            ### assume that each scene are represented with a 2D image mask
            scene_data_ori = cv2.imread(f'{scene_folder}/{scene_id}.png')
            scene_data_ori = scene_data_ori[:,:,0]
            
            # print(scene_path)
            # continue
            pose_seq_np_ori_flip = swap_left_right(pose_seq_np_ori.copy())
            pose_seq_np_ori_flip[..., 0] = pose_seq_np_ori_flip[..., 0] + 6.2 * 2
            scene_data_ori_flip = np.fliplr(scene_data_ori.copy())

            for tmp_i, (scene_data, pose_seq_np) in enumerate(zip([scene_data_ori, scene_data_ori_flip], [pose_seq_np_ori, pose_seq_np_ori_flip])):
                floor_plane_mask_img = scene_data[:, :]
                xs, ys = pose_seq_np[:, :1, 0].squeeze(), pose_seq_np[:, :1, 1].squeeze()
                H, W = floor_plane_mask_img.squeeze().shape[:2]

                grid_x, grid_y = np.floor(xs / FLOOR_PLANE_GRID_SCALE).astype(np.int32), np.floor(ys / FLOOR_PLANE_GRID_SCALE).astype(np.int32)
                root_trajectoy = np.concatenate([grid_x.reshape(-1,1), grid_y.reshape(-1,1)], axis=-1)
                # kind = get_easy_hard_kind(floor_plane_mask_img, root_trajectoy, (grid_x[0], grid_y[0]), (grid_x[-1], grid_y[-1]), f'{motion_id}_{motion_idx}', plot_flag=True)
                
                motion, scene_data_process, transform_vector, scale_rt = canonicalize_motion_and_scene(pose_seq_np, scene_data)

                caption_dict = mirror_motion_text_dict if tmp_i == 1 else motion_text_dict
                data = {}
                data['motion'] = motion
                data['scene'] = scene_data_process
                data['length'] = len(motion)
                data['motion_scale_rt'] = scale_rt
                data['transform_vector'] = transform_vector
                for caption in caption_dict:
                    caption['tokens'] = caption['tokens'].split(' ')
                data['caption'] = caption_dict 
                joblib.dump(data, f'{save_data_dir}/{motion_id}_{motion_idx}_{tmp_i}.pkl')
                motion_total_num += 1

    print(f'{split} data has {motion_total_num} motion-scene pairs!')

    