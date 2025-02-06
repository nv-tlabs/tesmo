import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
# import spacy
import itertools
import os.path as osp
import copy

from torch.utils.data._utils.collate import default_collate
# from data_loaders.amass.sampling import FrameSampler

from data_loaders.humanml_utils import HML_USE_LOCAL_JOINT_MASK

# to get original joint positions.
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_root_rot_pos_abs
from data_loaders.humanml.common.quaternion import quaternion_to_cont6d
from data_loaders.amass.babel import BABEL

import matplotlib.pyplot as plt
from PIL import Image
import glob
import torch
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from data_loaders.humanml.scripts.motion_process import process_file, recover_from_ric
from scipy.spatial.transform import Rotation as R
import joblib
# Set 'PATH' to the default system path
os.environ['PATH'] = os.defpath # this is used to fix "import trimesh" in the cluster.

import trimesh
from trimesh import transform_points
from pyquaternion import Quaternion as Q

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *
from data_loaders.amass.tools_teach.easyconvert import to_matrix, matrix_to
from data_loaders.humanml.scripts.motion_process import face_joint_indx,uniform_skeleton,tgt_offsets
import json

male_bm_path = os.path.join(os.path.dirname(__file__), '../../../body_models/smplx/SMPLX_NEUTRAL.npz') 

code_dir = os.path.dirname(__file__)+'/../../..'
HUMANISE_DIR = f'{code_dir}/dataset/3dfront_fitting'
walk_anno_root_path = f'{HUMANISE_DIR}/files/align_data_release/walk'
walk_anno_pattern = f'{HUMANISE_DIR}/files/align_data_release/walk/*/anno.pkl'


code_dir = os.path.dirname(__file__)+'/../../..'


MOTION_TYPES = [
    '_0',
    '_1',
    '_0_with_transition',
    '_1_with_transition',
]


FLOOR_PLANE_GRID_SIZE = 256 # size of floor plane grids (square)
FLOOR_PLANE_GRID_SCALE = 6.2*2/FLOOR_PLANE_GRID_SIZE # scale of floor plane grid (metre)                    
comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    # import pdb;pdb.set_trace()
    sample_idx = np.arange(0, bdata['trans'].shape[0], ratio).astype(np.int32)
    for key in bdata.files:
        if key not in ['gender', 'mocap_framerate', 'mocap_frame_rate', 'betas']:
            trans_dict[key] = bdata[key][sample_idx]

    return trans_dict
    
def load_motion(amass_npz_fname, start_frame=0, end_frame=None, viz=False, return_all=False):
    bdata = np.load(amass_npz_fname)
    bdata = map_originalFrame_to_20fps(bdata)
    
    # import pdb;pdb.set_trace()
    # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
    subject_gender = bdata['gender']
    if type(subject_gender) == bytes:
        subject_gender = subject_gender.decode()

    #print('Data keys available:%s'%list(bdata.keys()))
    #print('The subject of the mocap sequence is  {}.'.format(subject_gender))
    support_dir = os.path.dirname(__file__)+'/../../../thirdparty/HumanML3D'
    bm_fname = osp.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
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

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

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
    
def get_easy_hard_kind(scene_map, start_point, end_point):
    # 0 is easy case, with no collision between two points
    # 1 is hard case, with collision between two points
    if scene_map.max() < 255:
        scene_map = scene_map.copy() * 255.0

    points = get_line_points(start_point[0], start_point[1], end_point[0], end_point[1])
    all_value = 0
    for o_i in points:
        all_value += 255.0 - scene_map[o_i[1], o_i[0]]  # first is y, second is x

    
    if all_value > 3 * 255:
    # if all_value > 6 * 255:
        if 1:
            plt.figure(figsize=(10, 10))
            plt.imshow(scene_map)
            plt.scatter(np.array(points)[:, 0], np.array(points)[:, 1], marker='.', color='r')
            plt.colorbar()
            plt.savefig('debug_results/easy_hard.png')
            plt.close('all')
            print('debug_results/easy_hard.png')

        return 1
    else:
        return 0

'''For use of training text motion matching model, and evaluations'''
# this is the dataset for 2D floor maps as condition.
class Text2MotionDatasetV2(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, keep_last_step=False, **kwargs):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.num_frames = num_frames if num_frames else False
        self.max_motion_length = opt.max_motion_length
        self.keep_last_step = keep_last_step

        if (self.num_frames == False) or type(self.num_frames)==int:
            min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        else:
            min_motion_len = self.num_frames[0]
            self.max_motion_length = self.num_frames[1]
        
        if hasattr(self.opt, 'motion_window_length'):
            print(f'reset max motion length from {self.max_motion_length} to {self.opt.motion_window_length}')
            self.max_motion_length = self.opt.motion_window_length
            
        max_motion_length_filter = 200
        # for SAMP_all Only
        # if hasattr(self.opt, 'samp_data') and self.opt.samp_data:
        # import pdb;pdb.set_trace()
        if 'SAMP' in self.opt.data_root:
            print('SAMP data loader #####')
            min_motion_len = 20
            max_motion_length_filter = 400
        
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:size] # only load the first size data

        self.is_train = 'train' in split_file
        if self.is_train: 
            print('status: train !!!') 
        else: 
            print('status test ---')

        # import pdb;pdb.set_trace()
        self.extra_rootRotPos_feat = getattr(self.opt, 'extra_rootRotPos_feat', False)
        
        self.max_distances = getattr(self.opt, 'max_distances', None)
        self.debug = getattr(self.opt, 'debug', 0)
        
        print(f'max_distance: {self.max_distances}')
        
        if self.debug:
            self.debug_dir = os.path.join('./debug', os.path.basename(kwargs['opt_name']).split('.')[0])
            os.makedirs(self.debug_dir, exist_ok=True)
        else:
            self.debug_dir = None
        print('debug: ', self.debug)
        print('debug dir: ', self.debug_dir)
             
        male_bm = BodyModel(bm_fname=male_bm_path, num_betas=10).to(comp_device)
        faces = c2c(male_bm.f)
        joints_num = 22        
        # dataset_type = 'humanise'

        if not hasattr(self.opt, 'dataset_type'):
            dataset_type = 'npz'
        else:
            dataset_type = self.opt.dataset_type
        
        dataset_dict = {
            'npz': 'amass in 3D FRONT',
            'pkl': 'MIME generate scenes',
        }
        print(f'load dataset type: {dataset_dict[dataset_type]}')

        # restore=False # the global pkl | for adding humanml local pose.
        restore = getattr(self.opt, 'restore', True)
        restore_npz = False # the each npz file

        ## half_hard
        self.half_hard_case = getattr(self.opt, 'half_hard_case', False)

        self.grid_search_pos_ori = getattr(self.opt, 'grid_search_pos_ori', False) # this is for sanity check different pos and ori
        if self.grid_search_pos_ori:
            restore=False # the global pkl
            # restore_npz = True # the each npz file
            size = 1
            id_frame = int(getattr(self.opt, 'id_frame', 0))
        else:
            id_frame = None

        print('---------------------')
        print('half_hard_case: ', self.half_hard_case)
        print('grid_search_pos_ori: ', self.grid_search_pos_ori)
        print('restore: ', restore)
        print('restore_npz: ', restore_npz)
        print('id_frame: ', id_frame)
        print('size : ', size)
        print('---------------------\n')

        self.floor_maps_num = getattr(self.opt, 'floor_maps_num', None)
        self.floor_map_idx = getattr(self.opt, 'floor_map_idx', 0)
        self.all_samples = getattr(self.opt, 'all_samples', None)

        # floor_maps_num: 1
        # all_samples: 33
        print('floor_maps_num: ', self.floor_maps_num)
        print('all_samples : ', self.all_samples)
        
        # import pdb;pdb.set_trace()

        # load input data and cooresponding sample idx.
        data_store_path = getattr(self.opt, 'data_store_path', None)
        sample_id = getattr(self.opt, 'sample_id', None)
        print('data_store_path: ', data_store_path)
        print('sample_id : ', sample_id)
        print('---------------------\n')

        self.eval_locomotion = not self.is_train
        print('eval_locomotion: ', self.eval_locomotion)

        if data_store_path is not None:
            print('load from ', data_store_path)
            restore = True
        else:        
            if self.is_train:
                data_dir = f'{HUMANISE_DIR}/align_data_obj_v2'
                restore_data_dir = f'{HUMANISE_DIR}/align_data_obj_v2_store'
                if self.opt.root_representation == 'root_pos_abs_rotcossin_height_pose_with_humanml':
                    data_store_path = f'{restore_data_dir}/all_data_train_{dataset_type}_268.pkl'
                else:
                    data_store_path = f'{restore_data_dir}/all_data_train_{dataset_type}.pkl'
            else:
                # ! change the dir to test split
                data_dir = f'{HUMANISE_DIR}/align_data_obj_v2_test'
                restore_data_dir = f'{HUMANISE_DIR}/align_data_obj_v2_test_store'
                if self.opt.root_representation == 'root_pos_abs_rotcossin_height_pose_with_humanml':
                    data_store_path = f'{restore_data_dir}/all_data_test_{dataset_type}_{size}_hardHalf_{self.half_hard_case}_268.pkl'
                else:
                    data_store_path = f'{restore_data_dir}/all_data_test_{dataset_type}_{size}_hardHalf_{self.half_hard_case}.pkl'

        
        restore = False
        if restore and os.path.exists(data_store_path) and dataset_type == 'npz':
            print(f'load data !!!! from {data_store_path}')
            input_all_data = pickle.load(open(data_store_path, 'rb'))
            name_list = input_all_data['name_list']
            length_list = input_all_data['length_list']
            data_dict = input_all_data['data_dict']
            
            if sample_id is not None:
                name_list = name_list[sample_id]
                length_list = length_list[sample_id]
                data_dict = {k:v[sample_id] for k,v in data_dict.items()}
                
        else:
            if dataset_type == 'npz':
                new_id_list = glob.glob(os.path.join(data_dir, '*.npy'))
                
                if self.is_train:
                    train_split = os.path.join(data_dir, 'train.txt')
                    test_split = os.path.join(data_dir, 'test.txt')
                    if not os.path.exists(train_split):
                        print('save to train and test split to ', train_split)
                        import random
                        random.shuffle(id_list)
                        
                        train_list = [new_id_list[i] for i in range(int(len(new_id_list) * 0.8))]
                        test_list = [new_id_list[i] for i in range(int(len(new_id_list) * 0.8), len(new_id_list))]
                        
                        with open(train_split, 'w') as f:
                            f.writelines([one+'\n' for one in train_list])
                        with open(test_split, 'w') as f:
                            f.writelines([one+'\n' for one in test_list])
                    else:
                        print('load train and test split from ', train_split, test_split)
                        with open(train_split, 'r') as f:
                            train_list = f.readlines()
                            train_list = [one[:-1] for one in train_list]
                        with open(test_split, 'r') as f:
                            test_list = f.readlines()
                            test_list = [one[:-1] for one in test_list]

                    id_list = train_list
                    self.half_hard_case = False

                else:
                    print('all list: ', len(new_id_list))
                    
                    if not self.half_hard_case:
                        if id_frame is None:
                            id_list = new_id_list[:size+10]
                        else:
                            id_list = new_id_list[id_frame:id_frame+1]
                    else:
                        id_list = new_id_list # all in the test split.
                    
                print('inference all list length: ', len(id_list))
                
            new_name_list = []
            length_list = []
            
            save_vis_dir = './debug_vis_npz_maps'
            os.makedirs(save_vis_dir, exist_ok=True)

            easy_cnt = 0
            hard_cnt = 0

            if self.all_samples is not None:
                iter_id_list = id_list[:self.all_samples]
            else:
                iter_id_list = id_list

            useful_list = copy.deepcopy(iter_id_list)
            useless_list = []
            all_kind_list = []
            
            for cnt, id_name in enumerate(tqdm(iter_id_list)):
                print(cnt, id_name)
                # only load the first size data including half of easy case and hard case. | only use for test.
                print('----------- easy_cnt, hard_cnt: ', easy_cnt, hard_cnt)
                if self.half_hard_case and easy_cnt + hard_cnt >= size: 
                    break

                try:
                    if dataset_type == 'npz': # AMASS Motion into 3D FRONT dataset
                        process_path_name = os.path.basename(id_name)

                        # load preporcessed scene-motion paris from pkl file
                        print('load exampls from ', os.path.join(restore_data_dir, process_path_name.split('.')[0]))
                        sub_files = glob.glob(os.path.join(restore_data_dir, process_path_name.split('.')[0], '*.pkl'))
                        for one_files in sub_files:
                            data = joblib.load(one_files)
                            dict_name = os.path.basename(one_files).split('.')[0]
                            if np.isnan(data['motion']).sum() > 0:
                                print('skip :', one_files)
                                continue
                            
                            _, _, mirror_flag = os.path.basename(one_files).split('.')[0].split('_')
                            if int(mirror_flag) == 1 and self.is_train==False: continue

                            ## check hard or easy case.
                            if self.half_hard_case:
                                floor_plane_mask_img = data['scene'][:, :].squeeze()
                                pose_seq_np = data['motion'].copy()
                                length = data['length']
                                xs, ys = pose_seq_np[:length, 0], pose_seq_np[:length, 1]
                                ys = ys * -1
                                H, W = floor_plane_mask_img.shape[:2]
                                grid_x, grid_y = np.floor(xs / FLOOR_PLANE_GRID_SCALE).astype(np.int32)+128, np.floor(ys / FLOOR_PLANE_GRID_SCALE).astype(np.int32)+128
                                
                                kind = get_easy_hard_kind(floor_plane_mask_img, (grid_x[0], grid_y[0]), (grid_x[-1], grid_y[-1]))
                                if kind == 0:
                                    if easy_cnt >= size//2:
                                    # if easy_cnt >= 0:
                                        continue
                                    easy_cnt += 1
                                else:
                                    if hard_cnt >= size //2:
                                    # if hard_cnt >= size:
                                        continue
                                    hard_cnt += 1
                            
                            data_dict[dict_name] = data
                            new_name_list.append(dict_name)
                            length_list.append(data['motion'].shape[0])
                            # import pdb;pdb.set_trace()

                except Exception as e:
                    print(e)
                    useful_list.remove(id_name)
                    useless_list.append(id_name)
                    pass
            
            if self.grid_search_pos_ori:
                # grid search the final position and orientation. 
                # batch_size 253 with the original one.
                def check_inside_floor(scene_map, point): # y > 0, down;
                    x, y = point * FLOOR_PLANE_GRID_SCALE + 128
                    if scene_map[y, x] > 0:
                        return True # inside the floor;
                    else:
                        return False
                
                tmp_data = data_dict[new_name_list[0]]

                # ! does not work on the cluster.
                for grid_orientation in np.arange(0, 360, 60): # 6
                    grid_orientation = np.random.uniform(0, 60) + grid_orientation
                    for radius in np.arange(0, 7, 1): # 7
                        radius = np.random.uniform(0, 1) + radius
                        for position_orientation in np.arange(0, 360, 60): # 6
                            position_orientation = np.random.uniform(0, 60) + position_orientation
                            one_element = copy.deepcopy(tmp_data)
                            # import pdb;pdb.set_trace()
                            # one_element = copy.deepcopy(data_dict[new_name_list[0]].replace('amass_data', 'amass_data_local'))

                            x = radius * np.cos(position_orientation / 180 * np.pi)
                            y = radius * np.sin(position_orientation / 180 * np.pi)

                            # grid_position = np.array([x, y])
                            grid_motion = np.array([x, y, np.cos(grid_orientation / 180 * np.pi), np.sin(grid_orientation / 180 * np.pi)])
                            
                            # import pdb;pdb.set_trace()
                            length = one_element['length']
                            # one_element['motion'][length-1, :4] = grid_motion
                            one_element['motion'][length-15:, :4] = grid_motion[None].repeat(15).reshape((4,15)).T # random shift for the final frames.
                            # one_name = new_name_list[0] + f'_grid_{grid_orientation}_{radius}_{position_orientation}'
                            one_name = f'grid_{grid_orientation}_{radius}_{position_orientation}'
                            data_dict[one_name] = one_element
                            new_name_list.append(one_name)
                            
                            # TODO: filter out those end points inside the object.
                            # 
                            length_list.append(one_element['length'])

                print('add grid search examples: ', len(data_dict))

                print('save to ', data_store_path.replace('.pkl', f'_id{id_frame}_grid_search.pkl'))
                
                if os.path.exists(data_store_path.replace('.pkl', f'_id{id_frame}_grid_search.pkl')):
                    with open(data_store_path.replace('.pkl', f'_id{id_frame}_grid_search.pkl'), 'rb') as f:
                        data = pickle.load(f)
                        data_dict = data['data_dict']
                        new_name_list = data['name_list']
                        mean = data['mean']
                        std = data['std']
                        length_list = data['length_list']

                if True:
                    final_points = []
                    for key, value in data_dict.items():
                        length = value['length']
                        final_points.append(value['motion'][length-1, :2])
                    plt.figure()
                    plt.scatter(np.array(final_points)[:, 0], np.array(final_points)[:, 1], marker='.', color='r')
                    # add orientation visualization.
                    plt.savefig('debug_results/grid_search.png')
                    plt.close()
            
            # import pdb;pdb.set_trace()
            print('total number of data: ', len(data_dict), '/', len(id_list))
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            print('total name list and length list: ', len(name_list), '/', len(length_list))
            print(new_name_list)           
            print('save to ', data_store_path)
            
            with open(data_store_path, 'wb') as f:
                pickle.dump(
                    {'name_list': name_list,
                    'length_list': length_list,
                    'mean': mean,
                    'std': std,
                    'data_dict': data_dict,
                    'useful_list': useful_list,
                    'useless_list': useless_list}, f)

        
        self.mean = mean
        self.std = std
        
        if not self.eval_locomotion:
            Mean, Std, data_motion = self.get_new_mean_std(data_dict, kwargs)
            self.mean = Mean # each data point should be normalized with the same mean and std.
            self.std = Std
            
        else:
            if self.opt.root_representation == 'root_pos_abs_rotcossin_height_only_norm':
                data_motion = []
                for key, value in data_dict.items():
                    motion = value['motion']
                    motion = self.get_motion_from_ori_humanml(motion, min_motion_len, max_motion_length_filter)
                    data_dict[key]['motion'] = motion
                    data_motion.append(motion)
               
                dir_path = f'{code_dir}/dataset/humanml_opt_pos_abs_rotcossin_height_norm_double_indicator'
                self.mean = np.load(os.path.join(dir_path, 'Mean_train.npy'))
                self.std = np.load(os.path.join(dir_path, 'Std_train.npy'))
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        
    def get_new_mean_std(self, data_dict, kwargs, kind=None, return_data_motion=False):
        
        # import pdb;pdb.set_trace()
        if kind == 'interaction':
        
            opt_name = os.path.basename(kwargs['opt_name']).split('.')[0]

            self.load_pretained_mean_std = getattr(self.opt, 'load_pretained_mean_std', False)
            self.load_pretained_path = getattr(self.opt, 'load_pretained_path', None)

            if True: # normalization problem; we need to load the original training model;
                # dir_path = f'{code_dir}/dataset/humanml_opt_pos_abs_rotcossin_height_norm_double_indicator'

                ## 268 is the best; humanml_opt_pos_abs_rotcossin_height_humanml_pose_norm
                # 268 | this is for Trace model. | but this was used for previous train cmdm model before 12.16.
                if ('humanml_pose' in opt_name or 'humanml_sdf' in opt_name) and not self.load_pretained_mean_std:  # * this is dropped out.
                    dir_path = f'{code_dir}/dataset/humanml_opt_pos_abs_rotcossin_height_humanml_pose_norm_action0' 
                else: # 261 # this should worked for controlnet.
                    if self.load_pretained_path is not None:
                        dir_path = self.load_pretained_path
                    else: # pretrained model.
                        dir_path = f'{code_dir}/dataset/humanml_opt_pos_abs_rotcossin_height_humanml_pose_norm_clean_finalpose' # ! this is for pretrained model.
                    
                Mean = np.load(os.path.join(dir_path, 'Mean_train.npy'))
                Std = np.load(os.path.join(dir_path, 'Std_train.npy'))
                print('!!!!!!! \n load mean and std from: ', dir_path, '\n !!!!!!!')

                if 'humanml_sdf' in opt_name: # TODO: calculate the mean and std.
                    # append 88 1 for mean and 0 for std
                    Mean = np.concatenate((Mean, np.zeros((88))))
                    Std = np.concatenate((Std, np.ones((88))))

            # get the data motion.
            if return_data_motion:
                data_motion = []
                for key, value in data_dict.items():
                    data_motion.append(value['motion'])
                # data_motion = np.concatenate(data_motion, axis=0)
                return Mean, Std, data_motion
            return Mean, Std
        
        if self.opt.root_representation == 'root_pos_abs_rotcossin_height_only_norm':
            print(f'recalculating mean and std for {self.opt.root_representation} ***** ')
            
            data_motion = []
            for key, value in data_dict.items():
                # ! recanonicalize the root position
                # frame, value
                data_dict[key]['motion'][:, [0,1]] = data_dict[key]['motion'][:, [0,1]] - data_dict[key]['motion'][0, [0,1]] 
                data_dict[key]['motion'][:, :5]
                
                if self.max_distances is not None: # clean the data by the distance.
                    # get the farest distance distribution. | visualize the end points into a heat map;
                    pass
                    old_length = data_dict[key]['length']
                    
                    distance = np.linalg.norm(data_dict[key]['motion'][:, [0,1]], axis=1)
                    
                    if (distance > self.max_distances).sum() > 0:
                        outside_first_flag = np.nonzero(distance > self.max_distances)[0].min()
                        print(f'outside_distance {self.max_distances} flag:  ', outside_first_flag)
                        data_dict[key]['length'] = outside_first_flag + 1
                        data_dict[key]['motion'][outside_first_flag+1:] *= 0.0
                
                data_motion.append(data_dict[key]['motion'])
            
            Mean = np.concatenate(data_motion).mean(axis=0)
            Std = np.concatenate(data_motion).std(axis=0)
            
        return Mean, Std, data_motion

    def get_motion_from_ori_humanml(self, motion, min_motion_len, max_motion_length_filter, name=None):
        ############################################# get input motion from original motion  #############################################
        
        if (len(motion)) < min_motion_len or (len(motion) >= max_motion_length_filter):
            return None
        
        if hasattr(self.opt, 'root_representation') and (self.opt.root_representation=='root_pos_abs_rotcossin_height_only_norm'):
            r_rot_ang, r_pos = recover_root_rot_pos_abs(torch.from_numpy(motion).cuda().float()) # get root joint rotation and position， here
            r_cos = torch.cos(r_rot_ang)
            r_sin = torch.sin(r_rot_ang)
            
            motion = np.concatenate((r_pos.cpu().numpy()[:, [0, 2]], r_cos[:, None].cpu().numpy(), r_sin[:, None].cpu().numpy(), r_pos.cpu().numpy()[:, [1]]), -1)

            # ! only hack the y dimension
            motion[:, 1] = motion[:, 1] * -1
            angle = np.arctan2(motion[:, 3], motion[:,2])
            motion[:, 2] = np.cos(-angle) # -180
            motion[:, 3] = np.sin(-angle)

        
        elif hasattr(self.opt, 'root_representation') and (self.opt.root_representation=='root_pos_abs_rotcossin_height_pose_with_humanml'): # with local pose as input.
            r_rot_ang, r_pos = recover_root_rot_pos_abs(torch.from_numpy(motion).cuda().float()) # get root joint rotation and position
            r_cos = torch.cos(r_rot_ang)
            r_sin = torch.sin(r_rot_ang)    
            motion = np.concatenate((r_pos.cpu().numpy()[:, [0, 2]], r_cos[:, None].cpu().numpy(), r_sin[:, None].cpu().numpy(), r_pos.cpu().numpy()[:, [1]], motion), -1)

        return motion
        
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        if data.shape[-1] != self.std.shape[-1]: # used for inference, when position_root=True
            new_shape = data.shape[-1]
            return data*self.std[:new_shape]+self.mean[:new_shape]
        else:
            return data * self.std + self.mean

    def inv_transform_th(self, data):
        # import pdb;pdb.set_trace()
        if data.shape[-1] != self.std.shape[-1]: # used for inference, when position_root=True
            new_shape = data.shape[-1]
            return data * torch.from_numpy(self.std[:new_shape]).to(
                data.device) + torch.from_numpy(self.mean[:new_shape]).to(data.device)
        else:
            return data * torch.from_numpy(self.std).to(
                data.device) + torch.from_numpy(self.mean).to(data.device)
        
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        # print(f'idx: {idx} m_length', m_length)

        idx = self.pointer + item
        # print(idx, self.name_list[idx])
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['caption']
        scene_data = data['scene']
        if scene_data.max() == 255: scene_data = scene_data / 255.0
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        
        # import pdb;pdb.set_trace()

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        motion = motion[:m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # zeros inpainting.
        if m_length < self.max_motion_length:
            if hasattr(self.opt, 'motion_padding') and self.opt.motion_padding == 'repeat':
                motion = np.concatenate([motion,
                                        np.repeat(motion[-1:, ...], self.max_motion_length - m_length, axis=0),
                                        ], axis=0)
                m_length = self.max_motion_length # ! train all frames, including padding frames.
                
            else:
                motion = np.concatenate([motion,
                                        np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                        ], axis=0)
        
        if 'kind' in data.keys():
            kind = data['kind']
            if 'ori_humanml_motion' in data.keys():
                ori_humanml_motion = data['ori_humanml_motion']
                if self.extra_rootRotPos_feat:
                    start_goal = motion[[-1]]
                    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [start_goal], scene_data, kind, ori_humanml_motion
                else:
                    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [], scene_data, kind, ori_humanml_motion

            else:
                if self.extra_rootRotPos_feat:
                    start_goal = motion[[-1]]
                    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [start_goal], scene_data, kind
                else:
                    return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [], scene_data, kind
                
        else:
            if self.extra_rootRotPos_feat:
                start_goal = motion[[-1]]
                return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [start_goal], scene_data
            else:
                return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [], scene_data


'''For use of training text motion matching model, and evaluations'''
# this is the dataset for inpaintint
class Text2MotionDatasetV3(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, **kwargs):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.num_frames = num_frames if num_frames else False
        self.max_motion_length = opt.max_motion_length
        if (self.num_frames == False) or type(self.num_frames)==int:
            min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        else:
            min_motion_len = self.num_frames[0]
            self.max_motion_length = self.num_frames[1]
        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        id_list = id_list[:size]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                if self.num_frames != False:
                                    if len(n_motion) >= self.max_motion_length:
                                        bias = random.randint(0, len(n_motion) - self.max_motion_length)
                                        data_dict[new_name] = {'motion': n_motion[bias: bias+self.max_motion_length],
                                                               'length': self.max_motion_length,
                                                               'text': [text_dict]}
                                        length_list.append(self.max_motion_length)

                                    else:
                                        data_dict[new_name] = {'motion': n_motion,
                                                               'length': len(n_motion),
                                                               'text': [text_dict]}
                                        length_list.append(len(n_motion))

                                else:
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    length_list.append(len(n_motion))

                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    if self.num_frames != False:
                        if len(motion) >= self.max_motion_length:
                            bias = random.randint(0, len(motion) - self.max_motion_length)
                            data_dict[name] = {'motion': motion[bias: bias + self.max_motion_length],
                                                   'length': self.max_motion_length,
                                                   'text': [text_dict]}
                            length_list.append(self.max_motion_length)

                        else:
                            data_dict[name] = {'motion': motion,
                                               'length': len(motion),
                                               'text': text_data}
                            length_list.append(len(motion))

                    else:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        length_list.append(len(motion))

                    new_name_list.append(name)
            except Exception as e:
                print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        # import pdb; pdb.set_trace()
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def inv_transform_th(self, data):
        # import pdb;pdb.set_trace()
        if data.shape[-1] != self.std.shape[-1]: # used for inference, when position_root=True
            new_shape = data.shape[-1]
            return data * torch.from_numpy(self.std[:new_shape]).to(
                data.device) + torch.from_numpy(self.mean[:new_shape]).to(data.device)
        else:
            return data * torch.from_numpy(self.std).to(
                data.device) + torch.from_numpy(self.mean).to(data.device)

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        # FIXME: I removed the extra return value ([]) at the end
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []