import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import itertools
import os.path as osp
import copy

from torch.utils.data._utils.collate import default_collate
from data_loaders.amass.sampling import FrameSampler

from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml_utils import HML_USE_LOCAL_JOINT_MASK

# to get original joint positions.
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
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_root_rot_pos_abs
from scipy.spatial.transform import Rotation as R
import trimesh
from trimesh import transform_points
from pyquaternion import Quaternion as Q

from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *
from data_loaders.amass.tools_teach.easyconvert import to_matrix, matrix_to
from data_loaders.humanml.scripts.motion_process import face_joint_indx,uniform_skeleton,tgt_offsets
from data_loaders.humanml.utils.get_scene_bps import ObjectSceneBPS, load_bps_set, SAMPLE_POINTS_NUM # this is used to get BPS
from data_loaders.humanml.utils.plot_script import explicit_plot_3d_image_SDF
from data_loaders.humanml.utils.utils import swap_left_right
# TODO: build a more efficient data load for SAMP: merge all data preprocess into one file. 
from data_loaders.humanml.utils.utils_samp import get_contact_label, get_interaction_text_description, \
    canonicalize_motion_and_scene_new, add_entire_motion, canonicalize_poses_to_object_space, \
    save_sample_poses, visualize_interactive_scene, add_subset_motion, \
    Myclass, load_motion, get_id_list, get_restore_data, \
    load_mannual_text_dict

from data_loaders.humanml.utils.utils_samp import get_contact_label_from_dict, load_mannual_dict, get_data_augmentaion_on_motion_mannual_label
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2
from model.scene_condition import query_feature_grid_3D

from data_loaders.humanml.utils.utils_samp import get_current_time, get_test_obj_name
from scipy.spatial.transform import Rotation

code_dir = os.path.dirname(__file__) + '/../../..'


'''For use of training text motion matching model, and evaluations'''
class Text2MotionDataset_SAMP(Text2MotionDatasetV2):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, num_frames, size=None, keep_last_step=False, **kwargs):

        # put the parse parameters into kwargs: needs things to do.
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
        
        self.max_distances = getattr(self.opt, 'max_distances', 1.5) # this is needed for SDF volume. 
        print(f'max_distance: {self.max_distances}')
        # from xy plane into xy plane.
        trans_matrix = np.array([[1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]])
        self.extra_rootRotPos_feat = getattr(self.opt, 'extra_rootRotPos_feat', False)

        print('-------------- data load setting --------------\n')
        self.is_train = 'train' in split_file
        if self.is_train: 
            print('status: train !!!') 
        else: 
            print('status test ---')

        self.debug = getattr(self.opt, 'debug', 0)
        if self.debug:
            self.debug_dir = os.path.join('./debug', os.path.basename(kwargs['opt_name']).split('.')[0])
            os.makedirs(self.debug_dir, exist_ok=True)
        else:
            self.debug_dir = None
        print('debug: ', self.debug)
        print('debug dir: ', self.debug_dir)
             
        ######### get the processed data list.
        restore = getattr(self.opt, 'restore', True)
        restore_npz = False # the each npz file
        print('restore: ', restore)
        print('restore_npz: ', restore_npz)

        print('-------------- grid search robustness setting --------------\n')
        # this is for sanity check different pos and ori
        self.grid_search_pos_ori = getattr(self.opt, 'grid_search_pos_ori', False) 
        if self.grid_search_pos_ori:
            restore=False # the global pkl
            size = 1
            id_frame = int(getattr(self.opt, 'id_frame', 0))
        else:
            id_frame = None
        print('grid_search_pos_ori: ', self.grid_search_pos_ori)
        print('id_frame: ', id_frame)
        print('size : ', size)
        

        ######## get specific data information.
        ROOT_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ROOT_DATA_DIR_LOCAL=f'{ROOT_CODE_DIR}/dataset/SAMP_interaction'
        obj_data_dir = f'{ROOT_DATA_DIR_LOCAL}/SAMP_objects'
        self.obj_data_dir = obj_data_dir
        male_bm_path = os.path.join(os.path.dirname(__file__), '../../../body_models/smpl_models/smplh/SMPLH_NEUTRAL.npz')
        ROOT_DATA_DIR= f'{ROOT_CODE_DIR}/dataset/SAMP_interaction/'
        body_data_dir = f'{ROOT_DATA_DIR}/body_meshes_smpl'
        
        self.load_filter_obj = getattr(self.opt, 'load_filter_obj', False)
        if self.load_filter_obj:
            obj_transformation_pkl = f'{obj_data_dir}/chair_all_filter.pkl'
        else:
            obj_transformation_pkl = f'{obj_data_dir}/chair_all.pkl'
        
        with open(obj_transformation_pkl, 'rb') as f:
            obj_transformation = pickle.load(f)
            self.obj_transformation = {}
            
            for one in obj_transformation:
                # one['src_file']
                # ! change the abspath for src and tgt;
                src_name = one['src_file']
                tgt_name = one['tgt_file']
                
                src_name = src_name.replace('/ps/project/scene_generation', f'{ROOT_CODE_DIR}/dataset')
                tgt_name = tgt_name.replace('/home/hyi/data/dataset/humanise_bundle/scene_aware_motion_gen/dataset', f'{ROOT_DATA_DIR}')
                tgt_name = tgt_name.replace('/is/cluster/work/hyi/workspace/summon/data/samp/summon', f'{ROOT_CODE_DIR}/dataset/samp_summon')

                one['src_file'] = src_name
                one['tgt_file'] = tgt_name
                
                # if not os.path.exists(os.path.dirname(tgt_name)): print(tgt_name)
                self.obj_transformation[tgt_name] = one

        ### todo: change the code to run on the new objects.
        
        print('-------------- action & obj training setting --------------\n')

        # -1: all five kinds of motion in training. 
        self.action_label = getattr(self.opt, 'action_label', -1) 
        self.aug_num = getattr(self.opt, 'aug_num', 20) 
        print('action label: ', self.action_label)
        print('augmentation number: ', self.aug_num)

        data_dir = obj_data_dir
        if self.is_train:
            restore_data_dir = f'{ROOT_DATA_DIR_LOCAL}/train_pairs'
        else:
            restore_data_dir = f'{ROOT_DATA_DIR_LOCAL}/test_pairs'
        os.makedirs(restore_data_dir, exist_ok=True)

        split_postfix = getattr(self.opt, 'split_postfix', '')
        self.load_all_obj = getattr(self.opt, 'load_all_obj', False) # set as 1
        self.load_bps = getattr(self.opt, 'load_bps', False) # set as 1
        if self.load_bps:
            self.bps_set = load_bps_set()
            restore_npz = False
        
        self.return_distance_vector = getattr(self.opt, 'return_distance_vector', False) # set as 1
        print(f'return_distance_vector: {self.return_distance_vector}')
        print(f'load bps: {self.load_bps} *********** \n')

        print('-------------- data agumentation setting --------------\n')
        self.random_translation_rotation = getattr(self.opt, 'random_translation_rotation', False) # this only works in test phase.
        print('random_translation_rotation: ', self.random_translation_rotation)
        ### use comprehensive text.
        self.use_comprehensive_text = getattr(self.opt, 'use_comprehensive_text', False)
        
        self.mannual_text_dict, self.mannual_walk_to_sit_text_dict, self.mannual_stand_to_sit_text_dict, self.stand_sit_stand_text_dict, self.walk_sit_stand_text_dict= load_mannual_text_dict()

        self.use_flip_object = getattr(self.opt, 'use_flip_object', False)
        print('use_flip_object: ', self.use_flip_object)
        print('-------------- finished ****  data agumentation setting --------------\n')

        data_dict = {}
        id_list = get_id_list(data_dir, is_train=self.is_train, post_fix=split_postfix)
        # train_split = os.path.join(data_dir, f'train{split_postfix}.txt')
        print('list length: ', len(id_list))
        
        print('-------------- restore previous input pkl and sample id for guidance inference --------------\n')        
        data_store_path = getattr(self.opt, 'data_store_path', None)
        sample_id = getattr(self.opt, 'sample_id', None)
        print('data_store_path: ', data_store_path)
        print('sample_id : ', sample_id)
        self.restore_size = getattr(self.opt, 'restore_size', 0)
        if data_store_path is not None:
            print('load from ', data_store_path)
        else:
            if self.is_train:
                if self.load_bps: 
                    data_store_path = f'{restore_data_dir}/train_{self.action_label}_distance{self.max_distances}_augnum{self.aug_num}_post{split_postfix}_bps.pkl'
                else:
                    data_store_path = f'{restore_data_dir}/train_{self.action_label}_distance{self.max_distances}_augnum{self.aug_num}_post{split_postfix}_sdfPath{self.save_sdf_path}_sdfVolume{self.sdf_extent}.pkl'
            else:
                if self.load_bps: 
                    data_store_path = f'{restore_data_dir}/test_{self.action_label}_distance{self.max_distances}_augnum{self.aug_num}_post{split_postfix}_bps.pkl'
                else:
                    data_store_path = f'{restore_data_dir}/test_{size}_{self.action_label}_distance{self.max_distances}_augnum{self.aug_num}_post{split_postfix}_sdfPath{self.save_sdf_path}_sdfVolume{self.sdf_extent}.pkl'
            data_store_path = data_store_path.replace('.pkl', f'_size{len(id_list)}'+'.pkl') 
        print('store path: ', data_store_path)
        
        obj_bps_dict = {}
        obj_points_dict = {}
        
        if restore and os.path.exists(data_store_path): 
            print(f'load data !!!! from {data_store_path}')
            input_all_data = pickle.load(open(data_store_path, 'rb'))
            new_name_list = input_all_data['name_list']
            length_list = input_all_data['length_list']
            mean = input_all_data['mean']
            std = input_all_data['std']
            data_dict = input_all_data['data_dict']
            if self.load_bps:
                obj_bps_dict = input_all_data['obj_bps_dict'] 
                obj_points_dict = input_all_data['obj_points_dict'] 
            print('total name list and length list: ', len(new_name_list), '/', len(length_list))

            if sample_id is not None:
                new_name_list = new_name_list[sample_id:sample_id+1]
                length_list = length_list[sample_id:sample_id+1]

        else:
            ######### preprocess the data at the beginning.
            new_name_list = []
            length_list = []
            save_interval = 1000
            action_status_dict = load_mannual_dict() # this is get the action frames.
            text_dict = get_interaction_text_description()
                
            for cnt, id_name in enumerate(tqdm(id_list[0:])):    
                print(cnt, id_name) 
                motion_name = id_name
                process_path_name = os.path.join(body_data_dir, f'{id_name}')

                if restore_npz and os.path.exists(os.path.join(restore_data_dir, id_name)): 
                    data_dict, new_name_list, length_list = get_restore_data(os.path.join(restore_data_dir, id_name), data_dict, new_name_list, length_list)
                    continue

                # load motion
                if 'chair_mo00' in process_path_name: 

                    # merge this into start_frame, end_frame loading status.
                    tmp_start_frame_dict = {
                        "chair_mo_stageII": {'start': 170, 'stand_start': 320, 'sit_start': 420, 'sit_end': 690, 'stand_end': 750, 'end': 850},
                        "chair_mo001_stageII": {'start': 100, 'stand_start': 300, 'sit_start': 420, 'sit_end': 580, 'stand_end': 650, 'end': 750},
                        "chair_mo002_stageII": {'start': 100, 'stand_start': 210, 'sit_start': 320, 'sit_end': 580, 'stand_end': 660, 'end': 750},
                        "chair_mo003_stageII": {'start': 200, 'stand_start': 330, 'sit_start': 430, 'sit_end': 630, 'stand_end': 720, 'end': 800},
                        "chair_mo004_stageII": {'start': 120, 'stand_start': 330, 'sit_start': 430, 'sit_end': 600, 'stand_end': 680, 'end': 760},
                        "chair_mo005_stageII": {'start': 210, 'stand_start': 350, 'sit_start': 460, 'sit_end': 660, 'stand_end': 860, 'end': 950},
                        "chair_mo006_stageII": {'start': 100, 'stand_start': 360, 'sit_start': 470, 'sit_end': 1060, 'stand_end': 1100, 'end': 1180},
                        "chair_mo007_stageII": {'start': 120, 'stand_start': 320, 'sit_start': 400, 'sit_end': 720, 'stand_end': 810, 'end': 870},
                        "chair_mo008_stageII": {'start': 140, 'stand_start': 230, 'sit_start': 320, 'sit_end': 650, 'stand_end': 760, 'end': 880},
                        "chair_mo009_stageII": {'start': 110, 'stand_start': 220, 'sit_start': 380, 'sit_end': 560, 'stand_end': 710, 'end': 800},
                    }
                    cfg_fps = 30
                    target_fps = 20
                    start_frame = int(tmp_start_frame_dict[motion_name]['start'] * target_fps / cfg_fps)

                    motion_v, motion_j, body_face = load_motion(process_path_name, start_frame=start_frame, return_all=True, recalculate=False) 
                else:    
                    motion_v, motion_j, body_face = load_motion(process_path_name, return_all=True, recalculate=False) 

                pose_seq_np_ori = motion_j # N, 73, 3
                
                # get object list.
                if id_name in action_status_dict.keys(): # TODO: check whether it will contain all objects or not.
                    obj_name = [one for one in list(self.obj_transformation.keys()) if id_name in one]
                else: 
                    obj_name = [one for one in glob.glob(os.path.join(obj_data_dir, motion_name, '*/*.obj')) if one in list(self.obj_transformation.keys())]
                
                # TODO: get specific interaction objects for the test split.
                if not self.is_train:
                    print('get test obj for', id_name)
                    filter_obj_name = get_test_obj_name(id_name)
                    # obj_name = [one for one in obj_name if filter_obj_name in one]
                    new_obj_name = []
                    for one_obj_name in filter_obj_name:
                        new_obj_name += [one for one in obj_name if one_obj_name in one]
                    obj_name = new_obj_name
                    print('obj name: ', obj_name)

                if len(obj_name) == 0:
                    print('no object found: ', id_name)
                    import pdb;pdb.set_trace()
                    continue

                if not self.load_all_obj:
                    obj_name = obj_name[0:1]
                else:
                    if not self.is_train:
                        obj_name = obj_name[:3] 
                
                # all interact objects.
                obj_name_flip = []
                for one in obj_name:
                    one_trans_dict = self.obj_transformation[one]
                    src_file = one_trans_dict['src_file']
                    
                    if self.use_flip_object:
                        one_trans_dict_flip = copy.deepcopy(one_trans_dict)
                        rotation = Rotation.from_matrix(one_trans_dict_flip['rotation'][:3, :3])
                        angles = rotation.as_euler('xyz')
                        angles[2] *= -1
                        rotation_mat = Rotation.from_euler('xyz', angles).as_matrix()

                        one_trans_dict_flip['rotation'] = np.eye(4)
                        one_trans_dict_flip['rotation'][:3, :3] = rotation_mat
                        one_trans_dict_flip['trans'][0] *= -1
                        # only y-axis rotation is inverse.
                        one_trans_dict_flip['transform'] = one_trans_dict_flip['rotation'].copy()

                        one_trans_dict_flip['transform'][:3, -1] = one_trans_dict_flip['trans']
                        flip_one = one.replace('.obj', '_flip.obj')  
                        obj_name_flip.append(flip_one)
                        self.obj_transformation[flip_one] = one_trans_dict_flip

                    # import pdb;pdb.set_trace()
                    ### TODO: change it to interact with the same file.
                    bps_save_dir = os.path.dirname(one_trans_dict['tgt_file'])
                    os.makedirs(bps_save_dir, exist_ok=True)
                    
                    scene_data_ori = None
                    scene_data_ori_flip = None

                    if self.load_bps:
                        if osp.exists(src_file) == False:
                            raise Exception(f'can not find {src_file}')
                        
                        bps_obj = ObjectSceneBPS(self.bps_set, src_file, trans_dict=one_trans_dict, save_dir=bps_save_dir, obj_category='chair', build=False,
                                                        return_distance_vector=self.return_distance_vector) # TODO: change it into height = 0.0
                        obj_bps_dict[one] = bps_obj.bps_feat[0]
                        obj_points_dict[one] = bps_obj.sample_points
                        # import pdb;pdb.set_trace()
                        # ! this is important, to change the translation to move the object zero align.
                        one_trans_dict['trans'] += one_trans_dict['rotation'][:3, :3]@bps_obj.center_transl # center_align: change the original object translation.
                        # print((self.obj_transformation[one]['trans']-one_trans_dict['trans']).sum())

                        if self.use_flip_object:
                            # zero flip.
                            bps_obj_flip = ObjectSceneBPS(self.bps_set, src_file, trans_dict=one_trans_dict_flip, save_dir=bps_save_dir, obj_category='chair', build=False, left_right_flip=True, 
                                                        return_distance_vector=self.return_distance_vector) 
                            obj_bps_dict[flip_one] = bps_obj.bps_feat[0] # 1000,3
                            obj_points_dict[flip_one] = bps_obj.sample_points
                            one_trans_dict_flip['trans'] += one_trans_dict_flip['rotation'][:3, :3]@bps_obj_flip.center_transl # center_align: change the original object translation.

                # get action label.
                if id_name in action_status_dict.keys():
                    action_status = action_status_dict[id_name]
                    interaction_still_idx = int((action_status['sit_start'] + action_status['sit_end']) / 2)
                    available_list = get_contact_label_from_dict(motion_j, interaction_still_idx, max_distance=self.max_distances)
                else:
                    sit_label_list, inside_object_list, available_list = get_contact_label(motion_j, obj_name, max_distance=self.max_distances)
                    
                #### specific data augmentation; 
                if self.use_flip_object:
                    # TODO: do the mirror data augmentation.
                    pose_seq_np_ori_flip = swap_left_right(pose_seq_np_ori.copy())
                    pose_seq_np_ori_flip[..., 0] = pose_seq_np_ori_flip[..., 0]

                    pose_list = [pose_seq_np_ori, pose_seq_np_ori_flip]
                    scene_list = [scene_data_ori, scene_data_ori_flip] 
                    bps_list = [bps_obj, bps_obj_flip]
                    obj_name_list = [obj_name, obj_name_flip]
                else:
                    pose_list = [pose_seq_np_ori]
                    scene_list = [scene_data_ori]
                    bps_list = [bps_obj]
                    obj_name_list = [obj_name]

                # get action kind
                kind = 'sit'
                if 'lie' in id_name:
                    kind = 'lie'

                for tmp_i, (scene_data, pose_seq_np, one_bps, one_obj) in enumerate(zip(scene_list, pose_list, bps_list, obj_name_list)):
                
                    # test phase
                    if not self.is_train and tmp_i == 1: 
                        continue
                        
                    # get the subset of motion clip and text description.
                    assert id_name in action_status_dict.keys()
                    if self.use_comprehensive_text and id_name in self.mannual_walk_to_sit_text_dict.keys():
                        walk_to_sit_sample = [self.mannual_walk_to_sit_text_dict[id_name]]
                        stand_to_sit_sample = [self.mannual_stand_to_sit_text_dict[id_name]]
                        stand_to_stand_sample = [self.stand_sit_stand_text_dict[id_name]]
                        walk_to_stand_sample = [self.walk_sit_stand_text_dict[id_name]]
                        tmp_comprehensive_text = True
                    else:
                        walk_to_sit_sample = None
                        stand_to_sit_sample = None
                        stand_to_stand_sample = None
                        walk_to_stand_sample = None
                        tmp_comprehensive_text = False
                    
                    if self.is_train:
                        tmp_aug_num = self.aug_num
                    else:
                        tmp_aug_num = int(self.aug_num / 10) # evaluate 10 times less.
                        # tmp_aug_num = int(self.aug_num / 4) # each kind 25

                    new_start_f_list, new_end_f_list, new_text_list = get_data_augmentaion_on_motion_mannual_label(action_status_dict[id_name], available_list,
                            text_dict, aug_num=tmp_aug_num, kind=kind, min_motion_len=40, max_motion_len=196, action_type=self.action_label, 
                            use_comprehensive_text=tmp_comprehensive_text, stand_to_sit=stand_to_sit_sample, walk_to_sit=walk_to_sit_sample, 
                            walk_to_stand=walk_to_stand_sample, stand_to_stand=stand_to_stand_sample)

                    if tmp_i == 1: # flip
                        def replace_left_right(text):
                            placeholder = "temp"  # Ensure this placeholder does not exist in the original text

                            # Replace "left" with "temp"
                            text = text.replace("left", placeholder)

                            # Replace "right" with "left"
                            text = text.replace("right", "left")

                            # Replace "temp" (originally "left") with "right"
                            text = text.replace(placeholder, "right")

                            return text
                        new_text_list = [replace_left_right(one_text) for one_text in new_text_list]

                    vis_idx = 0
                    for new_start_f, new_end_f, new_text in tqdm(zip(new_start_f_list, new_end_f_list, new_text_list), total=len(new_start_f_list)):
                        # print(new_start_f, new_end_f)
                        pose_seq_np_sub = pose_seq_np.copy()[new_start_f:new_end_f] # frames, njoints, 3                        
                        # save_sample_poses(pose_seq_np_sub, self.debug_dir, post_fix='org')
                        # ! add translation for the start poses.

                        # ! this function does not have inverse facial direction: walk forwards -> human walk backwards.
                        motion, _, transform_mat = canonicalize_motion_and_scene_new(pose_seq_np_sub, None, return_transform=True)
                        ## post-process to get the global representation of 3D joints.
                        motion = self.get_motion_from_ori_humanml(motion, min_motion_len, max_motion_length_filter, id_name)
                        
                        if motion is None:
                            print('motion error !!!')
                            continue
                        
                        ############################################# merge text, motion into training dict  #############################################
                        flag = False
                        
                        name = f'{motion_name}_{new_start_f}_{new_end_f}_{tmp_i}'
                        if name in new_name_list:
                            print('save setting in ', name)
                            continue
                        add_entire_motion(data_dict, length_list, new_name_list, name, \
                            motion, one_obj, transform_mat, new_text, opt=self.opt, restore_dir=restore_data_dir, restore=restore_npz)
                        
                        data_dict[name]['object_mesh'] = one_bps.mesh
                        

            print('human object interaction !!! number of data: ', len(data_dict.keys()), '/', len(length_list))
            
            print('--------- add humanml dataset into training. --------------\n')
            self.extra_humanml_list = getattr(self.opt, 'extra_humanml_list', None)
            if self.extra_humanml_list is not None and self.is_train:
                tmp_transform_mat = np.zeros_like(transform_mat)
                tmp_transform_mat[:3, :3] = np.eye(3)
                new_obj_name = None
                
                print('load extra humanml list: ', self.extra_humanml_list)
                new_id_list = []
                with cs.open(os.path.join(code_dir, self.extra_humanml_list), 'r') as f:
                    for line in f.readlines():
                        new_id_list.append(line.strip())
                
                for cnt, id_name in enumerate(tqdm(new_id_list)):
                # for cnt, id_name in enumerate(tqdm(new_id_list[:50])):
                    print(cnt, id_name)
                    name = id_name
                    # get original motion
                    motion = np.load(pjoin(opt.motion_dir, id_name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= max_motion_length_filter):
                        continue
                    
                    motion = self.get_motion_from_ori_humanml(motion, min_motion_len, max_motion_length_filter, id_name)
                    try:
                        text_name = id_name
                        text_data = []
                        with cs.open(pjoin(opt.text_dir, text_name + '.txt')) as f:
                            for line in f.readlines():
                                # line： 'a figure moshes with their arms backwards in a circle.#a/DET figure/NOUN mosh/VERB with/ADP their/DET arm/NOUN backwards/ADV in/ADP a/DET circle/NOUN#0.0#0.0\n'
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
                                    print('f_tag: ', f_tag, 'to_tag: ', to_tag)
                                    try:
                                    # if 1:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        import random
                                        new_name = name+ '_' + random.choice('ABCDEFGHIJKLMNOPQRSTUVW') 
                                        while new_name in data_dict:
                                            new_name = name + '_' + random.choice('ABCDEFGHIJKLMNOPQRSTUVW')

                                        if self.num_frames != False and len(n_motion) >= self.max_motion_length:
                                            add_subset_motion(data_dict, length_list, new_name_list, new_name, \
                                                    n_motion, new_obj_name, tmp_transform_mat, [text_dict], opt=self.opt, restore_dir=restore_data_dir, restore=restore_npz)
                                        else:
                                            print(len(data_dict))
                                            add_entire_motion(data_dict, length_list, new_name_list, new_name, \
                                                motion, new_obj_name, tmp_transform_mat, [text_dict], opt=self.opt, restore_dir=restore_data_dir, restore=restore_npz)
                                    except:
                                    # else:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break

                            if flag:
                                if self.num_frames != False and len(motion) >= self.max_motion_length: # extract the max_motion_length;
                                    add_subset_motion(data_dict, length_list, new_name_list, name, \
                                            motion, new_obj_name, tmp_transform_mat, text_data, opt=self.opt, restore_dir=restore_data_dir, restore=restore_npz)
                                else:
                                    add_entire_motion(data_dict, length_list, new_name_list, name, \
                                            motion, new_obj_name, tmp_transform_mat, text_data, opt=self.opt, restore_dir=restore_data_dir, restore=restore_npz)
                    except Exception as e:
                        print(e)
                            
            if self.grid_search_pos_ori:
                data_dict, new_name_list, length_list = self.grid_search(data_dict, new_name_list, length_list)
            
            print('total number of data: ', len(data_dict.keys()), '/', len(id_list))
            new_name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])) # this is used for sorted the length.

            print('total name list and length list: ', len(new_name_list), '/', len(length_list), 'set list: ', len(set(new_name_list)))

            if restore:            
                print('save to ', data_store_path)
                with open(data_store_path, 'wb') as f:
                    pickle.dump(
                        {'name_list': new_name_list,
                        'length_list': length_list,
                        'mean': mean,
                        'std': std,
                        'data_dict': data_dict,
                        'obj_bps_dict': obj_bps_dict,
                        'obj_points_dict': obj_points_dict
                        }
                , f)
        
        # get mean and std.
        Mean, Std, data_motion = self.get_new_mean_std(data_dict, kwargs, return_data_motion=True)

        ### each data points; normalize with the same mean and std;
        # eval/val split: load training mean and std;
        self.mean = Mean # each data point should be normalized with the same mean and std.
        self.std = Std    

        # draw data distribution;
        if restore:
            debug_save_dir = data_store_path[:-4]
            self.debug_save_dir = debug_save_dir
            print(debug_save_dir)
            if not os.path.exists(debug_save_dir) and False:
                self.draw_data_distribution(data_motion, debug_save_dir)
        else:
            self.debug_save_dir = None
        
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

        self.obj_bps_dict = obj_bps_dict # FIXME: load pkl, the bps dict tends to be tensor. 
        self.obj_points_dict = obj_points_dict

        for one, value in self.obj_bps_dict.items():
            if isinstance(value, torch.Tensor):
                self.obj_bps_dict[one] = value.cpu().numpy()
        
        # import pdb;pdb.set_trace()

    def get_motion_from_ori_humanml(self, motion, min_motion_len, max_motion_length_filter, name=None): 
    
        ############################################# get input motion from original motion  #############################################
        if hasattr(self.opt, 'root_representation') and (self.opt.root_representation=='root_pos_abs_rotcossin_height_pose_with_humanml'): # with local pose as input.
            r_rot_ang, r_pos = recover_root_rot_pos_abs(torch.from_numpy(motion).cuda().float()) # get root joint rotation and position
            r_cos = torch.cos(r_rot_ang)
            r_sin = torch.sin(r_rot_ang)    
            motion = np.concatenate((r_pos.cpu().numpy()[:, [0, 2]], r_cos[:, None].cpu().numpy(), r_sin[:, None].cpu().numpy(), r_pos.cpu().numpy()[:, [1]], motion), -1)
        else:
            raise Exception(f'not support other representation!')

        return motion

    def grid_search(self, data_dict, name_list, length_list, 
                        data_store_path=None, mean=None, std=None, id_frame=0):
        # ! random sample the start positions from arbitrary poses;
        
        new_data_dict = {}
        new_name_list = []
        new_length_list = []
        
        # ! does not work on the cluster.
        for grid_orientation in np.arange(0, 360, 60): # 6
            # grid_orientation = np.random.uniform(0, 60) + grid_orientation
            
            for radius in np.arange(0.8, 1.5, 0.2): # 6
                radius = np.random.uniform(0, 0.3) + radius

                for position_orientation in np.arange(0, 360, 60): # 6
                    position_orientation = np.random.uniform(0, 60) + position_orientation
                    one_element = copy.deepcopy(data_dict[name_list[0]])

                    x = radius * np.cos(position_orientation / 180 * np.pi)
                    y = radius * np.sin(position_orientation / 180 * np.pi)

                    grid_motion = np.array([x, y, np.cos(grid_orientation / 180 * np.pi), np.sin(grid_orientation / 180 * np.pi)])
                    
                    length = one_element['length'] # will be changed in the getitem function.
                    print('ori_length: ', length)

                    tmp_length = 160
                    if length < tmp_length:
                        tmp_motion = np.zeros((tmp_length, 268))
                        tmp_motion[:length, :] = one_element['motion'][:, :]
                        one_element['motion'] = tmp_motion
                    else:
                        one_element['motion'] = one_element['motion'][:length] # cut the motion.

                    length = tmp_length # 8 seconds.
                    one_element['length'] = length

                    one_element['motion'][length-1-10:, :4] = grid_motion
                    one_name = name_list[0] + f'_grid_{grid_orientation:.2f}_{radius:.2f}_{position_orientation:.2f}'
                    new_data_dict[one_name] = one_element
                    new_name_list.append(one_name)
                    new_length_list.append(length)

        print('add grid search examples: ', len(new_data_dict))
        
        if data_store_path is not None:
            print('save to ', data_store_path.replace('.pkl', f'_id{id_frame}_grid_search.pkl'))
            with open(data_store_path.replace('.pkl', f'_id{id_frame}_grid_search.pkl'), 'wb') as f:
                pickle.dump(
                    {'name_list': new_name_list,
                    'length_list': new_length_list,
                    'mean': mean,
                    'std': std,
                    'data_dict': new_data_dict}
            , f)

        return new_data_dict, new_name_list, new_length_list
        

    def get_new_mean_std(self, data_dict, kwargs, return_data_motion=False): # load pretrained mean and std in HUMANML3D.
        self.load_pretained_mean_std = getattr(self.opt, 'load_pretained_mean_std', False)
        self.load_pretained_path = getattr(self.opt, 'load_pretained_path', None)

        dir_path = f'{code_dir}/dataset/humanml_opt_pos_abs_rotcossin_height_humanml_pose_norm_clean_finalpose'
        Mean = np.load(os.path.join(dir_path, 'Mean_train.npy'))
        Std = np.load(os.path.join(dir_path, 'Std_train.npy'))
        print('!!!!!!! \n load mean and std from: ', dir_path, '\n !!!!!!!')
        
        # get the data motion.
        if return_data_motion:
            data_motion = []
            for key, value in data_dict.items():
                data_motion.append(value['motion'])
            # data_motion = np.concatenate(data_motion, axis=0)
            return Mean, Std, data_motion
        
        return Mean, Std

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        scene_data = data['scene'] # obj name;
        
        if not self.is_train: 
            object_mesh = data['object_mesh']
        else:
            object_mesh = None

        if scene_data is not None: # no scenes
            if self.load_all_obj or len(scene_data) == 1:
                obj_id = random.choice(scene_data)
            else:
                obj_id = scene_data[0]
        
        # import pdb;pdb.set_trace() 

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
        m_length = (m_length // self.opt.unit_length) * self.opt.unit_length 
        motion = motion[:m_length] # always give the end indicator.
        
        transform_mat = data['tranform_mat']
        if scene_data is not None:
            obj_transform_mat = self.obj_transformation[obj_id]
            if self.random_translation_rotation and not self.is_train:
                rand_x, rand_y = np.random.uniform(-0.3, 0.3, 2) 
                motion[-1, 0] += rand_x 
                motion[-1, 1] += rand_y

                print('obj aug trans: ', rand_x, rand_y)
                # change the final corresponding object position;
                obj_transform_mat['trans'][0] -= rand_y # opsite x-axis
                obj_transform_mat['trans'][1] -= rand_x
            
            if self.load_bps:
                bps_data = self.obj_bps_dict[obj_id]
                obj_point_data = self.obj_points_dict[obj_id]
        else:
            obj_id = None
            obj_transform_mat = {
                'trans': np.zeros(3),
                'scale': np.eye(4),
                'rotation': np.eye(4)
            }

            if self.return_distance_vector:
                bps_data = np.zeros((1000, 3))    
            else:
                bps_data = np.zeros(1000)

            obj_point_data = np.zeros((SAMPLE_POINTS_NUM, 3))

        "Z Normalization during training."
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
        
        # import pdb;pdb.set_trace()

        # print(bps_data.shape, obj_point_data.shape)
        # return the object.
        if self.load_bps: 
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), \
                        [object_mesh, transform_mat, obj_transform_mat['trans'], obj_transform_mat['rotation'], obj_id, obj_transform_mat['scale']], bps_data, [obj_point_data]
        else:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []


