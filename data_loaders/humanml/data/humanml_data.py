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
from torch.utils.data._utils.collate import default_collate
from data_loaders.amass.sampling import FrameSampler
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml_utils import HML_USE_LOCAL_JOINT_MASK

# to get original joint positions.
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos, recover_root_rot_pos_abs
from data_loaders.humanml.common.quaternion import quaternion_to_cont6d
from data_loaders.amass.babel import BABEL
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2

import matplotlib.pyplot as plt
import pickle

MOTION_TYPES = [
    '_0',
    '_1',
    '_0_with_transition',
    '_1_with_transition',
]

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

'''For use of training text motion matching model, and evaluations'''
# TODO: use this as baseline.
# class Text2MotionDataset_HumanML(data.Dataset):

class Text2MotionDataset_HumanML(Text2MotionDatasetV2):
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

        self.extra_rootRotPos_feat = getattr(self.opt, 'extra_rootRotPos_feat', False)
        
        self.max_distances = getattr(self.opt, 'max_distances', None)
        self.debug = getattr(self.opt, 'debug', 0)
        

        restore = True

        self.is_train = 'train' in split_file
        if self.is_train: 
            print('status: train !!!') 
            data_store_path = kwargs['opt_name'].replace('.txt', '_train.pkl')
        else: 
            print('status test ---')
            data_store_path = kwargs['opt_name'].replace('.txt', '_test.pkl')

        print(f'max_distance: {self.max_distances}')
        
        if self.debug:
            self.debug_dir = os.path.join('./debug', os.path.basename(kwargs['opt_name']).split('.')[0])
            os.makedirs(self.debug_dir, exist_ok=True)
        else:
            self.debug_dir = None
        print('debug: ', self.debug)
        print('debug dir: ', self.debug_dir)

        import pdb;pdb.set_trace()
        if restore and os.path.exists(data_store_path): # save different data.
            print(f'load data !!!! from {data_store_path}')
            input_all_data = pickle.load(open(data_store_path, 'rb'))
            name_list = input_all_data['name_list']
            length_list = input_all_data['length_list']
            mean = input_all_data['mean']
            std = input_all_data['std']
            data_dict = input_all_data['data_dict'] 
            
            print('total name list and length list: ', len(name_list), '/', len(length_list))
        else:
            data_dict = {}
            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())
            id_list = id_list[:size]
    
            new_name_list = []
            length_list = []
            for name in tqdm(id_list):
            # for name in tqdm(id_list[:10]):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))

                    if (len(motion)) < min_motion_len or (len(motion) >= max_motion_length_filter):
                        # print('motion: {} len {}, min_motion_len: {}, len(motion): {}'.format(name, len(motion), min_motion_len, len(motion)))
                        continue
                    
                    motion = self.get_motion_from_ori_humanml(motion, min_motion_len, max_motion_length_filter) # this modify the motion axis; 
                    
                    ### TODO: get the textual description.
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
                            else:   # subset of the motion sequence.
                                # print('f_tag: ', f_tag, 'to_tag: ', to_tag)
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
                                            # bias = 0
                                            if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                                indicator = np.zeros((self.max_motion_length, 1))
                                                indicator[[0, -1], :] = 1.0
                                                n_motion = np.concatenate((n_motion[bias: bias + self.max_motion_length], indicator), -1)
                                            else:
                                                n_motion = n_motion[bias: bias+self.max_motion_length]
                                                
                                            data_dict[new_name] = {'motion': n_motion,
                                                                'length': self.max_motion_length,
                                                                'text': [text_dict]}
                                            length_list.append(self.max_motion_length)

                                        else:
                                            if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                                indicator = np.zeros((n_motion.shape[0], 1))
                                                indicator[[0, -1], :] = 1.0
                                                n_motion = np.concatenate((n_motion, indicator), -1)
                                            
                                            data_dict[new_name] = {'motion': n_motion,
                                                                'length': len(n_motion),
                                                                'text': [text_dict]}
                                            length_list.append(len(n_motion))

                                    else:
                                        if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                            indicator = np.zeros((n_motion.shape[0], 1))
                                            indicator[[0, -1], :] = 1.0
                                            n_motion = np.concatenate((n_motion, indicator), -1)
                                        
                                        data_dict[new_name] = {'motion': n_motion,
                                                                'length': len(n_motion),
                                                            'text':[text_dict]}
                                        length_list.append(len(n_motion))

                                    new_name_list.append(new_name)
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    # if motion.shape[-1] == 263:
                    #     import pdb;pdb.set_trace()

                    # import pdb;pdb.set_trace()
                    if flag:
                        # indicator is in the root_representation.
                        
                        if self.num_frames != False: # extract the max_motion_length;
                            if len(motion) >= self.max_motion_length:
                                bias = random.randint(0, len(motion) - self.max_motion_length)
                                # bias = 0
                                
                                if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                    indicator = np.zeros((self.max_motion_length, 1))
                                    indicator[[0, -1], :] = 1.0
                                    motion = np.concatenate((motion[bias: bias + self.max_motion_length], indicator), -1)
                                else:
                                    motion = motion[bias: bias + self.max_motion_length]
                                
                                data_dict[name] = {'motion': motion,
                                                    'length': self.max_motion_length,
                                                    'text': [text_dict]}
                                length_list.append(self.max_motion_length)
                            else:
                                if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                    indicator = np.zeros((motion.shape[0], 1))
                                    indicator[[0, -1], :] = 1.0
                                    motion = np.concatenate((motion, indicator), -1)
                                
                                data_dict[name] = {'motion': motion,
                                                'length': len(motion),
                                                'text': text_data}
                                length_list.append(len(motion))
                        else:
                            if hasattr(self.opt, 'indicator') and self.opt.indicator: 
                                indicator = np.zeros((motion.shape[0], 1))
                                indicator[[0, -1], :] = 1.0
                                motion = np.concatenate((motion, indicator), -1)
                                
                            data_dict[name] = {'motion': motion,
                                            'length': len(motion),
                                            'text': text_data}
                            length_list.append(len(motion))

                        new_name_list.append(name)

                except Exception as e:
                    print(e)
                    pass

            print('total number of data: ', len(data_dict), '/', len(id_list))
            name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            print('total name list and length list: ', len(name_list), '/', len(length_list))

            if restore:            
                print('save to ', data_store_path)

                with open(data_store_path, 'wb') as f:
                    pickle.dump(
                        {'name_list': name_list,
                        'length_list': length_list,
                        'mean': mean,
                        'std': std,
                        'data_dict': data_dict
                        }
                , f)
                    
            
        self.mean = mean # this is load from precalculated mean and std;
        self.std = std
        self.get_new_mean_std(data_dict, kwargs)

        if restore:
            debug_save_dir = data_store_path[:-4]
            self.debug_save_dir = debug_save_dir
            print(debug_save_dir)
            if not os.path.exists(debug_save_dir) and False:
                self.draw_data_distribution(data_motion, debug_save_dir)
            # import pdb;pdb.set_trace()
        else:
            self.debug_save_dir = None
            
        # import pdb;pdb.set_trace()
        #             
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def get_new_mean_std(self, data_dict, kwargs, return_data_motion=False):
        if hasattr(self.opt, 'root_representation') and \
            self.opt.root_representation in ['root_pos_abs_only_norm', 'root_pos_abs_rotcossin_height_only_norm', \
                'root_pos_abs_rotcossin_only_norm', 'root_pos_abs_rotcossin_height_pose_only_norm', 'root_pos_abs_rotcossin_height_pose_with_humanml']:
            
            # import pdb;pdb.set_trace()
            print(f'recalculating mean and std for {self.opt.root_representation} ***** ')
            
            data_motion = []
            for key, value in data_dict.items():
                # recanonicalize the root position
                # frame, value
                # zeros as initialization;
                data_dict[key]['motion'][:, [0,1]] = data_dict[key]['motion'][:, [0,1]] - data_dict[key]['motion'][0, [0,1]] 
                
                # import pdb;pdb.set_trace()
                if self.opt.root_representation in ['root_pos_abs_rotcossin_height_only_norm', 'root_pos_abs_rotcossin_only_norm', \
                    'root_pos_abs_rotcossin_height_pose_only_norm', 'root_pos_abs_rotcossin_height_pose_with_humanml']: # canonicalize the global trajectory.

                    # renormalize the motion: make the start as zeros; ||| this is wrong.
                    cos_theta = data_dict[key]['motion'][:, 2]
                    sin_theta = data_dict[key]['motion'][:, 3]
                    rot_mat = np.zeros((len(cos_theta), 2, 2))
                    rot_mat[:, 0, 0] = cos_theta
                    rot_mat[:, 0, 1] = sin_theta
                    rot_mat[:, 1, 0] = -sin_theta
                    rot_mat[:, 1, 1] = cos_theta
                    motion_before = data_dict[key]['motion'][:, [0,1]].copy()
                    

                    data_dict[key]['motion'][:, [0,1]] =  np.matmul(np.linalg.inv(rot_mat[0, ]), motion_before.T).T
                    if data_dict[key]['motion'][0, [0,1]].sum() != 0.0:
                        # import pdb;pdb.set_trace()
                        pass

                    rot_mat[:] = np.matmul(np.linalg.inv(rot_mat[0,]), rot_mat[:])
                    data_dict[key]['motion'][:, 2] = rot_mat[:, 0, 0]
                    data_dict[key]['motion'][:, 3] = rot_mat[:, 0, 1]
                
                if self.max_distances is not None: # clean the data by the distance.
                    # get the farest distance distribution. | visualize the end points into a heat map;
                    pass
                    old_length = data_dict[key]['length']
                    
                    distance = np.linalg.norm(data_dict[key]['motion'][:, [0,1]], axis=1)
                    
                    if (distance > self.max_distances).sum() > 0:
                        outside_first_flag = np.nonzero(distance > self.max_distances)[0].min()
                        print('outside_first_flag: ', outside_first_flag)
                        data_dict[key]['length'] = outside_first_flag + 1
                        data_dict[key]['motion'][outside_first_flag+1:] *= 0.0
                
                data_motion.append(data_dict[key]['motion'])
            
            Mean = np.concatenate(data_motion).mean(axis=0)
            Std = np.concatenate(data_motion).std(axis=0)
            
            # Training split: calculate and save;
            data_save_dir = kwargs['opt_name'].split('.')[0]

            import pdb;pdb.set_trace()
            if self.is_train:                
                os.makedirs(data_save_dir, exist_ok=True)
                print(f'save mean and std to {data_save_dir}')
                np.save(os.path.join(data_save_dir, 'Mean_train.npy'), Mean)
                np.save(os.path.join(data_save_dir, 'Std_train.npy'), Std)
            else:
                print('load mean and std from training split')
                assert os.path.exists(data_save_dir), f'{data_save_dir} does not exist'
                Mean = np.load(os.path.join(data_save_dir, 'Mean_train.npy'))
                Std = np.load(os.path.join(data_save_dir, 'Std_train.npy'))
                
            # eval/val split: load training mean and std;
            # ! we do not predict indicator; this is only used for input.

            if self.opt.root_representation == 'root_pos_abs_rotcossin_height_pose_with_humanml':
                self.mean = np.concatenate((Mean[:5], self.mean))
                self.std = np.concatenate((Std[:5], self.std))
                print(self.mean.shape, self.std.shape)
                
            else:
                self.mean = Mean
                self.std = Std

            if return_data_motion:
                return data_motion

    def __getitem__(self, item):
        # print(f'idx: {idx} m_length', m_length)

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
        if False:
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
        else:
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
            # print(self.opt.unit_length, len(motion), m_length)
            # make sure every start frame is the same. 
            # motion[m_length-1:] = motion[-1:][None].repeat(len(motion)-m_length-1, 1) 
            motion = motion[:m_length] # always give the end indicator.

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
        
        if self.extra_rootRotPos_feat:
            start_goal = motion[[-1]]
            # import pdb;pdb.set_trace()
            # print(word_embeddings.shape, motion.shape)
            # print(tokens)
            # FIXME: I removed the extra return value ([]) at the end
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), [start_goal]
        else:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), []