import os.path as osp
from os.path import join as pjoin

import numpy as np
from torch.utils import data

from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2, Text2MotionDatasetV3
from data_loaders.humanml.data.samp_data import Text2MotionDataset_SAMP
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, load_mode, datapath='./dataset/humanml_opt.txt', split="train", keep_last_step=False, **kwargs):
        self.load_mode = load_mode
        
        print('load opt file ', datapath)
        self.batch_size = kwargs.get('batch_size')

        self.dataset_name = 't2m'
        self.dataname = 't2m'
        self.split = split
        self.keep_last_step = keep_last_step
        
        
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        opt.load_mode = load_mode
        self.opt = opt

        # import pdb;pdb.set_trace()
        
        print('---------------- HumanML3D ----------------')
        print(f'load data: load_mode: {load_mode} meta_dir:{opt.meta_dir}, data_root:{opt.data_root}')
        print('---------------- info ----------------')
        
        # ! no_mirror data is used during training and testing.
        split_postfix = opt.split_postfix if hasattr(opt, 'split_postfix') else ''
        
        print('Loading dataset %s ...' % opt.dataset_name)
        print('Split postfix %s ...' % split_postfix)

        
        ### at the begining to calculate the normalization code;
        if load_mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            
        elif load_mode in ['train', 'eval', 'text_only', 'prefix', 'text']: 
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if load_mode == 'eval': # ! this is important for evaluation.
            # used by T2M models (including evaluators)
            # this is to translate ours to their norms for calculating the embedding.
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}{split_postfix}.txt')
        
        print(f'load split text: {self.split_file}')
        print('load mode: ', load_mode)

        if load_mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file, **kwargs)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            if hasattr(opt, 'dataset_kind') and 'SAMP' in opt.dataset_kind: 
                kwargs['opt_name'] = datapath
                self.t2m_dataset = Text2MotionDataset_SAMP(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)
            elif hasattr(opt, 'dataset_kind') and 'inpainting' in opt.dataset_kind:
                self.t2m_dataset = Text2MotionDatasetV3(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)
            else:
                self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, **kwargs)
            self.num_actions = 1 # dummy placeholder

        print('dataloader kind: ', self.t2m_dataset)
        

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()