import numpy as np
import torch
import torch.nn as nn
import clip
# https://github.com/GuyTevet/motion-diffusion-model/blob/main/model/rotation2xyz.py
from model.rotation2xyz import Rotation2xyz
from .c_transformer import *
from model.mdm import MDM
from model.tools import OutputProcess, InputProcess, EmbedAction
from model.trace_helpers import MapEncoder, query_feature_grid, visualize_feature_grid
from model.scene_condition import ObjectEncoder, query_feature_grid_3D
from model.base_models import ResNetUNet, convrelu
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.utils_samp import canonicalize_poses_to_object_space_th

from data_loaders.humanml.utils.get_scene_bps import get_one_directional_bps_feature, load_bps_set
import numpy as np

torch.autograd.set_detect_anomaly(True)
from model.mdm import MDM

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module



class CMDM(MDM):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, inv_transform_th=None, **kargs):
        super(CMDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, inv_transform_th, **kargs)

        ## binary indicator.
        father_args = vars(kargs['args'])
        self.double_indicator = father_args.get('indicator_double', False)
        
        self.root_representation = father_args.get('root_representation') # ! this should definitely have.
        print(f'root_representation in mdm: {self.root_representation}')
        
        indicator_emb_dim = 0
        if self.double_indicator:
            print("Using indicator to denote which dimension of the input needs to be inpainted!")
            indicator_emb_dim = self.input_feats

        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        self.input_process = InputProcess(self.data_rep, (self.input_feats + indicator_emb_dim) if self.double_indicator else self.input_feats, self.latent_dim)
        self.cfg_startEnd = kargs.get('cfg_startEnd', False) # only used for training.
        

        self.cfg_noObj = kargs.get('cfg_noObj', False)
        self.cond_mask_prob_noObj = kargs.get('cond_mask_prob_noObj', 0.)
        print(f'cfg on object: {self.cfg_noObj}; scale: {self.cond_mask_prob_noObj}')
        

        ### inpainting with specific dimension.
        self.only_mask_dim = father_args.get('only_mask_dim', [])
        print(f'input with these dimension: {self.only_mask_dim}')        
        
        self.no_start_point_rot_height = father_args.get('no_start_point_rot_height', False)
        print(f'no_start_point_rot_height: {self.no_start_point_rot_height}')        
                        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers, 
                                                return_intermediate=False)

        if 'finalPosPelvis' in self.cond_mode: # this is already in the pretrained model.
            print(f'[WARNING] Using finalPosPelvis as condition!')
            self.final_pos_encoder = nn.Sequential(
                nn.Linear(5, self.latent_dim),
            )

        if self.input_feats <= 268: 
            # * in original model, it does not have in the branch.
            # *  this is old version trained for 2D scene grid.
            self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        else:
            import pdb;pdb.set_trace()
            self.c_input_process = InputProcess(self.data_rep, (self.input_feats + indicator_emb_dim) if self.double_indicator else self.input_feats, self.latent_dim)

        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers,
                                                    return_intermediate=True)

        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
        self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond': # this is extra text condition;
            if 'text' in self.cond_mode:
                self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            if 'action' in self.cond_mode:
                self.c_embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

            if 'sceneGrid' in self.cond_mode:
                print(f'[WARNING] Using sceneGrid as condition!')
                if 'unet' in self.cond_mode:
                    self.map_encoder = ResNetUNet(1, 512)
                else:
                    map_res=256
                    map_encoder_model_arch = 'resnet18'
                    self.map_encoder = MapEncoder(
                        model_arch=map_encoder_model_arch,
                        input_image_shape=(1, map_res, map_res),
                        global_feature_dim=8,
                        grid_feature_dim=512,
                    )
            
            if 'objectBPS' in self.cond_mode:
                print(f'[WARNING] Using object BPS as condition!') # need non-linear activation function.; Thus have three layers.
                self.bps_set = load_bps_set()

                self.c_merge_bps_info = ObjectEncoder(
                    'bps', # data_rep
                    1000*3, # [bps point xyz, w/o], bps point->object distance, bps point->body joint distance, with bps point->body joint nearest label.
                    self.latent_dim,
                )
            
            if 'finalPosPelvis' in self.cond_mode: # this is already in the pretrained model.
                print(f'[WARNING] Using finalPosPelvis as condition!')
                self.c_final_pos_encoder = nn.Sequential(
                    nn.Linear(5, self.latent_dim),
                )
                
    def open_gradient(self):
        # for p in model.parameters():
    #     p.requires_grad = False
        train_layer_name = [
            # 'input_hint_block',
            'c_input_process',
            'c_sequence_pos_encoder',
            'c_seqTransEncoder',
            'zero_convs',
            'c_embed_timestep',
            ## the below is the scene information.
            'c_embed_text', # ! the diversity is too less. 
            'map_encoder',
            'c_merge_bps_info', 
            'c_final_pos_encoder'
        ]
        for name in train_layer_name:
            # getattr(self, name).train()
            module = getattr(self, name, None)
            if module is None:
                print('module is None:', name)
                continue
            for p in module.parameters():
                p.requires_grad = True

    def freeze_norm_params(self): # freeze bn norm; now is using LayerNorm;
        fixed_layer_name = [
            # 'input_hint_block',
            'clip_model',
            'input_process',
            'sequence_pos_encoder',
            'seqTransEncoder',
            'embed_timestep',
            'output_process',
        ]
        for name in fixed_layer_name:
            getattr(self, name).eval()
         
    def set_train_mode(self):
        # self.train()
        self.freeze_norm_params()

    def get_binary_indicator(self, x, y=None, cond_mode=None, cfg_startEnd=False, force_mask=False, only_mask_dim=None):
        indicator_emb = torch.ones(x.shape).to(x.device)
        
        last_frames = y['lengths']
        if cfg_startEnd and self.training: # only used for training.
            startEnd_mask = torch.bernoulli(torch.ones(x.shape[0], device=x.device) * self.cond_mask_prob_startEnd).view(x.shape[0], 1)  # 1-> use null_cond, 0-> use condition
        else:
            if not force_mask:
                startEnd_mask = torch.ones(x.shape[0], device=x.device).view(x.shape[0], 1)  # 1: denotes need to inpaint; 0: denotes fixed.
            else:
                startEnd_mask = torch.zeros(x.shape[0], device=x.device).view(x.shape[0], 1)            
        
        # import pdb;pdb.set_trace()
        if 'startEnd' in cond_mode: 
            for b_i in range(last_frames.shape[0]):
                if only_mask_dim is not None and len(only_mask_dim) > 0: # only mask defined dim.
                    indicator_emb[b_i, only_mask_dim, :, [0]] = 1.0 - startEnd_mask[b_i]
                    indicator_emb[b_i, only_mask_dim, :, last_frames[b_i]-1:] = 1.0 - startEnd_mask[b_i]
                else:
                    if self.no_start_point_rot_height:
                        indicator_emb[b_i, [0,1], :, [0]] = 1.0 - startEnd_mask[b_i] # start point: only take x,z as input
                    else:
                        indicator_emb[b_i, :, :, [0]] = 1.0 - startEnd_mask[b_i]
                    indicator_emb[b_i, :, :, last_frames[b_i]-1:] = 1.0 - startEnd_mask[b_i]
        elif 'startOnly' in cond_mode:
            for b_i in range(last_frames.shape[0]):
                if only_mask_dim is not None and len(only_mask_dim) > 0: # only mask defined dim.
                    indicator_emb[b_i, only_mask_dim, :, [0]] = 1.0 - startEnd_mask[b_i]
                else:
                    indicator_emb[b_i, :, :, [0]] = 1.0 - startEnd_mask[b_i]
        elif 'startRootEnd' in cond_mode: # start pose, and root end as end condition;
            for b_i in range(last_frames.shape[0]):
                indicator_emb[b_i, :, :, [0]] = 1.0 - startEnd_mask[b_i]
                assert only_mask_dim is not None
                if only_mask_dim is not None and len(only_mask_dim) > 0: # only mask defined dim. # [0,1,4] or [0,1,2,3,4]
                    indicator_emb[b_i, only_mask_dim, :, last_frames[b_i]-1:] = 1.0 - startEnd_mask[b_i]
                
        return indicator_emb

    def random_mask(self, x):
        bs, _, _, seq_len = x.shape
        choose_seq_num = np.random.choice(seq_len - 1, 1) + 1
        x_ = []
        mask = torch.zeros((bs, seq_len), device=x.device)
        for b in range(bs):
            choose_seq = np.random.choice(seq_len, choose_seq_num, replace=False)
            choose_seq.sort()
            mask[b, choose_seq] = 1
            x_b = x[b, :, :, choose_seq]
            x_.append(x_b)
        x = torch.stack(x_)
        return x, mask
    
    def mask_cond_no_object(self, cond, y_flag, force_mask=False):

        # import pdb;pdb.set_trace()
        # cond is 3 shapes;
        
        bs = cond.shape[0]

        if force_mask: 
            return torch.zeros_like(cond)
        
        # ! if the dropped object is not sufficient.
        elif self.training and self.cond_mask_prob_noObject > 0.: # drop condition
        # elif self.training: # drop condition
            import pdb;pdb.set_trace()
            assert y_flag is not None
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            if y_flag.sum() > y_flag.shape[0] * (1 - self.cond_mask_prob_noObject): # if the dropped object is not sufficient.
                selected_obj_num = math.ceil(y_flag.sum() - y_flag.shape[0] * (1 - self.cond_mask_prob_noObject))
                select_obj_idx = mask.nonzero()[:selected_obj_num].squeeze()
                y_flag[select_obj_idx] = 0
        
            return cond * (1. * y_flag.view(bs, 1, 1)) # 1: with object; 0 with dropped out object
        
        else:
            return cond 
        
    def cmdm_forward_attnmask(self, x, timesteps, y=None, weight=1.0):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, nfeats, njoints, nframes = x.shape

        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]

        # seq_mask = y['hint'].sum(-1) != 0
        # guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]
        
        # import pdb;pdb.set_trace()
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode and not 'notext' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb = emb + self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        if 'action' in self.cond_mode and not 'notext' in self.cond_mode:
            action_emb = self.c_embed_action(y['action'])
            emb = emb + self.mask_cond(action_emb, force_mask=force_mask)

        if 'finalPosPelvis' in self.cond_mode:
            # use the GT end frame as condition.
            # if use object position, this should use the same normalization.

            # import pdb;pdb.set_trace()

            last_frames = y['lengths']

            input_gt_poses = y['input_motion'][:, :, 0, :].permute(0, 2, 1) # after normalization.
            final_pos = []
            for tmp_i, tmp_f in enumerate(last_frames-1): # this is slow.
                final_pos.append(input_gt_poses[tmp_i, tmp_f, [0,1,2,3,4]])
            final_pos = torch.stack(final_pos, axis=0) # [B, 5] # [x,z, rot_cos, rot_sin, height]

            final_pos_feat = self.c_final_pos_encoder(final_pos.float()) # [B, d, 5] -> [B, d, 512]
            # add the final position of the pelvis as the condition, add in `emb`
            emb += final_pos_feat[None]        
        
        if 'sceneGrid' in self.cond_mode:
            
            scene_data = y['scene'].float().to(x.device)
            if 'unet' in self.cond_mode:
                scene_grid = self.map_encoder(scene_data)
            else:
                scene_feat, scene_grid = self.map_encoder(scene_data)

            # import pdb;pdb.set_trace()
            real_x = self.inv_transform_th(x.permute(0, 2,3,1)).permute((0, 3, 1, 2))
            
            pos = real_x[:,:2,0,:].permute((0,2,1)).clone() # B, T, 2

            # pos = x[:,:2,0,:].permute((0,2,1)).clone() # B, T, 2
            normalization_scale = scene_grid.shape[-1] / (6.4 * 2)
            pos[..., 1] = pos[..., 1] * -1 # flip y axis
            pos = pos * normalization_scale + scene_grid.shape[-1] / 2
            
            # visualization the scene grid and position.
            if timesteps[0] % 50 == 0 and False: 
                visualize_feature_grid(pos.detach().clone(), scene_grid.detach().clone(), step=timesteps[0])

            interp_feats = query_feature_grid(pos, scene_grid)   #[B, T, d] 

        # import pdb;pdb.set_trace()
        if self.double_indicator and x.shape[1] > 268: # this would not influence the goal-reaching trajectory model.
            force_no_startEnd = y.get('uncond_startEnd', False)
            binary_indicator_emb = self.get_binary_indicator(x, y, self.cond_mode, cfg_startEnd=self.cfg_startEnd, only_mask_dim=self.only_mask_dim, force_mask=force_no_startEnd) # default: False->no force
            
            # overwrite the start and end frame with the GT.
            input_motion = y.get('input_motion', None)
            assert input_motion is not None
            if 'startEnd' in self.cond_mode or 'startOnly' in self.cond_mode or 'startRootEnd' in self.cond_mode: 
                input_motion.requires_grad = True # this is used for inference guidance.
                # x[binary_indicator_emb == 0] = input_motion[binary_indicator_emb == 0]
                x.detach()[binary_indicator_emb == 0] = input_motion[binary_indicator_emb == 0]
                x.requires_grad = True # this is used for inference guidance. | FIXME this has bugs for num_repetition.

            x = torch.cat((x, binary_indicator_emb), axis=1)

        if 'objectBPS' in self.cond_mode: # use the predicted poses.
            # ! should not write hard code !!!
            # get the world poses, that starts from (0, 0, 1, 0, height)
            all_pose_shape = int(x.shape[1] / 2)
            input_pose = x[:, :all_pose_shape, 0, :] 
            # b, f, frames = input_pose.shape
            # import pdb;pdb.set_trace
            whole_poses = self.inv_transform_th(input_pose.permute((0, 2, 1))) # the feature dim is the last.
            real_world_poses = recover_from_ric(whole_poses.float(), joints_num=22, root_rep=self.root_representation) # old joint representation. TODO: add joint representation.

            # TODO: re-normalize the pose to the original pose in world 3D space.
            transform_mat = y['transform_mat']
            # self.obj_transformation[obj_name[0]]
            obj_tranform_trans = y['obj_transform_trans']
            obj_tranform_rotation = y['obj_transform_rotation']
            
            # import pdb;pdb.set_trace() # the translation and orientaion in the world space.
            device=real_world_poses.device
            obj_coord_poses = canonicalize_poses_to_object_space_th(real_world_poses, transform_mat.to(device), obj_tranform_trans.to(device), obj_tranform_rotation.to(device))
        
        # import pdb;pdb.set_trace()
        # ! no need to use the text, timestep;
        x = self.c_input_process(x)

        if 'sceneGrid' in self.cond_mode:
            x = x + interp_feats.permute((1,0,2))                    
        
        if 'objectBPS' in self.cond_mode:
            import pdb;pdb.set_trace()
            # if no object, y['scene'] == zeros; bps_to_joint_dis == zeros and bps_to_joint_idx == zeros;
            bs, frames = x.shape[1], x.shape[0]
            njoint = obj_coord_poses.shape[-2]
            # bps_data = y['obj_bps'].float().to(x.device)
            bps_data = y['scene'].float().to(x.device) # batch, 1000
            reshape_batch = obj_coord_poses.reshape(-1, njoint, 3).shape[0]
            bps_to_joint_dis, bps_to_joint_idx = get_one_directional_bps_feature(np.tile(self.bps_set[None], (reshape_batch, 1, 1)), obj_coord_poses.float().reshape(-1, njoint, 3), return_bps_indices=True, device=x.device) # get bps feature for each poses;
            bps_to_joint_dis = bps_to_joint_dis.reshape(bs, frames, -1)
            bps_to_joint_idx = bps_to_joint_idx.reshape(bs, frames, -1)
            bps_input_feat = torch.stack((bps_data[:, None].repeat(1, frames, 1), bps_to_joint_dis, bps_to_joint_idx), axis=-1) # batch, frames, 1000, 3

            input_feat_merge = self.c_merge_bps_info(bps_input_feat.permute(0, 2, 3, 1)) # bs, njoints, nfeats, nframes

            import pdb;pdb.set_trace()
            # obj_flag = y.get('obj_flag', None)
            obj_id = y.get('obj_id', None)
            obj_flag = [1 if obj_i is not None else 0 for obj_i in obj_id]
            obj_flag = torch.tensor(obj_flag).to(x.device)

            if self.cfg_noObj:
                force_mask = y.get('uncond_noObj', False) # this is used for inference.
                tmp_input_feat_merge = self.mask_cond_no_object(input_feat_merge.permute(1, 0, 2), obj_flag, force_mask=force_mask)
                input_feat_merge = tmp_input_feat_merge.permute(1, 0, 2)
                x += input_feat_merge 
            else:
                x += input_feat_merge  # frame, batch, feat_dim;
        
        
        x_mask = y['mask']
        x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
        time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
        
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements
        output = self.c_seqTransEncoder(xseq.float(), src_key_padding_mask=~aug_mask[:,:,0].permute(1,0))  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight 

        # import pdb;pdb.set_trace()
        return control
    
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nfeats, njoints, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)

        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb = emb + self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb = emb + self.mask_cond(action_emb, force_mask=force_mask)

        if 'finalPosPelvis' in self.cond_mode and not 'nopretraining' in self.cond_mode:
            # use the GT end frame as condition.
            # if use object position, this should use the same normalization.

            last_frames = y['lengths']
            # import pdb;pdb.set_trace()

            input_gt_poses = y['input_motion'][:, :, 0, :].permute(0, 2, 1) # after normalization.
            final_pos = []
            for tmp_i, tmp_f in enumerate(last_frames-1): # this is slow.
                final_pos.append(input_gt_poses[tmp_i, tmp_f, [0,1,2,3,4]])
            final_pos = torch.stack(final_pos, axis=0) # [B, 5] # [x,z, rot_cos, rot_sin, height]


            final_pos_feat = self.final_pos_encoder(final_pos.float()) # [B, d, 5] -> [B, d, 512]
            # import pdb;pdb.set_trace()
            # add the final position of the pelvis as the condition, add in `emb`
            emb += final_pos_feat[None]

        if self.double_indicator:
            force_no_startEnd = y.get('uncond_startEnd', False)
            binary_indicator_emb = self.get_binary_indicator(x, y, self.cond_mode, cfg_startEnd=self.cfg_startEnd, only_mask_dim=self.only_mask_dim, force_mask=force_no_startEnd) # default: False->no force
            
            # overwrite the start and end frame with the GT.
            input_motion = y.get('input_motion', None)
            assert input_motion is not None
            assert 'startEnd' in self.cond_mode
            
            input_motion.requires_grad = True # this is used for inference guidance.
            x.detach()[binary_indicator_emb == 0] = input_motion[binary_indicator_emb == 0]
            if x.is_leaf:
                x.requires_grad = True # this is used for inference guidance.
                
            # import pdb;pdb.set_trace()
            x = torch.cat((x, binary_indicator_emb), axis=1)

        ### mdm inference.
        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_enc_new_attention':
                    
            x_mask = y['mask']
            x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
            time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
            
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements

            # * add controlled signal.
            output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0), control=control)[1:]  # , src_key_padding_mask=~maskseq)[1:]  # [seqlen, bs, d]

            
        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]

        return output

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if 'scene' in y.keys(): # 'scene' means for 2D floor masks;
            if x.shape[1] > 268 :
                control = self.cmdm_forward_attnmask(x, timesteps, y)
            else:
                ### ! check the inplace operation change the x itself.
                control = self.cmdm_forward_attnmask(x.clone(), timesteps, y) # this one will change the x.clone().is_leaf=False
        else:
            control = None
            
        # import pdb;pdb.set_trace()
        # -------------------
        output = self.mdm_forward(x, timesteps, y, control)
        return output


#### basic modules ####
class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.ModuleList([
            nn.Linear(self.input_feats, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        ])

    def forward(self, x):
        x = x.permute((1, 0, 2))

        for module in self.poseEmbedding:
            x = module(x)  # [seqlen, bs, d]
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)