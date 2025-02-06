import torch
import torch.nn as nn
from model.rotation2xyz import Rotation2xyz
from model.mdm import MDM
from model.tools import InputProcess
from model.trace_helpers import MapEncoder, query_feature_grid
from model.scene_condition import ObjectEncoder, query_feature_grid_3D
from data_loaders.humanml.utils.get_scene_bps import get_one_directional_bps_feature, load_bps_set
import numpy as np
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.utils_samp import canonicalize_poses_to_object_space_th

class MDM_MotionControl(MDM):
    # def __init__(self, **kargs):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):

        super(MDM_MotionControl, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)

        # assert self.arch == 'trans_enc' # we evaluate only on trans_enc arch
        
        # if self.dataset in ['humanmlabsolute']:
        #     input_feats = 263 + 3 + 6
        # elif self.dataset in ['babel', 'circle', 'samp']:
        #     input_feats = 135

        self.use_tta = False #use_tta # TODO: remove as we don't show tta feature
        
        self.lhand_idx = 20
        self.rhand_idx = 21

        father_args = vars(kargs['args'])
        self.double_indicator = father_args.get('indicator_double', False)

        # import pdb;pdb.set_trace()
        self.root_representation = father_args.get('root_representation') # ! this should definitely have.

        if self.root_representation == 'rel_root':
            self.root_representation = 'root_pos_abs_rotcossin_height_only_norm'
            
        print(f'root_representation in mdm: {self.root_representation}')

        ### condition.
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)  # mask all to remove condition
        
        ## binary indicator.
        indicator_emb_dim = 0
        if self.double_indicator:
            print("Using indicator to denote which dimension of the input needs to be inpainted!")
            indicator_emb_dim = self.input_feats

        # import pdb;pdb.set_trace()
        if 'goalReachingDIMOS' in self.cond_mode:
            self.input_process = InputProcess(self.data_rep, (self.input_feats + indicator_emb_dim + 6)
                                    , self.latent_dim)
        else:
            self.input_process = InputProcess(self.data_rep, (self.input_feats + indicator_emb_dim)
                                    if self.double_indicator else self.input_feats, self.latent_dim)
    
        print('***** MDM Motion Module ******')
        
        # arch is designed in MDM.
        # output is still same as the original MDM.
        
        
        # import pdb;pdb.set_trace()
        ### CFG training
        self.cfg_startEnd = kargs.get('cfg_startEnd', False) # only used for training.
        self.cond_mask_prob_startEnd = kargs.get('cond_mask_prob_startEnd', 0.)
        print(f'cfg on start and end: {self.cfg_startEnd}; scale: {self.cond_mask_prob_startEnd}')
        
        ### inpainting with specific dimension.
        # import pdb;pdb.set_trace()
        self.only_mask_dim = father_args.get('only_mask_dim', [])
        print(f'input with these dimension: {self.only_mask_dim}')        
        
        self.no_start_point_rot_height = father_args.get('no_start_point_rot_height', False)
        print(f'no_start_point_rot_height: {self.no_start_point_rot_height}')        
        
        ### condition.
        # TODO: add 2D Floor maps, 3D Scene Environments as extra condition.    
        if 'sceneGrid' in self.cond_mode:
            print(f'[WARNING] Using sceneGrid as condition!')
            map_encoder_model_arch = 'resnet18'
            map_res=256
            self.map_encoder = MapEncoder(
                model_arch=map_encoder_model_arch,
                input_image_shape=(1, map_res, map_res),
                global_feature_dim=8,
                grid_feature_dim=512,
            )
        
        if 'objectBPS' in self.cond_mode:
            print(f'[WARNING] Using object BPS as condition!') # need non-linear activation function.; Thus have three layers.
            self.bps_set = load_bps_set()

            self.merge_bps_info = ObjectEncoder(
                'bps', # data_rep
                1000*3, # [bps point xyz, w/o], bps point->object distance, bps point->body joint distance, with bps point->body joint nearest label.
                self.latent_dim,
            )

        if 'finalPosPelvis' in self.cond_mode:
            print(f'[WARNING] Using finalPosPelvis as condition!')
            self.final_pos_encoder = nn.Sequential(
                nn.Linear(5, self.latent_dim),
            )

        print(self.cond_mode)
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

            if 'action' in self.cond_mode:
                raise Exception("cond_mode action not implemented yet")
        
        print('cond_mode: {}'.format(self.cond_mode))

    
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
    
    def forward(self, x, timesteps, y=None, debug=False):
        
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        
        ! y is a dict with keys: for condition.
        cond_mode: 
        """

        # import pdb;pdb.set_trace()
        bs, nfeats, njoints, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        
        force_mask = y.get('uncond', False) # sometimes will change accroding to the inference flags.
        
      
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) # timestep embed + text embed

        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        
        if 'finalPosPelvis' in self.cond_mode:
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

            if 'extraToken' in self.cond_mode:
                emb_extra = final_pos_feat[None]
            else:
                emb += final_pos_feat[None]

        ##### the above are not correlated with the input x;


        #### ! extra embedding can be object translation or object final position;

        if self.double_indicator:
            force_no_startEnd = y.get('uncond_startEnd', False)
            binary_indicator_emb = self.get_binary_indicator(x, y, self.cond_mode, cfg_startEnd=self.cfg_startEnd, only_mask_dim=self.only_mask_dim, force_mask=force_no_startEnd) # default: False->no force
            
            # overwrite the start and end frame with the GT.
            input_motion = y.get('input_motion', None)
            assert input_motion is not None
            if 'startEnd' in self.cond_mode or 'startOnly' in self.cond_mode or 'startRootEnd' in self.cond_mode: 
                input_motion.requires_grad = True # this is used for inference guidance.
                # x[binary_indicator_emb == 0] = input_motion[binary_indicator_emb == 0]
                x.detach()[binary_indicator_emb == 0] = input_motion[binary_indicator_emb == 0]
                
                # ! note for inference guidance.
                if x.is_leaf:
                    x.requires_grad = True # this is used for inference guidance.
                
            # import pdb;pdb.set_trace()
            x = torch.cat((x, binary_indicator_emb), axis=1)

        if 'sceneGrid' in self.cond_mode:
            # TODO: get the original global poses; This is the pose after the normalization. 
            pos = x[:,:2,0,:].permute((0,2,1)) # B, T, 2

        if 'objectSDF' in self.cond_mode or 'objectBPS' in self.cond_mode: # use the predicted poses.
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

        if 'goalReachingDIMOS' in self.cond_mode:
            import pdb;pdb.set_trace()
            goal_pos = y['input_motion'] # batch, 
            last_frames = y['lengths']
            distance_grad_value = []
            for tmp_i, tmp_f in enumerate(last_frames-1):
                dis = x[tmp_i,[0,1,2,3,4], 0, :tmp_f+1] - goal_pos[tmp_i,[0,1,2,3,4], 0, tmp_f][:, None].repeat((1, tmp_f+1))
                value = torch.norm(dis[[0,1,4], :], dim=0)
                distance_grad_value.append(torch.cat([torch.cat([dis, value[None]]), torch.zeros((6, 196-tmp_f-1)).float().to(x.device)], dim=1)) # [6, 196]    

            distance_grad_value = torch.stack(distance_grad_value, axis=0) # [B, dim, T] 
            import pdb;pdb.set_trace()
            x = torch.cat([x, distance_grad_value[:, :, None]], axis=1) # only concatenate the input.

        # import pdb;pdb.set_trace()
        x = self.input_process(x)#[T, 1, d]

        # object position; they do not make much difference.

        if 'sceneGrid' in self.cond_mode:
            scene_data = y['scene'].float().to(x.device)
            scene_feat, scene_grid = self.map_encoder(scene_data)
            # import pdb;pdb.set_trace()
            real_pos = self.inv_transform_th(pos)
            interp_feats = query_feature_grid(real_pos, scene_grid)   #[B, T, d]  
            # interp_feats = query_feature_grid(pos, scene_grid)   #[B, T, d]  
            x += interp_feats.permute((1,0,2))           
        
        if 'objectSDF' in self.cond_mode:
            # TODO: organize the sdf data.
            sdf_data = y['sdf_grid'].float().to(x.device)
            sdf_grad_data = y['sdf_gradient_grid'].float().to(x.device)
            sdf_centers = y['sdf_centroid'].float().to(x.device)
            sdf_scales = y['sdf_scale'].float().to(x.device)
            sdf_input = torch.cat((sdf_data[:, None], sdf_grad_data.permute((0, 4, 1, 2, 3))), axis=1)

            # import pdb;pdb.set_trace()
            sdf_feat = query_feature_grid_3D(obj_coord_poses, sdf_input, sdf_centers, sdf_scales)   #[B, d, N]
            # sdf_grad_feat = query_feature_grid_3D(whole_poses, sdf_grad_data, sdf_centers, sdf_scales)   #[B, T, d]
            
            # import pdb;pdb.set_trace()
            bs, nfeats = sdf_feat.shape[0], sdf_feat.shape[1]
            njoints = 22
            sdf_feat = sdf_feat.reshape(bs, nfeats, -1, njoints).permute(0, 3, 1, 2)

            # import pdb;pdb.set_trace()
            input_feat_merge = self.merge_sdf_info(sdf_feat) # bs, njoints, nfeats, nframes
            
            # import pdb;pdb.set_trace()
            #TODO: mean pool and max pool
            x += input_feat_merge  # frame, batch, feat_dim;

        if 'objectBPS' in self.cond_mode:
            import pdb;pdb.set_trace()
            bs, frames = x.shape[1], x.shape[0]
            njoint = obj_coord_poses.shape[-2]
            # bps_data = y['obj_bps'].float().to(x.device)
            bps_data = y['scene'].float().to(x.device) # batch, 1000
            reshape_batch = obj_coord_poses.reshape(-1, njoint, 3).shape[0]
            bps_to_joint_dis, bps_to_joint_idx = get_one_directional_bps_feature(np.tile(self.bps_set[None], (reshape_batch, 1, 1)), obj_coord_poses.float().reshape(-1, njoint, 3), return_bps_indices=True, device=x.device) # get bps feature for each poses;
            bps_to_joint_dis = bps_to_joint_dis.reshape(bs, frames, -1)
            bps_to_joint_idx = bps_to_joint_idx.reshape(bs, frames, -1)
            bps_input_feat = torch.stack((bps_data[:, None].repeat(1, frames, 1), bps_to_joint_dis, bps_to_joint_idx), axis=-1) # batch, frames, 1000, 3

            input_feat_merge = self.merge_bps_info(bps_input_feat.permute(0, 2, 3, 1)) # bs, njoints, nfeats, nframes
            x += input_feat_merge  # frame, batch, feat_dim;

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_enc_new_attention':
            
            # import pdb;pdb.set_trace()
            x_mask = y['mask']
            x_mask = x_mask.permute((3, 0, 1, 2)).reshape(nframes, bs, -1)
            time_mask = torch.ones(bs, dtype=bool).reshape(1, bs, -1).to(x_mask.device)
            
            if 'extraToken' in self.cond_mode:
                # import pdb;pdb.set_trace()
                xseq = torch.cat((emb_extra, emb, x), axis=0)  # [seqlen+2, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                aug_mask = torch.cat((time_mask, time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0))[2:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            else:
                xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                aug_mask = torch.cat((time_mask, x_mask), 0).to(xseq.device) # ignore the attention for the padding elements
                output = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask[:,:,0].permute(1,0))[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        
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