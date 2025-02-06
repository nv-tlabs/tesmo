import torch
from model.cfg_sampler import wrap_model
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.model_blending import ModelBlender

### TODO: why this works, Blend the models;
def load_model_blending_and_diffusion(args_list, data, device, ModelClass=MDM, DiffusionClass=gd.GaussianDiffusion):
    models = [load_model(args, data, device, ModelClass)[0] for args in args_list]
    model = ModelBlender(models, [1/len(models)]*len(models)) if len(models) > 1 else models[0]
    
    _, diffusion = create_model_and_diffusion(args_list[0], data, DiffusionClass=DiffusionClass)
    return model, diffusion

def load_model(args, data, device, ModelClass=MDM):
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=ModelClass)
    model_path = args.model_path
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args)
    return model, diffusion

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if 't_pos_encoder.pe' in missing_keys:
        missing_keys.remove('t_pos_encoder.pe')
    if 't_pos_encoder.pe' in unexpected_keys:
        unexpected_keys.remove('t_pos_encoder.pe')
    
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def load_pretrained_mdm(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


def load_split_mdm(model, state_dict, cutting_point):
    new_state_dict = {}
    orig_trans_prefix = 'seqTransEncoder.'
    for k, v in state_dict.items():
        if k.startswith(orig_trans_prefix):
            orig_layer = int(k.split('.')[2])
            orig_suffix = '.'.join(k.split('.')[3:])
            target_split = 'seqTransEncoder_start.' if orig_layer < cutting_point else 'seqTransEncoder_end.'
            target_layer = orig_layer if orig_layer < cutting_point else orig_layer - cutting_point
            new_k = target_split + 'layers.' + str(target_layer) + '.' + orig_suffix
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or k.startswith('multi_person.') for k in missing_keys])


def create_model_and_diffusion(args, data, ModelClass=MDM, DiffusionClass=SpacedDiffusion):
    # import pdb;pdb.set_trace()
    model = ModelClass(**get_model_args(args, data)) # this is only for model.
    # import pdb;pdb.set_trace()
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion

def get_model_args(args, data):
    # this is for model itself.
    
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    # cond_mode = 'text' if args.dataset in ['humanml', 'babel', 'pw3d'] else 'action'
    if args.dataset in ['humanml', 'babel', 'pw3d', 'humanmlabsolute']:
        if hasattr(args, 'condition_type') and args.condition_type is not None:
            cond_mode = args.condition_type
            print('Using condition type: ', cond_mode)
        else:
            cond_mode = 'text'
    elif args.dataset in ['circle']:
        # cond_mode = 'text'  # scene
        # cond_mode = 'scene'
        cond_mode = args.condition_type  # 'scene_goal'
    else: # this is not used.
        cond_mode = 'action'
    
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml', 'pw3d']:
        data_rep = 'hml_vec'
        if hasattr(args, 'input_feats'):
            njoints = args.input_feats
        else:
            njoints = 263
        nfeats = 1
    elif args.dataset in ['babel', 'humanmlabsolute']:
        data_rep = 'rot6d' # with translation.
        njoints = 135
        nfeats = 1
    else:
        raise TypeError(f'dataset {args.dataset} is not currently supported')

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'diffusion-steps': args.diffusion_steps, 'batch_size': args.batch_size, 'use_tta': args.use_tta,
            'trans_emb': args.trans_emb, 'concat_trans_emb': args.concat_trans_emb, 
            'cond_mask_prob_startEnd': args.cond_mask_prob_startEnd,
            'cfg_startEnd': args.cfg_startEnd,
            'cfg_startEnd_addNoise': args.cfg_startEnd_addNoise,
            'cond_mask_prob_noObj': args.cond_mask_prob_noObj,
            'cfg_noObj': args.cfg_noObj,
            'inv_transform_th': data.dataset.t2m_dataset.inv_transform_th,
            'args': args}


#### ! this is for create diffsuion model
def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    
    print(f"number of diffusion-steps: {steps}")

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if hasattr(args, 'multi_train_mode'):
        multi_train_mode = args.multi_train_mode
    else:
        multi_train_mode = None

    # import pdb;pdb.set_trace()
    
    print('use diffusion class: ', DiffusionClass)
    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        batch_size=args.batch_size,
        lambda_extra_weight=args.lambda_extra_weight,
        lambda_extra_dim=args.lambda_extra_dim,
        multi_train_mode=multi_train_mode,
        # indicator=args.indicator, # this is for adding indicator;
        # indicator_gd=args.indicator_gd,
        lambda_start_end_weight=args.lambda_start_end_weight,
        lambda_vel_rcxyz_startEnd=args.lambda_vel_rcxyz_startEnd,
        cfg_startEnd=args.cfg_startEnd,
        cfg_startEnd_addNoise=args.cfg_startEnd_addNoise,
        cond_mask_prob_startEnd=args.cond_mask_prob_startEnd,
        mask_out=args.mask_out, # this is for mask out the predicted noise.
        no_last_frame_inference=args.no_last_frame_inference, # this is for mask out the predicted noise.
    )