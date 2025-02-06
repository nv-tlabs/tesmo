# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import numpy as np
import math
import torch
from PIL import Image
import shutil

from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from utils.fixseed import fixseed

from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion, InpaintingGaussianDiffusionOnlyGoal
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender

from sample.condition import StartGoalPosition, MapCollisionLoss


# import pdb;pdb.set_trace=lambda:None

def main():
    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    
    if args.max_frames is None:
        max_frames = 196 if args.dataset in ['kit', 'humanml', 'humanmlabsolute'] else 60
    else:
        max_frames = args.max_frames
        
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    device = args.device

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    
    if args.dataset == 'humanmlabsolute':
        from data_loaders.humanmlabsolute_utils import get_inpainting_mask
    else:    
        from data_loaders.humanml_utils import get_inpainting_mask

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size ({args.batch_size}) or reduce num_samples ({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    if args.split is not None:
        split = args.split #'sample_walk'
    else:
        split = 'test'
        
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    
    # current bug: use different normalization from training stage.
    # import pdb; pdb.set_trace()
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=split,
                              load_mode='train',
                              size=args.num_samples,
                              opt_name=args.opt_name)  # in train mode, you get both text and motion.
    
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    DiffusionClass = InpaintingGaussianDiffusionOnlyGoal if args.only_goal_mask else DiffusionClass
    
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data) # ! model_kwargs is what can be inputted to the model.
    
    ori_input_motions, model_kwargs = next(iterator)
    
    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    can_maps = None
    
        
    # TODO: indicator start = 1.0
    if args.input_trajectory is not None:
        # import pdb;pdb.set_trace()
        print(f'input motion kind: {args.input_motion_kind}')
        
        if args.input_motion_kind == 'a_star':
            # import pdb;pdb.set_trace()
            from utils.input_trajectory import load_trajectory, insert_body_trajectory
            input_trajectory = load_trajectory(args.input_trajectory)
            # TODO: change it with our designed motion trajectory.
            input_motions = insert_body_trajectory(ori_input_motions, input_trajectory, body_part='root_horizontal', dataset=data.dataset)
            
        elif args.input_motion_kind == 'absolute_xz_rot':
            from utils.input_trajectory import absolute_to_relative, convert_root_global_to_local
            input_results = np.load(args.input_trajectory, allow_pickle=True).tolist()
            
            if os.path.exists(os.path.join(os.path.dirname(args.input_trajectory), 'sorted_idx.npy')):
                idx = np.load(os.path.join(os.path.dirname(args.input_trajectory), 'sorted_idx.npy'))
                print('redo the idx')
            else:
                idx = np.arange(input_results['motion'].shape[0])
            all_motions = input_results['motion'][idx] #
            input_lengths = input_results['lengths'][idx]
            model_kwargs['y']['lengths'] = torch.tensor(input_lengths).int() # load original dataset.
            
            abs_model_output = input_results['model_output'][0][idx]
            input_motions = convert_root_global_to_local(ori_input_motions, input_lengths, abs_model_output, dataset=data.dataset)    
            
            # import pdb;pdb.set_trace()

            if True:
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
                    # import pdb;pdb.set_trace()
        else:
            assert False, 'not implemented'
            # load motion
            # get the relative rot xz and rotation representation.
            
            
    elif args.input_distance is not None:
        pass
        # import pdb;pdb.set_trace()
        radius = args.input_distance
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
        
        
            
    elif args.input_length is not None:
        import pdb;pdb.set_trace()
        input_motions = ori_input_motions.clone()
        for idx, length in enumerate(model_kwargs['y']['lengths']):
            # if length > args.input_length:
            input_motions[idx, :, :, args.input_length-1] = input_motions[idx, :, :, length-1]
            input_motions[idx, :, :, args.input_length:] *= 0.0
        
        model_kwargs['y']['lengths'] = torch.ones(len(model_kwargs['y']['lengths'])).int() * args.input_length
        
    else:
        input_motions = ori_input_motions
        
        
    input_motions = input_motions.to(dist_util.dev())

    original_input_text = model_kwargs['y']['text']
    # import pdb;pdb.set_trace()
    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts

    # add inpainting mask according to args
    # assert max_frames == input_motions.shape[-1]
    
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    
    # ! model\-kwargs is in cpu;
    ## TODO: root is the first three dimension.
    model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, input_motions.shape)).float().to(dist_util.dev())
    
    if args.condition_type == 'goal': # seperate goal as condition.
        model_kwargs['y']['start_goal'] = model_kwargs['y']['start_goal'].to(dist_util.dev())

    # save the last output frame's output start point and end point;
    if args.keep_last_step:
        model_kwargs['y']['keep_last_step'] = True
    elif args.keep_n_step != 0:
        model_kwargs['y']['keep_n_step'] = args.keep_n_step
        
    from utils.smpl_util import load_body_models, run_smplx_model
    support_dir = 'body_models'
    bm_dict = load_body_models(support_base_dir=support_dir, device=device)
    
    
    all_motions = []
    model_output = []
    all_lengths = []
    all_text = []

    input_all_motions = []
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        # combination of different condition
        if args.guidance_mode == 'start_goal,floor_map_2d': 
            cond_fn_1 = StartGoalPosition(target=input_motions, 
                            motion_length_cut=model_kwargs['y']['lengths'],
                            only_xz=args.only_xz,
                            classifiler_scale=args.classifier_scale,
                            with_y=args.with_y,
                            with_orient=args.with_orient,
                            classifier_scale_orient=args.classifier_scale_orient,
                            with_decay=args.with_decay,
                            decay_step=args.decay_step,
                            decay_scale=args.decay_scale,
                            use_mse_loss=args.gen_mse_loss,
                            )
            cond_fn_2 = MapCollisionLoss(
                target=can_maps, 
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                motion_length_cut=model_kwargs['y']['lengths'],
                classifiler_scale=args.classifier_scale_2d_maps,
            )
            
            cond_fn_traj=[cond_fn_1, cond_fn_2]
            
        elif args.guidance_mode == 'start_goal':
            pass
            print(f'classifier scale : {args.classifier_scale}')
            cond_fn_traj = StartGoalPosition(target=input_motions, 
                            motion_length_cut=model_kwargs['y']['lengths'],
                            only_xz=args.only_xz,
                            classifiler_scale=args.classifier_scale,
                            with_y=args.with_y,
                            with_orient=args.with_orient,
                            classifier_scale_orient=args.classifier_scale_orient,
                            with_decay=args.with_decay,
                            decay_step=args.decay_step,
                            decay_scale=args.decay_scale,
                            use_mse_loss=args.gen_mse_loss,
                            )
            
        elif args.guidance_mode == "kps":
            pass
            
        elif args.guidance_mode == 'floor_map_2d':
            
            import pdb;pdb.set_trace()
            # TODO: canonicalize the floor map given the start point position and orientation.
            
            cond_fn_traj = MapCollisionLoss(
                target=can_maps, 
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                motion_length_cut=model_kwargs['y']['lengths'],
                classifiler_scale=args.classifier_scale,
            )
        else:
            print('no condition')
            cond_fn_traj = None
        
        ### TODO: add 2D Distance Transform Map and Silhouette Map for collision.
        
        ori_sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions, # this is the start input.
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn_traj,
            cond_fn_with_grad=args.cond_fn_with_grad, # ! this is same as Trace.
            
        )

        # import pdb;pdb.set_trace()
        root_represent = args.root_representation
        # import pdb;pdb.set_trace()
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if args.dataset == 'humanml' else 21
            sample = data.dataset.t2m_dataset.inv_transform(ori_sample.detach().cpu().permute(0, 2, 3, 1)).float()
            model_output.append(sample.cpu().numpy())
            sample = recover_from_ric(sample, n_joints, root_rep=root_represent)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        else:
            sample = ori_sample        
        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")
    
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'model_output': model_output,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    
    ################################################################
    print('start to visualize...')
    # ! only visualize the 2D trajectory + height (Yes or Not);
    
    # prepare the input motion
    if args.dataset in ['humanml'] and model.data_rep == 'hml_vec':
        # input_motions = input_motions[:, :263, :, :] # extract the origianl 263 dim from HumanML3D
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints, root_rep=root_represent)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
        
    use_sorted_index = np.arange(args.num_samples)
    
    npy_sorted_path = os.path.join(out_path, 'sorted_idx.npy')
    print(f"saving sorted idx results file to [{npy_sorted_path}]")
    np.save(npy_sorted_path, use_sorted_index)
    
    #### visualization
    if args.dataset in ['humanml'] and True:
        # Recover XYZ *positions* from HumanML3D vector representation
        
        # TODO check: input and generated are the same.
        # import pdb;pdb.set_trace()
        
        ### HumanML3D visualization
        for ori_sample_i in range(args.num_samples):
            sample_i = use_sorted_index[ori_sample_i]
            
            rep_files = []
            if args.show_input:
                caption = 'Input Motion'
                if args.show_all_frames:
                    length = input_motions.shape[-1]
                else:
                    length = model_kwargs['y']['lengths'][sample_i]
                    
                motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
                save_file = 'input_motion{:02d}.mp4'.format(ori_sample_i)
                animation_save_path = os.path.join(out_path, save_file)
                rep_files.append(animation_save_path)
                print(f'[({ori_sample_i}) "{caption}" | -> {save_file}]')
                plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode='gt',
                            gt_frames=gt_frames_per_sample.get(sample_i, []))
                
            for rep_i in range(args.num_repetitions):
                caption = all_text[rep_i*args.batch_size + sample_i]
                if args.guidance_param == 0 or args.cfg_motion_control:
                    caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
                else:
                    caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
                
                if args.show_all_frames:
                    length = input_motions.shape[-1]
                else:
                    length = all_lengths[rep_i*args.batch_size + sample_i] 
                    
                sample_motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = 'sample{:02d}_rep{:02d}.mp4'.format(ori_sample_i, rep_i)
                animation_save_path = os.path.join(out_path, save_file)
                rep_files.append(animation_save_path)
                print(f'[({ori_sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
                plot_3d_motion(animation_save_path, skeleton, sample_motion, title=caption,
                            dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                            gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','), \
                            input_motion=input_motions[sample_i].transpose(2, 0, 1)[:length], 
                            floor_maps=can_maps)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
        

        if args.num_repetitions > 1: # stack the videos
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
    
    
    #### run SMPLify
    if args.dataset in ['humanml'] and args.joint2mesh:
        from tqdm import tqdm
        from visualize.render_utils import get_bodies
        
        for ori_sample_i in tqdm(range(args.num_samples)):
            sample_i = use_sorted_index[ori_sample_i]

            if args.show_all_frames:
                length = input_motions.shape[-1]
            else:
                length = model_kwargs['y']['lengths'][sample_i]

            for rep_i in range(args.num_repetitions):
                body_mesh_folder = f'{out_path}/sample{sample_i}_rep{rep_i:02d}_obj'
                os.makedirs(body_mesh_folder, exist_ok=True)

                sample_motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                index = rep_i*args.batch_size + sample_i
                body_meshes = get_bodies(body_mesh_folder, index, sample_motion, export_perframe_mesh=True, same_transl=True)

                # import pdb;pdb.set_trace()
                
    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()