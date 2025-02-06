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
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.append(CODE_DIR)

from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from utils.fixseed import fixseed
from utils.input_trajectory import get_input_motion
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from sample.condition import load_condition, MapCollisionLoss

from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.utils.plot_script import explicit_plot_3d_image_SDF
from data_loaders.humanml.utils.utils_samp import save_sample_poses
import pickle
from data_loaders.humanml.utils.get_scene_bps import get_original_mesh
from data_loaders.humanml.utils.utils_samp import canonicalize_poses_to_object_space_th
import copy
from tqdm import tqdm

# import pdb;pdb.set_trace=lambda:None


# merge all image results into one.
def merge_imgs(img_root, save_image_name):
    image_files = sorted([file for file in os.listdir(img_root) if file.endswith('.png') and 'sample' in file and 'all' not in file], key=lambda x: int(os.path.basename(x).split('_')[0][6:]))
    im_height, im_width = Image.open(os.path.join(img_root, image_files[0])).size

    # 计算行数和列数
    num_images = len(image_files)
    # cols = int(math.sqrt(num_images))
    cols = 6
    rows = math.ceil(num_images / cols)

    # 创建一个新的图像矩阵
    result_image = Image.new('RGB', (cols * im_width, rows * im_height))

    # 遍历每张图片并将其拼接到图像矩阵中
    for i, file in enumerate(image_files):
        image = Image.open(os.path.join(img_root, file))
        # image = image.resize((image_size, image_size))
        row = i // cols
        col = i % cols
        result_image.paste(image, (col * im_width, row * im_height))

    # 保存拼接后的图像矩阵
    result_image.save(save_image_name)

def main():
    args_list = edit_inpainting_args() # this will load the training json, and replace the args with the json file.
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir

    print('\n save to ', out_path, '\n')

    if not args.not_run:
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
            out_path = os.path.join(os.path.dirname(args.model_path), 'edit_{}_{}_{}_seed{}_guidance{}'.format(name, niter, args.inpainting_mask, args.seed, args.guidance_param))
            if args.text_condition != '':
                out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')
        os.makedirs(out_path, exist_ok=True)
        
        print('Loading dataset...')
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size ({args.batch_size}) or reduce num_samples ({args.num_samples})'
        if args.split is not None:
            split = args.split #'sample_walk'
        else:
            split = 'test'
        args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
        data = get_dataset_loader(name=args.dataset,
                                batch_size=args.batch_size,
                                num_frames=max_frames,
                                split=split,
                                load_mode='train', # ! what's the differnet 
                                size=args.num_samples, 
                                opt_name=args.opt_name)  # in train mode, you get both text and motion.
        total_num_samples = args.num_samples * args.num_repetitions

        print("Creating model and diffusion...")
        DiffusionClass = SpacedDiffusion

        if 'cmdm' in args.condition_type: # controlnet module.
            print('load cmdm ****** \n')
            from model.scene_aware_component.cmdm import CMDM
            model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), ModelClass=CMDM, DiffusionClass=DiffusionClass)
        elif 'CMDMObj' in args.condition_type:
            print('load CMDMObj ****** \n')
            # args_list[0].diffusion_steps = 10 # debug
            from model.scene_aware_component.cmdm_object import CMDM_Object
            model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), ModelClass=CMDM_Object, DiffusionClass=DiffusionClass)
        else: 
            print('not support other condition types')
            return 
            
        print('model analysis: ')
        
        if isinstance(model, ClassifierFreeSampleModel):
            print('use ClassifierFreeSampleModel !!!!')
        else:
            pass

        iterator = iter(data) 
        ori_input_motions, model_kwargs = next(iterator)  # ! model_kwargs is what can be inputted to the model.
        
        print(f"saving visualizations to [{out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        can_maps = None

        assert args.root_representation in ['root_pos_abs_rotcossin_height_only_norm', 'root_pos_abs_rotcossin_height_pose_with_humanml'] 

        # TODO: input a floor maps; Given its random intialized start position and end orientation.    
        # ! take the start pose local pose as input; We can also add jittering on the init start pelvis position and orientation;
        input_motions, can_maps, extra_sample_motion = get_input_motion(
            ori_input_motions.clone(), 
            model_kwargs, 
            args.input_motion_kind, # use 
            out_path, 
            args.input_trajectory, # this is used to load a* generated trajectory.
            data, # used for canonicalization.
            args.input_distance, 
            args.input_length, 
            args.final_frame, 
            args.floor_map_2d, # ! this is the floor map from input args.
            args.root_representation)
        input_motions = input_motions.to(dist_util.dev())

        # blend_input_trajectory or the input itself. 
        if hasattr(args, 'blend_input_trajectory') and args.blend_input_trajectory:
            if extra_sample_motion is not None:
                extra_sample_motion = extra_sample_motion.to(dist_util.dev())
                model_kwargs['y']['extra_sample_motion'] = extra_sample_motion # use during the sampling input.
            else:
                model_kwargs['y']['extra_sample_motion'] = input_motions # use during the sampling input.
            
            # import pdb;pdb.set_trace()
            if hasattr(args, 'blend_mask') and args.blend_mask is not None:
                model_kwargs['y']['blend_mask'] = torch.zeros_like(input_motions).to(dist_util.dev()) # batch, dim, 1, frames
                model_kwargs['y']['blend_mask'][..., :args.blend_mask] = 1.0
                
            model_kwargs['y']['blend_scale'] = args.blend_scale if hasattr(args, 'blend_scale') else 0.5
        
        # change the input text.
        original_input_text = model_kwargs['y']['text']
        if args.text_condition != '':
            texts = [args.text_condition] * args.num_samples
            model_kwargs['y']['text'] = texts

        if args.keep_last_step:
            model_kwargs['y']['keep_last_step'] = True
        elif args.keep_n_step != 0:
            model_kwargs['y']['keep_n_step'] = args.keep_n_step
        print('********** \n')
        print('keep_last_step: ', args.keep_last_step)
        print('keep_n_step: ', args.keep_n_step)

        # add inpainting mask according to args
        # assert max_frames == input_motions.shape[-1]
        gt_frames_per_sample = {}
        
        # we need to overwrite the input motions.
        model_kwargs['y']['input_motion'] = input_motions
        
        # save the last output frame's output start point and end point;
        from utils.smpl_util import load_body_models, run_smplx_model
        
        support_dir = 'body_models'
        bm_dict = load_body_models(support_base_dir=support_dir, device=device)
        
        all_motions = []
        model_output = []
        all_lengths = []
        all_text = []
        ori_sample_list = []
        REAL_RUN = True # real run the inference;
        name_list = data.dataset.t2m_dataset.name_list

        npy_path = os.path.join(out_path, 'results.npy')
        
        if REAL_RUN:
            # save y
            input_y_path = os.path.join(out_path, 'input_y.pickle') 
            with open(input_y_path, 'wb') as fw:
                pickle.dump(model_kwargs['y'], fw)

            # import pdb;pdb.set_trace()
            for rep_i in range(args.num_repetitions):
                print(f'### Start sampling [repetitions #{rep_i}]')

                # import pdb;pdb.set_trace()
                if args.with_erosion and args.erosion_pixel > 0:
                    pass
                    # add erosion pixels for can_maps.
                    # can_maps = can_maps[:, 0].unsqueeze(1)
                    from scipy.ndimage import binary_erosion
                    # Create a kernel with a size of 4 pixels
                    kernel = np.ones((args.erosion_pixel, args.erosion_pixel))
                    # Convert the torch tensor to a NumPy array
                    numpy_image = can_maps.cpu().detach().numpy()

                    import torchshow as ts
                    # Apply erosion to the NumPy array
                    erosion_list = []
                    for i in range(numpy_image.shape[0]):
                        eroded_image = binary_erosion(numpy_image[i][0] * 1.0 / 255.0, kernel)
                        erosion_list.append(torch.from_numpy(eroded_image* 255.0).type(torch.uint8).to(dist_util.dev()))
                    # Convert the eroded NumPy array back to a torch tensor
                    torch_image = torch.stack(erosion_list)
                    
                    collision_used_image = torch_image.unsqueeze(1)
                else:
                    collision_used_image = can_maps

                cond_fn_traj = load_condition(model_kwargs, args, input_motions,
                    can_maps=collision_used_image, 
                    device=device, 
                    data=data, 
                    extra_sample_motion=extra_sample_motion
                )
                
                # add CFG scale to batch
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

                sample_fn = diffusion.p_sample_loop
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
                root_represent = args.root_representation
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
            all_motions = all_motions[:total_num_samples]  # [bs, njoints, 3, seqlen]
            all_text = all_text[:total_num_samples]
            all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
            
            # remember the repetitions.
            print(f"saving results file to [{npy_path}]")
            np.save(npy_path,
                    {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                    'model_output': model_output,
                    'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
            with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
                fw.write('\n'.join(all_text))
            with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
                fw.write('\n'.join([str(l) for l in all_lengths]))
        else:
            # load the results,  and only evaluate.
            result = np.load(npy_path, allow_pickle=True).item()
            all_motions = result['motion']
            all_lengths = result['lengths']
            all_text = result['text']
            
            root_represent = args.root_representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if args.dataset == 'humanml' else 21

        ################################################################
        print('start to visualize...')
        # ! only visualize the 2D trajectory + height (Yes or Not);
        
        # inverse transform the input motion into world coordinate.
        if args.dataset in ['humanml'] and model.data_rep == 'hml_vec':
            before_inv_input_motions = input_motions.clone()
            # input_motions = input_motions[:, :263, :, :] # extract the origianl 263 dim from HumanML3D
            input_motions_param = data.dataset.t2m_dataset.inv_transform(input_motions.detach().cpu().permute(0, 2, 3, 1)).float()
            input_motions = recover_from_ric(input_motions_param, n_joints, root_rep=root_represent)
            input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
            
        use_sorted_index = np.arange(args.num_samples) # with order.

        ### add distance error
        use_sorted_index = np.arange(args.num_samples)

        # import pdb;pdb.set_trace()
        
        #### visualization ####   
        if hasattr(data.dataset.t2m_dataset, 'load_bps') and data.dataset.t2m_dataset.load_bps:
            # real_world_poses: [B, N, J, 3]
            all_motion_tensor = torch.from_numpy(all_motions.copy()).to(device).permute(0, 3, 1, 2)
            njoints = all_motion_tensor.shape[-2]
            transform_mat = model_kwargs['y']['transform_mat']
            # self.obj_transformation[obj_name[0]]
            obj_tranform_trans = model_kwargs['y']['obj_transform_trans']
            obj_tranform_rotation = model_kwargs['y']['obj_transform_rotation']
            obj_coord_poses = canonicalize_poses_to_object_space_th(all_motion_tensor, transform_mat.to(device), obj_tranform_trans.to(device), obj_tranform_rotation.to(device))
            obj_coords_poses_list = []
            for tmp_idx, tmp_l in enumerate(all_lengths):
                obj_coords_poses_list.append(obj_coord_poses[tmp_idx, :tmp_l].cpu().numpy()) # frames, joints_num, 3

        if args.dataset in ['humanml']:
            # Recover XYZ *positions* from HumanML3D vector representation
            ### HumanML3D visualization
            for ori_sample_i in range(args.num_samples):
                for rep_i in range(args.num_repetitions):
                    # import pdb;pdb.set_trace()
                    sample_i = use_sorted_index[ori_sample_i]

                    rep_files = []
                    if args.show_input and rep_i == 0:
                        caption = 'Input Motion:' + all_text[rep_i*args.batch_size + sample_i]
                        if args.show_all_frames:
                            length = input_motions.shape[-1]
                        else:
                            length = all_lengths[rep_i*args.batch_size + sample_i]    
                        motion = input_motions[sample_i].transpose(2, 0, 1)[:length]

                        motion_para = input_motions_param[sample_i].cpu().numpy()[0, :length]
                        save_file = 'input_motion{:02d}_{}.mp4'.format(ori_sample_i, name_list[ori_sample_i])
                        animation_save_path = os.path.join(out_path, save_file)
                        print(f'[({ori_sample_i}) "{caption}" | -> {animation_save_path}]')
                        print(f'[({ori_sample_i}) "{caption}" | -> {save_file}]')
                        # import pdb;pdb.set_trace()
                        rep_files.append(animation_save_path)
                        plot_3d_motion(animation_save_path, skeleton, motion, not_show_video=args.not_show_video,
                                    ori_motion_param=motion_para, title=caption,
                                    dataset=args.dataset, fps=fps, vis_mode='gt',
                                    gt_frames=gt_frames_per_sample.get(sample_i, []))
                        print('finish plotting')
                        if args.not_show_prediction:
                            continue
                    else:
                        motion_para = None
                    
                    if can_maps is not None and not (hasattr(data.dataset.t2m_dataset, 'load_bps') and data.dataset.t2m_dataset.load_bps):
                        can_maps_example = (can_maps[ori_sample_i, 0].cpu().numpy() == 255) * 1.0
                    else:
                        can_maps_example=None

                    for rep_i in range(args.num_repetitions):
                        caption = all_text[rep_i*args.batch_size + sample_i]
                        if args.guidance_param == 0: # or args.cfg_motion_control:
                            caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
                        else:
                            caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
                        
                        if args.show_all_frames:
                            length = input_motions.shape[-1]
                        else:
                            length = all_lengths[rep_i*args.batch_size + sample_i] 
                        
                        sample_motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                        save_file = 'sample{:02d}_rep{:02d}_{}.mp4'.format(ori_sample_i, rep_i, name_list[ori_sample_i])
                        animation_save_path = os.path.join(out_path, save_file)
                        rep_files.append(animation_save_path)
                        print(f'[({ori_sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
                        print(f'[({ori_sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path.replace("mp4", "png")}]')
                        
                        plot_3d_motion(animation_save_path, skeleton, sample_motion.copy(), ori_motion_param=motion_para, 
                                    title=caption,  not_show_video=args.not_show_video,
                                    dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask, 
                                    gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','), \
                                    input_motion=input_motions[sample_i].transpose(2, 0, 1)[:length].copy(), 
                                    floor_maps=can_maps_example)
                        print('finish plotting')

                        # import pdb;pdb.set_trace()
                        if args.render_pyrender:
                            # visualize human-object interaction;
                            from visualize.render_utils import render_body_and_scene, get_bodies
                            import copy
                            
                            obj_coords_poses = obj_coords_poses_list[rep_i*args.batch_size + sample_i]
                            out_path_render = animation_save_path.replace('.mp4', '_pyrender.mp4')
                            
                            o_p = model_kwargs['y']['obj_id'][rep_i*args.batch_size + sample_i] 
                            o_p = data.dataset.t2m_dataset.obj_transformation[o_p]['src_file']

                            obj_trans_dict = {'scale': model_kwargs['y']['obj_transform_scale'][rep_i*args.batch_size + sample_i, 0].cpu().numpy()}
                            all_scene = get_original_mesh(o_p, obj_trans_dict)[0]
                            # all_scene = model_kwargs['y']['scene_mesh'][rep_i*args.batch_size + sample_i]
                            
                            min_height = all_scene.vertices[:, 1].min()
                            all_scene.vertices[:, 1] -= min_height
                            min_height = obj_coords_poses[:, :, 1].min()
                            obj_coords_poses[:, :, 1] -= min_height

                            print('save body mesh: ', args.joint2mesh)
                            if args.joint2mesh:
                                body_meshes = get_bodies(out_path, rep_i*args.batch_size + sample_i, obj_coords_poses, save_body_mesh=False, same_transl=True) # export_perframe_mesh=args.export_perframe_mesh)
                                render_body_flag = True
                                body_joint=None
                            else:
                                body_meshes = None
                                render_body_flag = False
                                body_joint=copy.deepcopy(obj_coords_poses)
                            
                            render_body_and_scene(body_meshes, out_path_render, body_joint=body_joint,  obj_mesh=copy.deepcopy(all_scene), only_top=False, render_body_flag=render_body_flag)
                            
                            # ! split utils seperated. 
                            if args.export_perframe_mesh:
                                body_mesh_folder = os.path.join(out_path, 'body_mesh_%d' % ori_sample_i)
                                os.makedirs(body_mesh_folder, exist_ok=True)
                                # real_num_frames
                                for frame_i in tqdm(range(body_meshes.vertices.shape[0])):
                                    body_meshes.save_obj(os.path.join(body_mesh_folder, 'frame{:03d}.obj'.format(frame_i)), frame_i)

                if args.num_repetitions > 1: # stack the videos
                    print('merge videos !!!!')
                    all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(ori_sample_i))
                    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
                    hstack_args = f' -filter_complex hstack=inputs={len(rep_files)} '
                    ffmpeg_rep_cmd = f'ffmpeg ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} -y {all_rep_save_file}'
                    print(ffmpeg_rep_cmd)
                    
                    os.system(ffmpeg_rep_cmd)
                    print(f'[({ori_sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
        

if __name__ == "__main__":
    main()
