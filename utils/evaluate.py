import numpy as np
from eval.eval_hsi_plausibility import calculate_euler_angles_batch, calculate_euler_angles
import os
import matplotlib.pyplot as plt
from data_loaders.humanml.utils.utils_samp import canonicalize_poses_to_object_space_th
from model.scene_condition import query_feature_grid_3D
import torch
import trimesh
from data_loaders.humanml.utils.utils_samp import save_sample_poses
import json

def get_vertices_from_sdf_volume(sdf_volume, voxel_size, grid_min):
    contact_index = (sdf_volume <= 0).nonzero()
    contact_v_x = (contact_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
    contact_v_y = (contact_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
    contact_v_z = (contact_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
    contact_v = np.stack([contact_v_x, contact_v_y, contact_v_z], -1)
    
    out_index = (sdf_volume > 0).nonzero()
    out_v_x = (out_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
    out_v_y = (out_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
    out_v_z = (out_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
    out_v = np.stack([out_v_x, out_v_y, out_v_z], -1)

    return contact_v, out_v

def evaluate_rec_goal_reaching(input_motions, all_motions, all_lengths, model_kwargs, out_path, args, debug_dir=None, save_intermediate_results=True):

    dist_error = []
    start_pose_dist_list = []
    end_pose_dist_list = []
    
    rec_start_height_dist_list = []
    rec_end_height_dist_list = []
    
    rec_start_pose_dist_list = []
    rec_end_pose_dist_list = []
    
    start_end_pose_dist_list= []
    # [bs, njoints, 3, seqlen]
    
    start_orient_dist_list = []
    end_orient_dist_list = []
    end_input = []
    end_input_orient = []
    end_input_orient_pred = []
    # for sample_i in range(len(model_kwargs['y']['lengths'])):
    for sample_i in range(len(all_lengths)):
        if args.show_all_frames:
            length = input_motions.shape[-1]
        else:
            # length = model_kwargs['y']['lengths'][sample_i]
            length = all_lengths[sample_i]

        # import pdb;pdb.set_trace()
        start_pose_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, 1] - all_motions[sample_i, 0, :, 0])[[0,2]])
        end_pose_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, length-1] - all_motions[sample_i, 0, :, length-2])[[0,2]])
        
        rec_start_pose_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, 0] - input_motions[sample_i, 0, :, 0])[[0,2]])
        rec_end_pose_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, length-1] - input_motions[sample_i, 0, :, length-1])[[0,2]])

        rec_start_height_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, 0] - input_motions[sample_i, 0, :, 0])[[1]])
        rec_end_height_dist = np.linalg.norm(np.abs(all_motions[sample_i, 0, :, length-1] - input_motions[sample_i, 0, :, length-1])[[1]])

        start_end_dist = np.sqrt((np.abs((input_motions[sample_i, 0, :, length-1] - input_motions[sample_i, 0, :, 0])[[0,2]])**2).sum())
        # import pdb;pdb.set_trace()
        start_pose_dist_list.append(start_pose_dist)
        end_pose_dist_list.append(end_pose_dist)
        
        rec_start_pose_dist_list.append(rec_start_pose_dist)
        rec_end_pose_dist_list.append(rec_end_pose_dist)
        
        rec_start_height_dist_list.append(rec_start_height_dist)
        rec_end_height_dist_list.append(rec_end_height_dist)
        
        start_end_pose_dist_list.append(start_end_dist)
        end_input.append(all_motions[sample_i, 0, :, length-1])
        
        # import pdb;pdb.set_trace()
        # add orientation error
        input_orient_start = input_motions[sample_i, 1, :, 0] - input_motions[sample_i, 2, :, 0]
        rec_orient_start = all_motions[sample_i, 1, :, 0] - all_motions[sample_i, 2, :, 0]
        dis_angle_start = calculate_euler_angles(input_orient_start, rec_orient_start)
        
        input_orient_end = input_motions[sample_i, 1, :, length-1] - input_motions[sample_i, 2, :, length-1]
        rec_orient_end = all_motions[sample_i, 1, :, length-1] - all_motions[sample_i, 2, :, length-1]
        
        end_input_orient_pred.append(rec_orient_end)
        end_input_orient.append(input_orient_end)

        dis_angle_end = calculate_euler_angles(input_orient_end, rec_orient_end)
        start_orient_dist_list.append(dis_angle_start) # y-axis
        end_orient_dist_list.append(dis_angle_end)
        # add height error

    start_pose_dist_list = np.stack(start_pose_dist_list)
    end_pose_dist_list = np.stack(end_pose_dist_list)
    rec_start_pose_dist_list = np.stack(rec_start_pose_dist_list)
    rec_end_pose_dist_list = np.stack(rec_end_pose_dist_list)
    rec_start_height_dist_list = np.stack(rec_start_height_dist_list)
    rec_end_height_dist_list = np.stack(rec_end_height_dist_list)
    
    start_orient_dist_list=np.array(start_orient_dist_list)
    end_orient_dist_list=np.array(end_orient_dist_list)
    
    # print('mean: ', start_pose_dist_list.mean(), end_pose_dist_list.mean())
    # print('std: ', start_pose_dist_list.std(), end_pose_dist_list.std()) 


    print('recon_mean: ', rec_start_pose_dist_list.mean(), rec_end_pose_dist_list.mean())
    print('recon_std: ', rec_start_pose_dist_list.std(), rec_end_pose_dist_list.std()) 
    print('recon_orient_mean: ', start_orient_dist_list.mean(), end_orient_dist_list.mean())
    print('recon_orient_std: ', start_orient_dist_list.std(), end_orient_dist_list.std()) 
    print('recon_height_mean: ', rec_start_height_dist_list.mean(), rec_end_height_dist_list.mean())
    print('recon_height_std: ', rec_start_height_dist_list.std(), rec_end_height_dist_list.std()) 
    
    print('save to ', os.path.join(out_path, 'dist_error.txt'))

    # import pdb;pdb.set_trace()

    
    # Debug: visualize the data point and the score;
    
    # import pdb;pdb.set_trace()
    with open(os.path.join(out_path, 'dist_error.txt'), 'w') as fw:
        # fw.write(f'mean: start pose {start_pose_dist_list.mean()} end pose: {end_pose_dist_list.mean()} \n')
        # fw.write(f'std: start pose {start_pose_dist_list.std()} end pose: {end_pose_dist_list.std()} \n' )
        # sorted_indices = np.argsort(end_pose_dist_list) # small to large.
        # sorted_arr = end_pose_dist_list[sorted_indices]
        # sorted_arr_start = start_pose_dist_list[sorted_indices]
        # for i, val in enumerate(sorted_arr):
        #     fw.write(f"Index: {sorted_indices[i]}, Value: end {val}, start {sorted_arr_start[i]} \n")

        fw.write(f"\n-------------------\n")
        fw.write(f'rec_orient_mean: start orient {start_orient_dist_list.mean()} end orient: {end_orient_dist_list.mean()} \n')
        fw.write(f'rec_orient_std: start orient {start_orient_dist_list.std()} end orient: {end_orient_dist_list.std()} \n' )
        
        fw.write(f"\n-------------------\n")
        fw.write(f'rec_height_mean: start height {rec_start_height_dist_list.mean()} end height: {rec_end_height_dist_list.mean()} \n')
        fw.write(f'rec_height_std: start height {rec_start_height_dist_list.std()} end height: {rec_end_height_dist_list.std()} \n' )
        
        # rec
        fw.write(f'rec_mean: start pose {rec_start_pose_dist_list.mean()} end pose: {rec_end_pose_dist_list.mean()} \n')
        fw.write(f'rec_std: start pose {rec_start_pose_dist_list.std()} end pose: {rec_end_pose_dist_list.std()} \n' )

        # rec_sorted_indices = np.argsort(rec_end_pose_dist_list) # small to large.
        rec_sorted_indices = np.arange(len(rec_end_pose_dist_list))
        rec_sorted_arr = rec_end_pose_dist_list[rec_sorted_indices]
        rec_sorted_arr_start = rec_start_pose_dist_list[rec_sorted_indices]
        for i, val in enumerate(rec_sorted_arr):
            fw.write(f"Index: {rec_sorted_indices[i]}, Value: end {val}, start {rec_sorted_arr_start[i]} \n")
        
        # import pdb;pdb.set_trace()
        print('show detailed orientation error\n')
        fw.write('show detailed orientation error \n')
        for i, val in enumerate(start_orient_dist_list):
            fw.write(f"Index: {i}, Value: start {val}, end {end_orient_dist_list[i]} \n")

    use_sorted_index = np.arange(args.num_samples)    

    if save_intermediate_results:
        ### store the input and predict results;
        # import pdb;pdb.set_trace()
        intermediate_npy_path = os.path.join(out_path, 'intermediate_results.npy')
        print(f"saving results file to [{intermediate_npy_path}]")
        # input_motions, all_motions, all_lengths
        np.save(intermediate_npy_path,
                {'motion': all_motions, 'input_motion': input_motions, 'lengths': all_lengths,
                'end_input': end_input, 'end_input_orient': end_input_orient, 'end_input_orient_pred': end_input_orient_pred,
                'use_sorted_index': use_sorted_index,
                })
    
    
    npy_sorted_path = os.path.join(out_path, 'sorted_idx.npy')
    print(f"saving sorted idx results file to [{npy_sorted_path}]")
    np.save(npy_sorted_path, use_sorted_index)
    
    np.save(os.path.join(out_path, 'rec_end_pose_dist_list.npy'), {
        'rec_end_pose_dist_list': rec_end_pose_dist_list,
        'rec_end_height_dist_list': rec_end_height_dist_list,
        'end_orient_dist_list': end_orient_dist_list,
    })

    return use_sorted_index, {
        'rec_end_pose_dist_list': rec_end_pose_dist_list,
        'rec_end_height_dist_list': rec_end_height_dist_list,
        'end_orient_dist_list': end_orient_dist_list,
    }

# def get_split_eval_list(rec_end_pose_dist_list, all_collision, all_collision_ratio, out_path, save_name):
#     pass

def get_split_eval_list(rec_end_pose_dist_list, all_collision, all_collision_ratio, out_path, save_name):
    # Example usage:
    # Assuming rec_end_pose_dist_list, all_collision, and all_collision_ratio are NumPy arrays
    # mean_collision, std_collision, mean_collision_ratio, std_collision_ratio = get_split_eval_list(rec_end_pose_dist_list, all_collision, all_collision_ratio, out_path, save_name)
    all_collision, all_collision_ratio = np.array(all_collision), np.array(all_collision_ratio)
    # import pdb;pdb.set_trace()
    # Initialize lists to store mean and std values for each distance range
    mean_collision_list, std_collision_list = [], []
    mean_collision_ratio_list, std_collision_ratio_list = [], []

    # Define distance ranges
    distance_ranges = [0.1, 0.2, 0.3]

    # Iterate over distance ranges
    for i, distance_range in enumerate(distance_ranges):
        # Select indices for distances within the current range
        if i == 0:
            indices = np.where(rec_end_pose_dist_list <= distance_range)[0]
        else:
            # indices = np.where((rec_end_pose_dist_list > distance_ranges[i-1]) &
            #                    (rec_end_pose_dist_list <= distance_range))[0]
            indices = np.where((rec_end_pose_dist_list <= distance_range))[0]

        # import pdb;pdb.set_trace()
        # Extract collision and collision_ratio values for the selected indices
        selected_collision = all_collision[indices]
        selected_collision_ratio = all_collision_ratio[indices]

        # Calculate mean and std for collision
        mean_collision = np.mean(selected_collision)
        std_collision = np.std(selected_collision)

        # Calculate mean and std for collision_ratio
        mean_collision_ratio = np.mean(selected_collision_ratio)
        std_collision_ratio = np.std(selected_collision_ratio)

        # Append results to the corresponding lists
        mean_collision_list.append(format(mean_collision, '.6f'))
        std_collision_list.append(format(std_collision, '.6f'))
        mean_collision_ratio_list.append(format(mean_collision_ratio, '.6f'))
        std_collision_ratio_list.append(format(std_collision_ratio, '.6f'))


    # Process distances greater than the last range
    indices = np.where(rec_end_pose_dist_list > distance_ranges[-1])[0]
    selected_collision = all_collision[indices]
    selected_collision_ratio = all_collision_ratio[indices]

    # Calculate mean and std for collision
    mean_collision = np.mean(selected_collision)
    std_collision = np.std(selected_collision)

    # Calculate mean and std for collision_ratio
    mean_collision_ratio = np.mean(selected_collision_ratio)
    std_collision_ratio = np.std(selected_collision_ratio)

    # Append results for the last range
    mean_collision_list.append(format(mean_collision, '.6f'))
    std_collision_list.append(format(std_collision, '.6f'))
    mean_collision_ratio_list.append(format(mean_collision_ratio, '.6f'))
    std_collision_ratio_list.append(format(std_collision_ratio, '.6f'))

    # Create a dictionary to store the results
    result_dict = {
        "distance_ranges": distance_ranges + ["inf"],  # "inf" represents "greater than the last range
        "mean_collision": mean_collision_list,
        "std_collision": std_collision_list,
        "mean_collision_ratio": mean_collision_ratio_list,
        "std_collision_ratio": std_collision_ratio_list
    }

    # Save the results to a JSON file
    output_file_path = f"{out_path}/{save_name}.json"
    with open(output_file_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=2)

    return mean_collision_list, std_collision_list, mean_collision_ratio_list, std_collision_ratio_list


def eval_map_collision(eval_func, input_motions, all_motions, all_lengths, model_kwargs, out_path, args):
    ###
    # input_motions: [bs, 1, njoints, 3, seqlen]; GT motion
    # all_motions: [bs, 1, njoints, 3, seqlen]; predicted motion
    # all_lengths: [bs]
    # model_kwargs: dict
    # out_path: str
    # args: args
    pass
    all_loss_list = []
    # for i in range(input_motions.shape[0]):
    for i in range(len(all_lengths)):
        # time step = [1] > [0]
        length=all_lengths[i]
        loss_dict = eval_func(input_motions[i:i+1][..., :length], [1], {'pred_xstart': all_motions[i:i+1][..., :length]}, None, idx=i)
        all_loss_list.append(loss_dict)

    results = {}
    for i in range(len(all_loss_list)):
        for key in all_loss_list[i].keys():
            if key not in results:
                results[key] = [all_loss_list[i][key].item()]
            else:
                results[key].append(all_loss_list[i][key].item())
        
    # save into json file
    import json
    json_path = os.path.join(out_path, 'eval_results_collision.json')
    print(f"saving eval results file to [{json_path}]")
    with open(json_path, 'w') as fw:
        for key in results.keys():
            print(f"{key}: {np.mean(results[key])}")
            fw.write('{}: {}\n'.format(key, np.mean(results[key])))
            for i in range(len(results[key])):
                fw.write('{}: {}\n'.format(i, results[key][i]))
    
    return results
    
def eval_object_collsion(real_world_poses, y, device, return_pose=False, return_obj=False, save_dir=None, all_lengths=None, calculate_sdf=True): # y is the input condition data.
    # real_world_poses: [B, N, J, 3]
    njoints = real_world_poses.shape[-2]

    transform_mat = y['transform_mat']
    # self.obj_transformation[obj_name[0]]
    obj_tranform_trans = y['obj_transform_trans']
    obj_tranform_rotation = y['obj_transform_rotation']
    obj_coord_poses = canonicalize_poses_to_object_space_th(real_world_poses, transform_mat.to(device), obj_tranform_trans.to(device), obj_tranform_rotation.to(device))
    
    if calculate_sdf:
        sdf_data = y['sdf_grid'].float().to(device)
        sdf_grad_data = y['sdf_gradient_grid'].float().to(device)
        sdf_centers = y['sdf_centroid'].float().to(device) # zero
        sdf_scales = y['sdf_scale'].float().to(device)
        sdf_input = torch.cat((sdf_data[:, None], sdf_grad_data.permute((0, 4, 1, 2, 3))), axis=1)

        sdf_feat = query_feature_grid_3D(obj_coord_poses, sdf_input, sdf_centers, sdf_scales)   #[B, d, N]
        # sdf_grad_feat = query_feature_grid_3D(whole_poses, sdf_grad_data, sdf_centers, sdf_scales)   #[B, T, d]
        
        # import pdb;pdb.set_trace()
        bs, nfeats = sdf_feat.shape[0], sdf_feat.shape[1]
        # njoints = 22
        sdf_feat = sdf_feat.reshape(bs, nfeats, -1, njoints).permute(0, 3, 1, 2)
    else:
        sdf_centers = y['sdf_centroid'].float().to(device)
        sdf_feat = None

    if return_pose and not return_obj:
        obj_coord_poses_new = obj_coord_poses.clone() - sdf_centers[:, None, None].repeat((1, obj_coord_poses.shape[1], obj_coord_poses.shape[2], 1))
        return sdf_feat, obj_coord_poses_new

    elif return_pose and return_obj:
        # import pdb;pdb.set_trace()
        obj_coord_poses_new = obj_coord_poses.clone() - sdf_centers[:, None, None].repeat((1, obj_coord_poses.shape[1], obj_coord_poses.shape[2], 1))

        obj_list = []
        for i in range(len(sdf_scales)):
            scale = sdf_scales[i].item()
            grid_min = scale * np.ones(3) * -1
            center = sdf_centers[i].cpu().numpy()
            scene_sdf_volume = sdf_data[i].cpu().numpy()

            dim = scene_sdf_volume.shape[0]
            # numpy process.

            # import pdb;pdb.set_trace()
            voxel_size = np.ones(3) * 2 * scale / dim 

            contact_v, out_v = get_vertices_from_sdf_volume(scene_sdf_volume, voxel_size, grid_min)
            contact_v += center 
            out_v += center

            out_mesh = trimesh.Trimesh(contact_v, process=False)
            
            if save_dir is not None and i % 5 == 0:
                save_sample_poses(obj_coord_poses_new[i, :all_lengths[i], :, :].cpu().numpy(), save_dir, post_fix=f'output_{i}') 
                template_save_fn = os.path.join(save_dir, f'in_sdf_sample_{i}.ply') 
                out_mesh.export(template_save_fn, vertex_normal=False) # export_ply
                
            obj_list.append(out_mesh)

        # sample_num = int(out_v.shape[0] / 1000)
        # sample_out_v = out_v[np.random.choice(out_v.shape[0], sample_num, replace=False)]
        # out_mesh = trimesh.Trimesh(sample_out_v, process=False)
        # template_save_fn = os.path.join(save_dir, 'out_sdf_sample_0.001.ply') 
        # out_mesh.export(template_save_fn, vertex_normal=False) # export_ply
        
        return sdf_feat, obj_coord_poses_new, obj_list

    return sdf_feat
