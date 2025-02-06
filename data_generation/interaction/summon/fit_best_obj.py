import argparse
import json
import math
import os
import random
import time
from tqdm import tqdm

import numpy as np
import open3d as o3d
import trimesh
import pickle as pkl
import torch
from scipy.spatial.transform import Rotation as R

import config

from place_obj_opt import grid_search, optimization
from utils import read_mpcat40, pred_subset_to_mpcat40, estimate_floor_height, read_sequence_human_mesh, merge_meshes, generate_sdf, trimesh_from_o3d, create_o3d_pcd_from_points, write_verts_faces_obj, create_o3d_mesh_from_vertices_faces, align_obj_to_floor

os.environ['PYOPENGL_PLATFORM'] = 'egl' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_type", type=str, default="samp") # samp or amass
    parser.add_argument("--seq_id", type=int, default=None)
    parser.add_argument("--sequence_name", type=str)
    parser.add_argument("--vertices_path", type=str)
    parser.add_argument("--contact_labels_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_probability",
                        action='store_true', 
                        default=False)
    parser.add_argument("--num_objs_save", type=int, default=50)
    parser.add_argument("--num_objs_try", type=int, default=200)
    parser.add_argument("--loss_thres", type=float, default=0.7) # 0.7 for samp 0.5 for amass
    args = parser.parse_args()
    
    sequence_name = args.sequence_name
    random.seed(sequence_name)
    output_dir = args.output_dir

    if args.seq_id is not None: # fit one sequence, get sequence name from list
        list_f = '../priorMDM/dataset/HumanML3D/summon/seq_list_0128.txt'
        with open(list_f, 'r') as f:
            seq_name_list = [s.strip() for s in f.readlines()]
        sequence_name = seq_name_list[args.seq_id]
        args.vertices_path = os.path.join(args.vertices_path, f"{sequence_name}_verts_ds2.npy")
        args.contact_labels_path = os.path.join(args.contact_labels_path, f"{sequence_name}.npy")

    vertices = np.load(open(args.vertices_path, "rb"))
    contact_labels = np.load(open(args.contact_labels_path, "rb"))
    # import pdb; pdb.set_trace()
    
    if args.input_probability:
        contact_labels = np.argmax(contact_labels, axis=-1)
    
    label_names, color_coding_rgb = read_mpcat40()
    # import pdb; pdb.set_trace()
    print(label_names)
    
    contact_labels = contact_labels.squeeze()
    
    # Map contact labels to predicted subset
    vertices_down = []
    contact_labels_mapped = []
    if args.dataset_type == "amass":
        jump_step=8
    elif args.dataset_type == "samp":
        jump_step=16
    for frame in range(contact_labels.shape[0]):
        contact_labels_mapped.append(pred_subset_to_mpcat40[contact_labels[frame]])
        vertices_down.append(vertices[frame * jump_step])
    vertices = np.array(vertices_down)
    contact_labels = np.array(contact_labels_mapped)
    
    # Load fitting parameters
    classes_eps = config.classes_eps
    pcd_down_voxel_size = config.voxel_size
    voting_eps = config.voting_eps
    cluster_min_points = config.cluster_min_points
    if sequence_name in config.params:
        params = config.params[sequence_name]
    else:
        # print("Sequence specific parameters undefined, using default")
        
        
        params = config.params["default"]
    grid_search_contact_weight = params["grid_search_contact_weight"]
    grid_search_pen_thresh = params["grid_search_pen_thresh"]
    grid_search_classes_pen_weight = params["grid_search_classes_pen_weight"]
    lr = params["lr"]
    opt_steps = params["opt_steps"]
    opt_contact_weight = params["opt_contact_weight"]
    opt_pen_thresh = params["opt_pen_thresh"]
    opt_classes_pen_weight = params["opt_classes_pen_weight"]
    
    # Get cuda
    if torch.cuda.is_available():
        # print("Using cuda")
        device = torch.device("cuda")
    else:
        # print("Using cpu")
        device = torch.device("cpu")
    
    
    
    
    # Estimate floor height
    floor_height = estimate_floor_height(vertices, contact_labels)
    # print("Estimated floor height is", floor_height)
    
    
    # Local majority voting
    # print("Performing local majority voting")
    cluster_contact_points = []
    cluster_contact_labels = []
    num_frames = contact_labels.shape[0]
    for obj_c in classes_eps:
        contact_points_class = []
        for frame in range(num_frames):
            contact_points_class.extend(vertices[frame][contact_labels[frame] == obj_c])
        if len(contact_points_class) == 0:
            continue
        contact_points_class = np.array(contact_points_class)
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_points_class)
        contact_pcd = contact_pcd.voxel_down_sample(voxel_size=pcd_down_voxel_size)
        contact_points_class = np.array(contact_pcd.points)
        cluster_contact_points.extend(contact_points_class)
        cluster_contact_labels.extend(np.full((contact_points_class.shape[0],), obj_c))
    cluster_contact_points = np.array(cluster_contact_points)
    cluster_contact_labels = np.array(cluster_contact_labels)
    # import pdb; pdb.set_trace()
    contact_pcd = o3d.geometry.PointCloud()
    if len(cluster_contact_points) == 0:
        print('Exit with no contact point')
        exit(0)
    contact_pcd.points = o3d.utility.Vector3dVector(cluster_contact_points)
    
    
    
    # Cluster all points
    # print("Clustering all points with eps", voting_eps, "...")
    start_time = time.time()
    cluster_labels = np.array(contact_pcd.cluster_dbscan(eps=voting_eps, min_points=cluster_min_points, print_progress=False))
    if cluster_labels.max() == -1:
        cluster_labels = np.zeros_like(cluster_labels)
    max_label = cluster_labels.max()
    # print("Clustering took {0} seconds".format(time.time()-start_time))
    # print("Num clusters", max_label + 1)
    voted_vertices = []
    voted_labels = []
    for label in range(max_label + 1):
        cluster_points = cluster_contact_points[cluster_labels == label]
        if len(cluster_points) < cluster_min_points:
            continue
        majority_label = np.argmax(np.bincount(cluster_contact_labels[cluster_labels == label]))
        # print("Cluster", label, "has", len(cluster_points), "points with majority label of", majority_label, label_names[majority_label])
        voted_vertices.extend(cluster_points)
        voted_labels.extend(np.full(cluster_points.shape[0], majority_label))
    voted_vertices = np.array(voted_vertices)
    voted_labels = np.array(voted_labels)
    voted_vertices = np.expand_dims(voted_vertices, axis=0)
    voted_labels = np.expand_dims(voted_labels, axis=0)
    vertices = voted_vertices
    contact_labels = voted_labels
    
    
    
    # Cluster points by contact label 
    # print("Clustering object points by contact label")
    
    clusters_classes = []
    clusters_points = []
    objects_indices = []
    num_frames = contact_labels.shape[0]
    for obj_c in classes_eps:
        # print("Object class", obj_c, label_names[obj_c])
        contact_points = []
        for frame in range(num_frames):
            contact_points.extend(vertices[frame][contact_labels[frame] == obj_c])
        if len(contact_points) == 0:
            
            continue
        print("Num_points", len(contact_points), obj_c)
        contact_points = np.array(contact_points)
        # import pdb; pdb.set_trace()
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_points)
        contact_pcd = contact_pcd.voxel_down_sample(voxel_size=pcd_down_voxel_size)
        contact_points = np.array(contact_pcd.points)
        # print("After downsampling, have", len(contact_points), "points")
        # print("Clustering with eps", classes_eps[obj_c], "...")
        start_time = time.time()
        cluster_labels = np.array(contact_pcd.cluster_dbscan(eps=classes_eps[obj_c], min_points=cluster_min_points, print_progress=False))
        if cluster_labels.max() == -1:
            cluster_labels = np.zeros_like(cluster_labels)
        max_label = cluster_labels.max()
        # print("Clustering took {0} seconds".format(time.time()-start_time))
        # print("Num clusters", max_label + 1)
        for label in range(max_label + 1):
            clusters_classes.append(obj_c)
            clusters_points.append(contact_points[cluster_labels == label])
            objects_indices.append(label)
        
    
    
    # For each cluster, fit best object
    # Iterate by object class, then by clusters of that class
    print(len(clusters_classes))
    if len(clusters_classes) > 0:
        # Create human SDF
        human_meshes = read_sequence_human_mesh(args.vertices_path)
        merged_human_meshes = merge_meshes(human_meshes)
        grid_dim = 256
        human_sdf_base_path = os.path.join(output_dir, sequence_name, "human")
        if not os.path.exists(human_sdf_base_path):
            os.makedirs(human_sdf_base_path) 
        sdf_path = os.path.join(human_sdf_base_path, "sdf.npy")
        json_path = os.path.join(human_sdf_base_path, "sdf.json")
        if os.path.exists(sdf_path) and os.path.exists(json_path):
            # print("Human SDF already exists, reading from file")
            json_sdf_info = json.load(open(json_path, 'r'))
            centroid = np.asarray(json_sdf_info['centroid'])    # 3
            extents = np.asarray(json_sdf_info['extents'])      # 3
            sdf = np.load(sdf_path)
        else:
            # print("Generating human SDF")
            centroid, extents, sdf = generate_sdf(trimesh_from_o3d(merged_human_meshes), json_path, sdf_path)
        sdf = torch.Tensor(sdf).float().to(device)
        centroid = torch.Tensor(centroid).float().to(device)
        extents = torch.Tensor(extents).float().to(device)
    for i, obj_c in enumerate(clusters_classes):
        cluster_points = clusters_points[i]
        cluster_points_tensor = torch.Tensor(cluster_points).float().to(device)
        obj_idx = objects_indices[i]
        obj_class_str = label_names[obj_c]
        if "chair" in obj_class_str and 'highstool' in sequence_name:
            obj_class_str = "stool"
        if ("chair" in obj_class_str or "bed" in obj_class_str) and ('lie_down' in sequence_name or 'sofa' in sequence_name):
            obj_class_str = "sofa"
        if "chair" in obj_class_str and 'lie_down' in sequence_name:
            obj_class_str = "sofa"
        if ("chair" in obj_class_str or "sofa" in obj_class_str) and 'reebok' in sequence_name:
            obj_class_str = "stool"
        if "chair" == obj_class_str and 'armchair' in sequence_name:
            obj_class_str = "armchair"
        obj_class_path = os.path.join("3D_Future_full", "models", obj_class_str)
        # obj_class_path = os.path.join("pseudo_dataset", obj_class_str)
        # print("Object class", obj_class_str, "Object index", obj_idx, "Num points", cluster_points.shape[0])
        
        cluster_base_path = os.path.join(output_dir, sequence_name, "fit_best_obj_0120_3", obj_class_str, str(obj_idx))
        if not os.path.exists(cluster_base_path):
            os.makedirs(cluster_base_path)
        # Save cluster PCD for visualization
        cluster_pcd_colors = np.zeros_like(cluster_points)
        cluster_pcd_colors += color_coding_rgb[obj_c]
        cluster_pcd = create_o3d_pcd_from_points(cluster_points, cluster_pcd_colors)
        # o3d.io.write_point_cloud(os.path.join(cluster_base_path, "cluster_pcd.ply"), cluster_pcd)
        # Get contact position
        contact_max_x = cluster_points[:,0].max()
        contact_min_x = cluster_points[:,0].min()
        contact_max_y = cluster_points[:,1].max()
        contact_min_y = cluster_points[:,1].min()
        contact_max_z = cluster_points[:,2].max()
        contact_min_z = cluster_points[:,2].min()
        contact_center_x = (contact_max_x + contact_min_x) / 2
        contact_center_y = (contact_max_y + contact_min_y) / 2
        # print("x size", (contact_min_x, contact_max_x))
        # print("y size", (contact_min_y, contact_max_y))
        # print("z size", (contact_min_z, contact_max_z))
        # Info about best fitted object
        best_obj_loss = float("inf")
        best_obj_id = ""
        # For each candidate object
        print('len of all_objects')
        all_objects = os.listdir(obj_class_path)
        random.shuffle(all_objects)
        all_objects = all_objects[:args.num_objs_try]
        success_count = 0
        def fit(obj_dir):
            try:
                obj_path = os.path.join(obj_class_path, obj_dir, "raw_model.obj")
                print("Trying obj at", obj_path)
                obj_mesh = trimesh.load(obj_path)
                obj_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(obj_mesh.vertices), o3d.utility.Vector3iVector(obj_mesh.faces))
                # assert False
                if '6801_bar_stool' in obj_path:
                    rot_mat = obj_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0.))
                    obj_mesh.rotate(rot_mat, center=(0, 0, 0))
                obj_verts = np.array(obj_mesh.vertices)
                obj_faces = np.array(obj_mesh.triangles)
                obj_mesh.compute_vertex_normals()
                save_obj_base_path = os.path.join(cluster_base_path, obj_dir)
                # Align object to floor
                floor_aligned_obj_path = os.path.join(save_obj_base_path, "floor_aligned.obj")
                print("Writing floor aligned object to", floor_aligned_obj_path)
                floor_aligned_verts, align_trans_mat = align_obj_to_floor(obj_verts, obj_faces, None)
                transformed_verts = np.copy(floor_aligned_verts)
                transformed_verts[:, 2] += floor_height
                align_trans_mat[2, 3] += floor_height
                # Transform object to cluster centroid
                obj_max_x = transformed_verts[:,0].max()
                obj_min_x = transformed_verts[:,0].min()
                obj_max_y = transformed_verts[:,1].max()
                obj_min_y = transformed_verts[:,1].min()
                obj_center_x = (obj_max_x + obj_min_x) / 2
                obj_center_y = (obj_max_y + obj_min_y) / 2
                x_transl = contact_center_x - obj_center_x
                y_transl = contact_center_y - obj_center_y
                transformed_verts[:, 0] += x_transl
                transformed_verts[:, 1] += y_transl
                obj_center_x += x_transl
                obj_center_y += y_transl
                obj_max_x += x_transl
                obj_max_y += y_transl
                obj_min_x += x_transl
                obj_min_y += y_transl
                # transformed_obj_path = os.path.join(save_obj_base_path, "transformed.obj")
                # # print("Writing transformed object to", transformed_obj_path)
                # write_verts_faces_obj(transformed_verts, obj_faces, transformed_obj_path)
                # Sample points from centered mesh
                obj_max_z = transformed_verts[:,2].max()
                obj_min_z = transformed_verts[:,2].min()
                # print("x size", (obj_max_x - obj_min_x))
                # print("y size", (obj_max_y - obj_min_y))
                # print("z size", (obj_max_z - obj_min_z))
                x_pts = int(math.ceil((obj_max_x - obj_min_x) * config.pts_per_unit))
                y_pts = int(math.ceil((obj_max_y - obj_min_y) * config.pts_per_unit))
                z_pts = int(math.ceil((obj_max_z - obj_min_z) * config.pts_per_unit))
                num_sample_points = x_pts * y_pts * z_pts
                assert num_sample_points < 100000, f"Too many points to sample!, {num_sample_points}"
                # print("Sampling", num_sample_points, "points")
                centered_verts = np.copy(transformed_verts)
                centered_verts[:, 0] -= obj_center_x
                centered_verts[:, 1] -= obj_center_y
                obj_pcd = create_o3d_mesh_from_vertices_faces(centered_verts, obj_faces).sample_points_poisson_disk(number_of_points=num_sample_points)
                obj_pcd = obj_pcd.voxel_down_sample(voxel_size=config.voxel_size)
                obj_pcd.estimate_normals()
                obj_points_centered = np.array(obj_pcd.points)
                obj_vert_normals = np.array(obj_pcd.normals)
                if cluster_points_tensor[:, 2].max() > obj_points_centered[:, 2].max() * 1.2:
                    return None, None, None, None, None, None, None, None, None, None, None, None
                if cluster_points_tensor[:, 2].min() < obj_points_centered[:, 2].max() * 0.8 / 4:
                    return None, None, None, None, None, None, None, None, None, None, None, None
                print("After downsampling, have", len(obj_points_centered), "points")
                # Grid search
                # print("Grid Searching...")
                start_time = time.time()
                grid_best_loss, grid_best_rot_deg, grid_best_transl_x, grid_best_transl_y, grid_best_points, grid_best_scale = grid_search(
                    obj_c,
                    obj_points_centered,
                    obj_vert_normals,
                    obj_center_x, obj_center_y,
                    obj_min_x, obj_min_y,
                    obj_max_x, obj_max_y,
                    cluster_points_tensor,
                    contact_min_x, contact_min_y,
                    contact_max_x, contact_max_y,
                    sdf, centroid, extents,
                    grid_search_contact_weight,
                    grid_search_pen_thresh,
                    grid_search_classes_pen_weight,
                    device
                )
                print("Grid search took {0} seconds".format(time.time()-start_time))
                print("Best loss", grid_best_loss)
                print("Best Rotation in degrees", grid_best_rot_deg, "Best x translation", grid_best_transl_x, "Best y translation", grid_best_transl_y, "Best grid scale", grid_best_scale)
                r = R.from_euler('XYZ', [0, 0, grid_best_rot_deg], degrees=True)
                candidate_verts_centered = r.apply(centered_verts)
                candidate_verts = np.copy(candidate_verts_centered)
                candidate_verts *= grid_best_scale
                candidate_verts[:, 0] += obj_center_x + grid_best_transl_x
                candidate_verts[:, 1] += obj_center_y + grid_best_transl_y
                # grid_search_best_path = os.path.join(save_obj_base_path, "grid_search_best.obj")
                # print("Writing best grid search result to", grid_search_best_path)
                # write_verts_faces_obj(candidate_verts, obj_faces, grid_search_best_path)
                json_dict = {}
                json_dict["loss"] = grid_best_loss
                json_dict["rot_deg"] = grid_best_rot_deg
                json_dict["transl_x"] = grid_best_transl_x
                json_dict["transl_y"] = grid_best_transl_y
                # json.dump(json_dict, open(os.path.join(save_obj_base_path, "grid_search_best.json"), 'w'))
                grid_pcd_colors = np.zeros_like(grid_best_points)
                grid_pcd_colors += color_coding_rgb[obj_c]
                grid_pcd = create_o3d_pcd_from_points(grid_best_points, grid_pcd_colors)
                # o3d.io.write_point_cloud(os.path.join(save_obj_base_path, "grid_search_best.ply"), grid_pcd)
                # Optimization
                grid_center_x = obj_center_x + grid_best_transl_x
                grid_center_y = obj_center_y + grid_best_transl_y
                # print("Optimizing...")
                start_time = time.time()
                best_loss, best_rot, best_transl_x, best_transl_y, best_points, best_scale, best_scale_z, best_pen_loss = optimization(
                    obj_c,
                    obj_points_centered,
                    obj_vert_normals,
                    grid_center_x, grid_center_y,
                    grid_best_rot_deg,
                    grid_best_scale,
                    1.,
                    cluster_points_tensor,
                    contact_min_x, contact_min_y,
                    contact_max_x, contact_max_y,
                    0.6, 1.4,
                    0.5, 1.0,
                    sdf, centroid, extents,
                    opt_contact_weight,
                    opt_pen_thresh,
                    opt_classes_pen_weight,
                    lr, opt_steps,
                    device
                )
                print("Optimization took {0} seconds".format(time.time()-start_time))
                print("Best loss", best_loss)
                print("Best Rotation in degrees", best_rot/math.pi*180, "Best x translation", best_transl_x, "Best y translation", best_transl_y, "best scale factor", best_scale, "best scale z", best_scale_z)
                r = R.from_euler('XYZ', [0, 0, best_rot], degrees=False)
                opt_obj_verts = candidate_verts_centered.copy()
                opt_obj_verts = opt_obj_verts*best_scale
                opt_obj_verts[:, 2] *= best_scale_z
                opt_obj_verts = r.apply(opt_obj_verts)
                opt_obj_verts[:, 0] += grid_center_x + best_transl_x
                opt_obj_verts[:, 1] += grid_center_y + best_transl_y
                print(opt_obj_verts.max(axis=0), best_points.max(axis=0))
                # import pdb; pdb.set_trace()
                return best_loss, best_rot + grid_best_rot_deg/180*np.pi, grid_center_x + best_transl_x, grid_center_y + best_transl_y, best_scale, best_scale_z, opt_obj_verts, obj_faces, save_obj_base_path, best_points, align_trans_mat, best_pen_loss
            except Exception as error:
                print("An error occurred in fitting this object", error)
                return None, None, None, None, None, None, None, None, None, None, None, None
        
        priority_queue = []
        # only save the best args.max_save objects, delete objects that poped from the queue
        import multiprocessing as mp
        pool = mp.Pool(processes=1)
        # all_object_data = []
        # for object_dir in tqdm(all_objects):
        #     obj_data = o3d.io.read_triangle_mesh(os.path.join(obj_class_path, object_dir, "raw_model.obj"))
        #     all_object_data.append(obj_data)
        for fit_result in tqdm(map(fit, all_objects), total=len(all_objects)):
            best_loss, best_rot, best_transl_x, best_transl_y, best_scale, best_scale_z, opt_obj_verts, obj_faces, save_obj_base_path, best_points, align_trans_mat, best_pen_loss = fit_result
            if best_loss is None:
                continue
            if best_loss < args.loss_thres:
                priority_queue.append((best_loss, save_obj_base_path))
                if len(priority_queue) > args.num_objs_save:
                    priority_queue.sort(key=lambda x: x[0])
                    worst_loss, worst_save_obj_base_path = priority_queue.pop()
                    if os.path.exists(worst_save_obj_base_path):
                        import shutil
                        print('removing worst save obj', worst_save_obj_base_path)
                        shutil.rmtree(worst_save_obj_base_path)
                    if worst_save_obj_base_path == save_obj_base_path:
                        continue
                if not os.path.exists(save_obj_base_path):
                    os.makedirs(save_obj_base_path)
                    opt_best_path = os.path.join(save_obj_base_path, "opt_best.obj")
                    print("Writing best optimization result to", opt_best_path)
                    write_verts_faces_obj(opt_obj_verts, obj_faces, opt_best_path)
                    json_dict = {}
                    json_dict["loss"] = best_loss
                    json_dict["rot_deg"] = best_rot/math.pi*180
                    json_dict["transl_x"] = best_transl_x
                    json_dict["transl_y"] = best_transl_y
                    json_dict["scale"] = best_scale
                    json_dict["scale_z"] = best_scale_z
                    json_dict["pen_loss"] = best_pen_loss
                    scale_mat = np.eye(4)
                    scale_mat[:3, :3] *= best_scale
                    scale_mat[2, 2] *= best_scale_z
                    rot_mat = np.eye(4)
                    rot_mat[:3, :3] = R.from_euler('XYZ', [0, 0, best_rot], degrees=False).as_matrix()
                    transl_mat = np.eye(4)
                    transl_mat[:2, 3] = [best_transl_x, best_transl_y]
                    transform = np.matmul(transl_mat, np.matmul(rot_mat, np.matmul(scale_mat, align_trans_mat)))
                    print(json_dict)
                    json.dump(json_dict, open(os.path.join(save_obj_base_path, "opt_best.json"), 'w'))
                    # save pickle
                    json_dict["align_trans_mat"] = align_trans_mat
                    json_dict["transform"] = transform.tolist()
                    with open(os.path.join(save_obj_base_path, "opt_best.pkl"), 'wb') as f:
                        pkl.dump(json_dict, f)
                    opt_pcd_colors = np.zeros_like(best_points)
                    opt_pcd_colors += color_coding_rgb[obj_c]
                    opt_pcd = create_o3d_pcd_from_points(best_points, opt_pcd_colors)
                    o3d.io.write_point_cloud(os.path.join(save_obj_base_path, "opt_best.ply"), opt_pcd)
        priority_queue.sort(key=lambda x: x[0])
        
        # print("Best fitted object has ID", priority_queue[0][1])
        json_dict = {}
        if len(priority_queue) > 0:
            json_dict["best_obj_id"] = priority_queue[0][1]
            json_dict["best_obj_loss"] = priority_queue[0][0]
            json_dict["saved_objs"] = [x[1] for x in priority_queue]
            json_dict["saved_loss"] = [x[0] for x in priority_queue]
        json.dump(json_dict, open(os.path.join(cluster_base_path, "best_obj_id.json"), 'w'))
        
