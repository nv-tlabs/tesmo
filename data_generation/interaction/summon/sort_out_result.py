import glob
import json
import pickle as pkl
import open3d as o3d
from tqdm import tqdm
import trimesh
import numpy as np
import sys
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return (pcd_down, pcd_fpfh)

def o3d_ransac(src, dst):
    
    voxel_size = 0.01
    distance_threshold = 1.5 * voxel_size

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # print('Downsampling inputs')
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    # print('Running RANSAC')
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
    )

    return result.transformation

def get_transform(src_file, tgt_file, scale, scale_z, prefix):
    mesh1 = trimesh.load(tgt_file)
    mesh_norm = trimesh.load(src_file)
    mesh_norm.vertices *= scale
    mesh_norm.vertices[:, 1] *= scale_z
    mesh_norm.export(f'/tmp/{prefix}_source.ply')
    mesh1.export(f'/tmp/{prefix}_target.ply')
    source = o3d.io.read_point_cloud(f'/tmp/{prefix}_source.ply')
    target = o3d.io.read_point_cloud(f'/tmp/{prefix}_target.ply')
    # v_s = np.array(source.points)
    # v_t = np.array(target.points)
    # v_s_center = np.mean(v_s, axis=0)
    # v_t_center = np.mean(v_t, axis=0)
    # dis_s = np.linalg.norm(v_s_center - v_s, axis=-1).max()
    # dis_t = np.linalg.norm(v_t_center - v_t, axis=-1).max()
    # scale = dis_t / dis_s
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    trans_init = o3d_ransac(source, target)
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=0.00001, relative_rmse=0.00001, max_iteration=100
    )
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source,
        target,
        0.1,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=icp_criteria
    )
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale
    scale_matrix[:3, 1] *= scale_z
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = reg_p2l.transformation[:3, :3]
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = reg_p2l.transformation[:3, 3]
    # if reg_p2l.fitness < 0.999 and scale_z < 1.0:
    #     mesh_norm.export(f'{prefix}_source.obj')
    #     mesh1.export(f'{prefix}_target.obj')
        # import pdb; pdb.set_trace()
    return reg_p2l.transformation, scale_matrix, rotation_matrix, translation_matrix, reg_p2l.fitness

if __name__ == '__main__':
    threeD_FUTURE_PATH = '' # TODO: set it here !
    SUMMON_DOWNLOAD_PATH = '' # TODO: git clone summon and set it here !

    prefix = sys.argv[1]
    src_type = sys.argv[2]
    best_obj_json = glob.glob(f'{SUMMON_DOWNLOAD_PATH}/{prefix}*/fit_best_obj_0120_3/*/*/best_obj_id.json')
    result = dict()
    print(f'to save at {SUMMON_DOWNLOAD_PATH}/{prefix}_{src_type}.pkl')
    for json_file in tqdm(best_obj_json):
        seq_name = json_file.split('/')[-5]
        result[seq_name] = []
        with open(json_file, 'r') as f:
            data = json.load(f)
        if 'saved_objs' not in data:
            continue
        saved_objs = data['saved_objs']
        saved_loss = data['saved_loss']
        for obj, loss in zip(saved_objs, saved_loss):
            try:
                obj_name = obj.split('/')[-1]
                target_obj = json_file.replace('best_obj_id.json', f'{obj_name}/opt_best.obj')
                info_pkl = json_file.replace('best_obj_id.json', f'{obj_name}/opt_best.pkl')
                with open(info_pkl, 'rb') as f:
                    obj_data = pkl.load(f)
                obj_data['src_file']=f'{threeD_FUTURE_PATH}/{obj_name}/{src_type}_model.obj'
                obj_data['tgt_file']=target_obj
                obj_data['contact_loss'] = obj_data['loss'] - obj_data['pen_loss']
                obj_data['transform'], obj_data['scale'], obj_data['rotation'], obj_data['translation'], fitness = get_transform(obj_data['src_file'], obj_data['tgt_file'], obj_data['scale'], obj_data['scale_z'], prefix)
                # if obj_data['scale_z'] != 1.0:
                #     import pdb; pdb.set_trace()
                if fitness < 0.9:
                    print(fitness, obj_data['scale_z'])
                else:
                    obj_data.pop('scale_z')
                    result[seq_name].append(obj_data)
            except:
                print('err')
        result[seq_name].sort(key=lambda x: x['loss'])
    with open(f'{SUMMON_DOWNLOAD_PATH}/{prefix}_{src_type}.pkl', 'wb') as f:
        pkl.dump(result, f)
    for seq_name, objs in result.items():
        print(seq_name)
        for obj in objs:
            print(obj['tgt_file'], obj['loss'], obj['pen_loss'], obj['contact_loss'])
            mesh = o3d.io.read_triangle_mesh(obj['src_file'])
            transform = np.eye(4)
        print()