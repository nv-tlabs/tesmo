import sys
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loaders.humanml.utils.utils_threeD_front import export_scene
import trimesh
import os
import numpy as np
import pickle 
from glob import glob
from pyquaternion import Quaternion as Q
from argparse import ArgumentParser
import joblib, cv2
from visualize.humor_viz.utils import viz_smpl_seq, create_video

# import pdb; pdb.set_trace=lambda:None

def load_scene_info(input_path):
    input_all_data = pickle.load(open(input_path, 'rb'))
    name_list = input_all_data['name_list']
    length_list = input_all_data['length_list']
    mean = input_all_data['mean']
    std = input_all_data['std']
    data_dict = input_all_data['data_dict']
    
    return name_list, length_list, mean, std, data_dict

def load_3D_front_mesh(scene_folder, scene_id):
    input_dir = f'{scene_folder}/{scene_id}'

    meshes_name_list = sorted(glob(os.path.join(input_dir, '*.obj')))
    meshes_list = []
    
    for one in meshes_name_list:
        new_mesh = trimesh.load_mesh(one, matrial=True, texture=True, process=False)
        meshes_list.append(new_mesh)

    return meshes_list


def load_meshes(input_dir):
    obj_list = sorted(glob(os.path.join(input_dir, '*.obj')))
    all_vertices = []
    all_mesh = []
    for one in obj_list:
        mesh = trimesh.load_mesh(one, input_process=False)
        all_mesh.append(mesh)
        vertices = mesh.vertices
        all_vertices.append(vertices)
    
    return all_mesh, np.stack(all_vertices) 

def render_body_and_scene(body_mesh, out_path_result, body_joint=None, obj_mesh=None, fps=30, only_top=False, only_side=True, 
            render_body_flag=True, render_ground=True, single_obj=True, save_mesh=False): # merge several meshes into one.
    transfrom_mat = np.array([[[1, 0, 0], [0, 0, 1], [0, 1, 0]]])
    
    if body_mesh is not None:
        render_body = []
        for obj in body_mesh:
            obj.vertices = np.matmul(obj.vertices[None], transfrom_mat)[0]
            obj.faces = obj.faces[..., [2,1,0]]
            render_body.append(obj)

    if obj_mesh is not None:
        static_mesh = []
        for obj in obj_mesh:
            obj.vertices = np.matmul(obj.vertices[None], transfrom_mat)[0]
            obj.faces = obj.faces[..., [2,1,0]]
            static_mesh.append(obj)
        render_ground = False
    else:
        render_ground = True

    
    if single_obj:
        bev_camera = [0., 0., 3.0]
        side_camera = [0.0, 2.4, 0.9]
        imw=480
        imh=480
    else:
        bev_camera = [1, 3, 1.0]
        side_camera = [0.0, 4, -6]
        imw = 480 * 2
        imh = 240 * 2
    
    # add side orientation.

    cur_offscreen = out_path_result is not None
    body_alpha = 1.0
    
    # default_cam_pose = trimesh.transformations.rotation_matrix(np.radians(-0), (0, 0, 1))[:3, :3]
    viz_smpl_seq(render_body,
                imw=imw, imh=imh, fps=fps,
                render_body=render_body_flag,
                body_color='pink',
                render_joints=None,
                render_skeleton=None,
                render_ground=render_ground,
                contacts=None,
                joints_seq=None,
                points_seq = None,
                body_alpha=body_alpha,
                use_offscreen=cur_offscreen,
                out_path=out_path_result,
                wireframe=False,
                RGBA=False,
                static_meshes=static_mesh,
                follow_camera=False,
                cam_offset=bev_camera,
                joint_color=[ 0.0, 1.0, 0.0 ],
                point_color=[0.0, 0.0, 1.0],
                skel_color=[0.5, 0.5, 0.5],
                joint_rad=0.015,
                point_rad=0.005,
                # cam_rot = default_cam_pose
        )
    
    
    if cur_offscreen:
        create_video(out_path_result + '/frame_%08d.' + '%s' % ('png'), out_path_result + '.mp4', 30)
        print(out_path_result + '.mp4')

        


def transform_body_to_scene_coordinate_system(body_mesh_list, transform_vector, motion_scale_rt):
    new_body_list = []
    for one_mesh in body_mesh_list:
        # import pdb;pdb.set_trace()
        rot_angle = np.radians(-transform_vector[3]) 
        rotation_matrix = trimesh.transformations.rotation_matrix(rot_angle, [0, 1, 0])
        one_mesh.apply_transform(rotation_matrix)
        trans_x = (transform_vector[0]/motion_scale_rt-6.2)*motion_scale_rt
        trans_z = (transform_vector[2]/motion_scale_rt-6.2)*motion_scale_rt
        inverse_xyz = np.array([trans_x, 0, trans_z])
        one_mesh.apply_translation(inverse_xyz) 
        new_body_list.append(one_mesh) 
        # print(-transform_vector[3], inverse_xyz)
    return new_body_list


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--input_pkl", default='True', type=str, help="")
    group.add_argument("--align_dir", default='', type=str, help="")
    group.add_argument("--sub_priormdm_dir", default='True', type=str, help="")
    group.add_argument("--save_dir", default='locomotion_visualize', type=str, help="")
    group.add_argument("--root_dir", default='', type=str, help="the save dir for the 2D trajectory generation.")
    group.add_argument("--sample_id", default=0, type=int, help="the object idx for each motion.")
    group.add_argument("--motion_id", default=0, type=int, help="the motion idx.")

    return parser.parse_args()

if __name__ == '__main__':

    parser = ArgumentParser()
    args = add_base_options(parser)
    input_scene_info = args.input_pkl
    align_data_dir = args.align_dir
    root_dir = args.root_dir
    sample_id = args.sample_id
    sub_priormdm = args.sub_priormdm_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    input_dir = f'{root_dir}/{sub_priormdm}/sample{sample_id}_rep00_obj'

    name_list, length_list, mean, std, data_dict = load_scene_info(input_scene_info)
    print(name_list)

    input_name, scene_idx, flip_flag = name_list[sample_id].split('_')
    # assert int(flip_flag) == 0
    
    npy_path = f'{root_dir}/results.npy'
    result = np.load(npy_path, allow_pickle=True).item()
    root_motions = result['motion'][sample_id]
    root_length = result['lengths'][sample_id]
    root_motions = np.array([root_motions[0, :, root_i] for root_i in range(root_motions.shape[-1])])
    
    # human bodies ->  original scenes.
    # align information.
    id_name = os.path.join(align_data_dir, f'{input_name}.npy')
    align_motion_npy = np.load(id_name, allow_pickle=True) 
    align_info = align_motion_npy[int(scene_idx)]

    id_name_store = os.path.join(align_data_dir+'_store', f'{input_name}/{name_list[sample_id]}.pkl')
    align_motion_store = joblib.load(id_name_store)
    motion_scale_rt = align_motion_store['motion_scale_rt']
    transform_vector = align_motion_store['transform_vector']

    # get the scene.
    scene_id = align_info['scene'].split('_')[-1]
    scene_folder = '/workspace/SHADE/threed_front_livingroom_mesh' # change it to your 3D-FRONT folder
    scene_mesh_list = load_3D_front_mesh(scene_folder, scene_id)
    print(f'load scene: {scene_folder}/{scene_id}')
    
    new_scene_mesh_list = []
    for mesh in scene_mesh_list:
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, \
        vertex_colors=np.array([[255,0,0]]).repeat(mesh.vertices.shape[0]))
        new_mesh.apply_scale(motion_scale_rt)
        
        if int(flip_flag)==1:
            mirror_matrix = np.diag([-1, 1, 1, 1])
            new_mesh.apply_transform(mirror_matrix)
        new_scene_mesh_list.append(new_mesh)

    all_mesh, motion = load_meshes(input_dir)
    new_body_mesh_list = []
    height_offset = motion.reshape(-1, 3).min(0)[1]
    down_sample = 1
    for mesh_i, mesh in enumerate(all_mesh):
        if mesh_i % down_sample == 0:
            mesh.vertices[:, 1] = mesh.vertices[:, 1] - mesh.vertices[:, 1].min()
            mesh.vertices[:, -1] *= -1 # z is flip
            mesh.faces = mesh.faces[:, [0, 2, 1]]

            if int(flip_flag)==1:
                mirror_matrix = np.diag([-1, 1, 1, 1])
                mesh.apply_transform(mirror_matrix)
            new_body_mesh_list.append(mesh)
    new_body_mesh_list = transform_body_to_scene_coordinate_system(new_body_mesh_list, transform_vector, motion_scale_rt)
    
    
    scene_body_mesh = trimesh.util.concatenate(new_scene_mesh_list + new_body_mesh_list)
    scene_body_mesh.export(f'{save_dir}/scene_{sample_id}.obj')
    # import pdb;pdb.set_trace()
    
    vid_dir = f'{save_dir}/sample{sample_id}_rep00_obj'
    render_body_and_scene(new_body_mesh_list, vid_dir, single_obj=False,
                          body_joint=None,  obj_mesh=new_scene_mesh_list, 
                          only_top=False, render_body_flag=True)


    

    



