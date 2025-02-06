import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mesh_to_sdf
import pickle
from visualize.humor_viz.utils import viz_smpl_seq, create_video
import numpy as np
import trimesh
from tqdm import tqdm
import copy
import torch
from visualize.vis_utils import npy2obj

def get_bodies(out_path, i, body_joints, save_body_mesh=False, same_transl=False, export_perframe_mesh=False):
    # save as an object. 
    body_mesh_pkl_path = os.path.join(out_path, 'body_mesh_%d.pkl' % i)
    # ! they move the first body to the original position canoniclize the first body.
    body_mesh = npy2obj(body_joints, sample_idx=0, rep_idx=0, device=0, cuda=True, input_poses=True)
    with open(body_mesh_pkl_path, 'wb') as fw:
        pickle.dump(body_mesh, fw)
    
    if same_transl:
        # * xz translation
        body_mesh.vertices[:, :, [0,2], :] += torch.from_numpy(np.tile(body_joints[0, 0, [0,2]][None, None, :, None], (body_mesh.vertices.shape[0], body_mesh.vertices.shape[1], 1, body_mesh.vertices.shape[-1]))).float()
    
    if export_perframe_mesh:
        for i in tqdm(range(len(body_mesh.vertices[0].permute(2,0,1)))):
            mesh = body_mesh.get_trimesh(0, i)
            mesh.export(os.path.join(out_path, f'frame_{i:03d}.obj'))

    return body_mesh

def load_transform_pkl():
    obj_transformation_pkl = 'dataset/SAMP_interaction/SAMP_objects/chair_all.pkl'

    if not os.path.exists(obj_transformation_pkl):
        obj_transformation_pkl = 'dataset/SAMP_interaction/SAMP_objects/chair_all.pkl'
    assert os.path.exists(obj_transformation_pkl)

    with open(obj_transformation_pkl, 'rb') as f:
        obj_transformation = pickle.load(f)

    return obj_transformation

def get_original_name_scale(obj_name, motion_id, obj_transformation):
    for one in obj_transformation:
        # print(one['tgt_file'])
        if obj_name in one['tgt_file'] and motion_id in one['tgt_file']:
            return one['src_file'], one['scale']

def get_original_motion_obj_name(o_p):
    if 'summon' in o_p:
        # * this is for summon 
        motion_name = o_p.split('/')[-6]
        obj_id = o_p.split('/')[-2]
    else:
        motion_name = o_p.split('/')[-3]
        obj_id = o_p.split('/')[-1].split('.')[0]
    return motion_name, obj_id

def transform_xz_plane_to_xy_plane(vertices, faces=None):
    transfrom_mat = np.array([[[1, 0, 0], [0, 0, 1], [0, 1, 0]]]) 
    if len(vertices.shape) == 2:
        new_vertices = np.matmul(vertices[None], transfrom_mat)[0]
    else:
        new_vertices = np.matmul(vertices, transfrom_mat)
    if faces is not None:
        new_faces = new_faces[..., [2,1,0]]
        return new_vertices, new_faces
    else:
        return new_vertices


def render_body_and_scene(body_mesh, out_path_result, body_joint=None, obj_mesh=None, fps=30, only_top=False, only_side=True, 
            render_body_flag=True, render_ground=True, single_obj=True, save_mesh=False): # merge several meshes into one.
    transfrom_mat = np.array([[[1, 0, 0], [0, 0, 1], [0, 1, 0]]])
    
    if body_joint is not None:
        body_joint = np.matmul(body_joint, transfrom_mat)

    if body_mesh is not None:
        import pdb;pdb.set_trace()
        body_mesh.vertices[0] = torch.from_numpy(np.matmul(body_mesh.vertices[0].permute(2, 0, 1).numpy(), transfrom_mat).transpose(1, 2, 0)).float() # a sequence of body;
        min_height = body_mesh.vertices[0].permute(2, 0, 1).reshape(-1, 3).min(0)[0][2].item()
        body_mesh.vertices[:, :, 2] -= min_height
        # min_height = 0.0
        body_mesh_front = copy.deepcopy(body_mesh)
        body_mesh_side2 = copy.deepcopy(body_mesh)

        for angle, b_mesh in zip([90, -90], [body_mesh_side2, body_mesh]):
            # add side rotation.
            angle = np.radians(angle)
            # Define the 3D rotation matrix around the Z-axis
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])[None]
            # Perform the rotation
            b_mesh.vertices[0] = torch.from_numpy(np.matmul(b_mesh.vertices[0].permute(2, 0, 1).numpy(), rotation_matrix).transpose(1, 2, 0)).float() # a sequence of body;
    else:
        body_mesh_front = None
        body_mesh_side2 = None

    if obj_mesh is not None:
        obj_mesh.vertices = np.matmul(obj_mesh.vertices[None], transfrom_mat)[0]
        obj_mesh.faces = obj_mesh.faces[..., [2,1,0]]
        # obj_mesh.vertices[:, 2] -= min_height
        obj_mesh_front = copy.deepcopy(obj_mesh)
        obj_mesh_side2 = copy.deepcopy(obj_mesh)
        for angle, o_mesh in zip([90, -90], [obj_mesh_side2, obj_mesh]):
            # add side rotation.
            angle = np.radians(angle)
            # Define the 3D rotation matrix around the Z-axis
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])[None]
            o_mesh.vertices = np.matmul(o_mesh.vertices[None], rotation_matrix)[0]

    else:
        obj_mesh_front = None
        obj_mesh_side2 = None
    
    if body_joint is not None:
        # import pdb;pdb.set_trace()
        body_joints_side = copy.deepcopy(body_joint)
        body_joints_side2 = copy.deepcopy(body_joint)
        # import pdb;pdb.set_trace()
        
        angle = 90
        # add side rotation.
        angle = np.radians(angle)
        # Define the 3D rotation matrix around the Z-axis
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0], [0, 0, 1]])[None]
        body_joints_side2 = np.matmul(body_joints_side2, rotation_matrix)

        angle = -90
        # add side rotation.
        angle = np.radians(angle)
        # Define the 3D rotation matrix around the Z-axis
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])[None]
        body_joints_side = np.matmul(body_joints_side, rotation_matrix)

    else:
        body_joints_side = None
        body_joints_side2 = None
        
    
    if single_obj:
        bev_camera = [0., 0., 3.0]
        side_camera = [0.0, 2.4, 0.9]
        imw=480
        imh=480
    else:
        bev_camera = [0.0, 0.0, 6.0]
        side_camera = [0.0, 2.8, 0.9]
        imw = 480 * 2
        imh = 240 * 2
    

        # if body_joint is not None:
        #     viz_joints = body_joint
        # else:
        #     viz_joints=None
    # add side orientation.
    for tmp_i, mesh, save_path, body_mesh_obj, viz_joints in zip([0, 1, 2], [obj_mesh_front, obj_mesh, obj_mesh_side2], \
        [out_path_result, out_path_result+'_side', out_path_result+'_side2'], [body_mesh_front, body_mesh, body_mesh_side2], \
        [body_joint, body_joints_side, body_joints_side2]):

        body_alpha = 1.0
        # imw = 720
        # imh = 720
        
        # import pdb;pdb.set_trace()

        points=None
        cur_offscreen = out_path_result is not None
        # TODO: visualize 3D meshes as well.  
        print(save_path)

        if body_mesh_obj is not None:
            # import pdb;pdb.set_trace()
            render_body = []
            b_faces = body_mesh_obj.faces[..., [2,1,0]]
            for i in range(body_mesh_obj.num_frames):
                b_vertices = body_mesh_obj.get_vertices(0, i)
                render_body.append(trimesh.Trimesh(vertices=b_vertices, faces=b_faces))
        else:
            render_body = None

        if tmp_i == 0 and save_mesh:
            # todo: save the scene mesh and bodies.
            # import pdb;pdb.set_trace()
            sample_bodies = render_body[0:-1:10]
            sample_bodies = trimesh.util.concatenate(sample_bodies)
            sample_bodies.export(save_path + 'all_bodies.obj')
            # mesh.export(save_path + 'scene.obj')

        if not only_side and tmp_i == 0: # add a top view.
            default_cam_pose = trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1))
            # self.default_cam_pose = np.dot(trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)), self.default_cam_pose)
            static_mesh = [mesh] if mesh is not None else None
            viz_smpl_seq(render_body, 
                        imw=imw, imh=imh, fps=fps,
                        render_body=render_body_flag,
                        body_color='pink',
                        render_joints=viz_joints is not None,
                        render_skeleton=viz_joints is not None and cur_offscreen,
                        render_ground=render_ground,
                        contacts=None,
                        joints_seq=viz_joints,
                        points_seq = points,
                        body_alpha=body_alpha,
                        use_offscreen=cur_offscreen,
                        out_path=save_path+'_top',
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
                        cam_rot = default_cam_pose[:3, :3],
                        # camera_intrinsics = [imw, imh, imw/2, imh/2]
                )

            if only_top:
                create_video(out_path_result + '_top/frame_%08d.' + '%s' % ('png'), out_path_result + '_top.mp4', 30)
                print(out_path_result + '_top.mp4')
                os.system('rm -rf %s' % (out_path_result + '_top'))
                # sys.exit()
                return 0

        if only_side and tmp_i >= 2:
            continue

        # import pdb;pdb.set_trace()
        if mesh is not None:
            static_mesh = [mesh]
        else:
            static_mesh = None
        viz_smpl_seq(render_body,
                        imw=imw, imh=imh, fps=fps,
                        render_body=render_body_flag,
                        body_color='pink',
                        render_joints=viz_joints is not None,
                        render_skeleton=viz_joints is not None and cur_offscreen,
                        render_ground=render_ground,
                        contacts=None,
                        joints_seq=viz_joints,
                        points_seq = points,
                        body_alpha=body_alpha,
                        use_offscreen=cur_offscreen,
                        out_path=save_path,
                        wireframe=False,
                        RGBA=False,
                        static_meshes=static_mesh,
                        follow_camera=True,
                        cam_offset=side_camera,
                        joint_color=[ 0.0, 1.0, 0.0 ],
                        point_color=[0.0, 0.0, 1.0],
                        skel_color=[0.5, 0.5, 0.5],
                        joint_rad=0.015,
                        point_rad=0.005,
                        # default_cam_pose = default_cam_pose
                )
        
    if cur_offscreen:
        create_video(out_path_result + '/frame_%08d.' + '%s' % ('png'), out_path_result + '.mp4', 30)
        create_video(out_path_result + '_side/frame_%08d.' + '%s' % ('png'), out_path_result + '_side.mp4', 30)
        create_video(out_path_result + '_side2/frame_%08d.' + '%s' % ('png'), out_path_result + '_side2.mp4', 30)
        create_video(out_path_result + '_top/frame_%08d.' + '%s' % ('png'), out_path_result + '_top.mp4', 30)
        
        # merge these two video together
        # os.system('ffmpeg -i %s -i %s -i %s -filter_complex hstack -y %s' % (out_path_result + '.mp4', out_path_result + '_side.mp4', out_path_result + '_top.mp4', out_path_result + '_merge.mp4'))
        # os.system('ffmpeg -i %s -i %s -i %s -filter_complex  hstack=inputs=3 -y %s' % (
        if only_side:
            os.system('ffmpeg -i %s -i %s  -filter_complex hstack -y %s' % (out_path_result + '.mp4', \
                    out_path_result + '_side.mp4', out_path_result + '_merge.mp4'))

        elif not only_top:
            os.system('ffmpeg -i %s -i %s -i %s -i %s -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack" -c:v libx264 -y %s' % (
                    out_path_result + '.mp4',
                    out_path_result + '_side.mp4',
                    out_path_result + '_side2.mp4',
                    out_path_result + '_top.mp4',
                    out_path_result + '_merge.mp4'
                ))

        print(out_path_result + '_merge.mp4')

        # remove the frames.
        print('rm -rf %s' % (out_path_result ))
        print('rm -rf %s' % (out_path_result + '_side'))
        os.system('rm -rf %s' % (out_path_result ))
        os.system('rm %s' % (out_path_result + '.mp4'))
        os.system('rm -rf %s' % (out_path_result + '_side'))
        os.system('rm %s' % (out_path_result + '_side.mp4'))
        os.system('rm -rf %s' % (out_path_result + '_side2'))
        os.system('rm %s' % (out_path_result + '_side2.mp4'))

        if not only_top:
            os.system('rm -rf %s' % (out_path_result + '_top'))
            os.system('rm %s' % (out_path_result + '_top.mp4'))
            

