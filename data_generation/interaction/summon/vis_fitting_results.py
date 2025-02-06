import os
from pathlib import Path
import numpy as np
import argparse
import open3d as o3d
from gen_human_meshes import gen_human_meshes
import json
from contactFormer import vis_utils
from tqdm import tqdm
import imageio.v3 as iio
import trimesh
import torch
import imageio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fitting_results_path", type=str, help="Path to the fitting results of some motion sequence")
    parser.add_argument("--vertices_path", type=str, help="Path to human vertices of some motion sequence")
    parser.add_argument("--contact_predicts_path", type=str, help="Path to human vertices of some motion sequence")

    args = parser.parse_args()
    input_dir = Path(args.fitting_results_path)
    vertices_path = Path(args.vertices_path)
    seq_name = input_dir.stem
    contact_preds = np.load(args.contact_predicts_path)
    print(contact_preds.max())

    # Check if human meshes are there
    human_mesh_dir = input_dir / 'human' / 'mesh'
    if not human_mesh_dir.exists():
        human_mesh_dir.mkdir()
        gen_human_meshes(vertices_path=vertices_path, output_path=human_mesh_dir)

    # Rendering results
    output_dir = input_dir / 'rendering'
    output_dir.mkdir(exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    #opt = vis.get_render_option()
    #opt.background_color = np.asarray([0, 0, 0])

    res_dir = input_dir / 'fit_best_obj'
    obj_mesh_list = []
    for obj_class_dir in res_dir.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            best_obj_id = best_obj_json['best_obj_id'].split('/')[-1]
            best_obj_path = obj_dir / best_obj_id / 'opt_best.obj'
            obj_mesh = o3d.io.read_triangle_mesh(str(best_obj_path))
            obj_mesh.compute_vertex_normals()
            R = obj_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0.))
            obj_mesh.rotate(R, center=(0, 0, 0))
            obj_mesh_list.append(obj_mesh)

    frame = 0
    jump_step = 16
    print(len(list(human_mesh_dir.iterdir())))
    for _ in tqdm(list(human_mesh_dir.iterdir())[::jump_step][:256]):
        mesh_list = [] + obj_mesh_list
        human_mesh_path = human_mesh_dir / f"human_{frame*jump_step}.ply"
        human_mesh = o3d.io.read_triangle_mesh(str(human_mesh_path))
        human_mesh.compute_vertex_normals()
        # mesh_list.append(human_mesh)
        cur_pred_semantic = contact_preds[frame, :, 0]
        semantic_vis = vis_utils.show_sample(np.asarray(human_mesh.vertices), cur_pred_semantic, np.asarray(human_mesh.triangles), True)[0]
        o3d.io.write_triangle_mesh(f"{frame}.ply", semantic_vis, write_vertex_colors=True)
        R = semantic_vis.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0.))
        semantic_vis.rotate(R, center=(0, 0, 0))
        mesh_list.append(semantic_vis)
        for i, geometry in enumerate(mesh_list):
            vis.add_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_dir / f"frame_{frame:04d}.png"))

        for geometry in mesh_list:
            vis.remove_geometry(geometry)
        frame += 1
    video_frames = []
    print(frame)
    with imageio.get_writer(filepath, fps=30) as video:
        for i in range(frame):
            img = iio.imread(str(output_dir / f"frame_{i:04d}.png"))
            video.append(img)
    vis.destroy_window()