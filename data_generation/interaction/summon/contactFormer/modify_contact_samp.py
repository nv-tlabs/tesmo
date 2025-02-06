import os
import json
import numpy as np
import argparse
import torch
from contactFormer import ContactFormer
from tqdm import tqdm
import data_utils as du
import pickle
import smplx
import open3d as o3d
import trimesh
import vis_utils
from utils import get_graph_params, ds_us

def get_prox_contact_labels(contact_parts='body', body_model='smplx'):
    # data_path = f'{mime_code_dir}/data'
    data_path = '/home/hyi/data/workspace/motion_generation/priorMDM/thirdparty/MIME/data'
    # contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
    if contact_parts == 'body':
        contact_body_parts = ['gluteus', 'back', 'thighs']
        if ADD_HAND_CONTACT:
            contact_body_parts.append('L_Hand')
            contact_body_parts.append('R_Hand')
    elif contact_parts == 'gluteus':
        contact_body_parts = ['gluteus']
    elif contact_parts == 'gluteus+hand':
        contact_body_parts = ['gluteus', 'L_Hand', 'R_Hand']
    elif contact_parts == 'gluteus+lhand':
        contact_body_parts = ['gluteus', 'L_Hand']
    elif contact_parts == 'gluteus+rhand':
        contact_body_parts = ['gluteus', 'R_Hand']
    elif contact_parts == 'thighs':
        contact_body_parts = ['thighs']
    elif contact_parts == 'feet':
        contact_body_parts = ['L_Leg', 'R_Leg'] 
    elif contact_parts == 'Lfeet':
        contact_body_parts = ['L_Leg']
    elif contact_parts == 'Rfeet':
        contact_body_parts = ['R_Leg']
    elif contact_parts == 'handArm':
        contact_body_parts = ['R_Hand', 'L_Hand', 'rightForeArm', 'leftForeArm']
    elif contact_parts == 'hand':
        contact_body_parts = ['R_Hand', 'L_Hand']
    elif contact_parts == 'Lhand':
        contact_body_parts = ['L_Hand']
    elif contact_parts == 'Rhand':
        contact_body_parts = ['R_Hand']
    elif contact_parts == 'Arm':
        contact_body_parts = ['rightForeArm', 'leftForeArm']
    elif contact_parts == 'LArm':
        contact_body_parts = ['leftForeArm']
    elif contact_parts == 'RArm':
        contact_body_parts = ['rightForeArm']
    elif contact_parts == 'whole_body_no_arm':
        contact_body_parts = ['whole_body_no_arm']

    if body_model == 'smplx':
        body_segments_dir = f'{data_path}/body_segments/smplx/body_segments'
    elif body_model == 'smpl':
        body_segments_dir = f'{data_path}/body_segments/smpl'

    contact_verts_ids = []
    # load prox contact label information.
    for part in contact_body_parts:
        with open(os.path.join(body_segments_dir, part + '.json'), 'r') as f:
            data = json.load(f)
            contact_verts_ids.append(list(set(data["verts_ind"])))
    contact_verts_ids = np.concatenate(contact_verts_ids)
    return contact_verts_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--seq_name", type=str, default=None,
                        help="sequence name")
    parser.add_argument("--load_model", type=str, default="../training/model_ckpt/epoch_0045.pt",
                        help="checkpoint path to load")
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="different number represents different variants of encoder")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="different number represents different variants of decoder")
    parser.add_argument("--n_layer", type=int, default=3, help="Number of layers in transformer")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads in transformer")
    parser.add_argument("--jump_step", type=int, default=16, help="Frame skip size for each input motion sequence")
    parser.add_argument("--dim_ff", type=int, default=512,
                        help="Dimension of hidden layers in positionwise MLP in the transformer")
    parser.add_argument("--f_vert", type=int, default=64, help="Dimension of the embeddings for body vertices")
    parser.add_argument("--max_frame", type=int, default=256,
                        help="The maximum length of motion sequence (after frame skipping) which model accepts.")
    parser.add_argument("--posa_path", type=str, default="../training/posa/model_ckpt/epoch_0349.pt",
                        help="The POSA model checkpoint that ContactFormer can pre-load")
    parser.add_argument("--output_dir", type=str, default="../results/output")
    parser.add_argument("--save_probability", dest='save_probability', action='store_const', const=True, default=False,
                        help="Save the probability of each contact labels, instead of the most possible contact label")
    parser.add_argument("--save_video", dest='save_video', action='store_const', const=True, default=False)
    parser.add_argument("--visualize", dest="visualize", action='store_const', const=True, default=False)
    parser.add_argument("--tpose_mesh_dir", type=str, default="../mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    ckpt_path = args_dict['load_model']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']
    output_dir = args_dict['output_dir']
    save_probability = args_dict['save_probability']

    device = torch.device("cuda")
    num_obj_classes = 8
    # For fix_ori
    fix_ori = True
    ds_weights = torch.tensor(np.load("./support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    _, _, D_1 = get_graph_params("../mesh_ds", 1, device)
    ds1 = ds_us(D_1).to(device)
    _, _, D_2 = get_graph_params("../mesh_ds", 2, device)
    ds2 = ds_us(D_2).to(device)

    manual_annos = []
    with open(f'{output_dir}/hand_annos.txt') as f:
        hand_annos = f.readlines()
        for l in hand_annos:
            if l[0] == '#':
                continue
            l = l.split(' ')
            seq_name = l[0]
            start_frame = int(l[1])
            end_frame = int(l[2])
            cls = int(l[3])
            bodypart = l[4].strip()
            manual_annos.append((seq_name, start_frame, end_frame, cls, bodypart))
    for seq_name, start_frame, end_frame, cls, bodypart in manual_annos:
        num_points = D_1.shape[1]
        bodypart_idx = get_prox_contact_labels(contact_parts=bodypart, body_model='smplx')
        bodypart_labels = torch.zeros(num_points).to(device)
        print("bodypart_idx", bodypart_idx)
        bodypart_labels[bodypart_idx] = 1
        bodypart_ds = ds2(ds1(torch.tensor(bodypart_labels).unsqueeze(0).unsqueeze(0).to(device))).squeeze()
        bodypart_ds = bodypart_ds > 0.
        print("bodypart_ds", bodypart_ds.sum())
        print(seq_name, start_frame, end_frame, cls)
        pred_file_old = pred_file = os.path.join(output_dir, seq_name + '.npy')
        if os.path.exists(os.path.join(output_dir, seq_name + '.npy.bak')):
            pred_file_old = os.path.join(output_dir, seq_name + '.npy.bak')
        pred_npy = np.load(pred_file_old)
        os.rename(pred_file_old, os.path.join(output_dir,seq_name + '.npy.bak'))
        floor_mask = pred_npy == 2
        pred_npy[~floor_mask] = 0
        pred_npy[start_frame//jump_step:(end_frame//jump_step)+1, bodypart_ds.cpu().numpy()] = cls
        np.save(pred_file, pred_npy)
        faces_arr = trimesh.load(os.path.join(args.tpose_mesh_dir, "mesh_2.obj")).faces
        all_vertices_can = np.load(os.path.join(data_dir.replace('pkl', 'can_ds_vert'), seq_name + '.npy'))
        # print(all_vertices_can.shape, pred_npy.shape)
        output_image_dir = os.path.join(output_dir, seq_name)
        if not os.path.exists(output_image_dir+'_bak'):
            os.rename(output_image_dir, output_image_dir+'_bak')
        if args.save_video or args.visualize:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window(visible=False)
            os.makedirs(output_image_dir, exist_ok=True)
            for i in range(len(pred_npy)):
                mesh = vis_utils.show_sample(all_vertices_can[i*jump_step], pred_npy[i,:,0], faces_arr, True)[0]
                R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0.))
                mesh.rotate(R, center=(0, 0, 0))
                visualizer.add_geometry(mesh)
                visualizer.poll_events()
                visualizer.update_renderer()
                visualizer.capture_screen_image(
                    os.path.join(output_image_dir, "frame_{:04d}.png".format(i*jump_step)))
                visualizer.remove_geometry(mesh)
                