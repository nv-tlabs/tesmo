import os
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

def load_smap_motion(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        full_poses = torch.tensor(data['pose_est_fullposes'], dtype=torch.float32)
        betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
        full_trans = torch.tensor( data['pose_est_trans'], dtype=torch.float32)
        fps = data['mocap_framerate']
        print("Number of frames is {}".format(full_poses.shape[0]))
    
    return betas, full_poses, full_trans, fps

def get_semantic_poses(betas, full_poses, full_transl, i):
    # extract frame_i body pose from a sequence of poses.
    tmp_poses = full_poses[i:i+1, :]
    result = {
        'betas': betas,
        'global_orient': tmp_poses[:, :3],
        'transl': full_transl[i:i+1, :],
        'body_pose': tmp_poses[:, 3:66],
        'jaw_pose': tmp_poses[:, 66:69],
        'leye_pose': tmp_poses[:, 69:72],
        'reye_pose': tmp_poses[:, 72:75],
        'left_hand_pose': tmp_poses[:, 75:120],
        'right_hand_pose': tmp_poses[:, 120:],
        'expression': torch.zeros((1, 10)),
        'keypoints_3d': torch.zeros((1, 25, 3)),
        'pose_embedding': torch.zeros((1, 32)),
        'gender': 'male'
    }

    return result

# Example usage
# python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir ../results/amass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--dataset_type", type=str, default="samp",
                        help="samp or amass")
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
    parser.add_argument("--seq_id", type=int, default=None)
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

    if torch.cuda.is_available():
        # print("Using cuda")
        device = torch.device("cuda")
    else:
        # print("Using cpu")
        device = torch.device("cpu")
    num_obj_classes = 8
    # For fix_ori
    fix_ori = True
    ds_weights = torch.tensor(np.load("./support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    os.makedirs(output_dir, exist_ok=True)
    if args.dataset_type == "samp":
        seq_file_list = os.listdir(data_dir)
        seq_name_list = [f.split('.')[0] for f in seq_file_list]
        smplx_models_path = './smpl-x_model/models'
        model_params = dict(model_path=smplx_models_path,
                            model_type='smplx',
                            ext='npz',
                            use_pca=False,
                            batch_size=1)
        body_model = smplx.create(gender="male", **model_params).to(device).eval()
    elif args.dataset_type == "amass":        
        seq_name_list = []
        vertices_file_list = [f for f in os.listdir(data_dir) if '_verts_can_ds2' in f]
        seq_name_list = [file_name.split('_verts_can_ds2')[0] for file_name in vertices_file_list]
        list_set = set(seq_name_list)
        seq_name_list = list(list_set)

    # Load in model checkpoints and set up data stream
    model = ContactFormer(seg_len=max_frame, encoder_mode=encoder_mode, decoder_mode=decoder_mode,
                          n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff, d_hid=512,
                          posa_path=posa_path).to(device)
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, D_1 = get_graph_params("../mesh_ds", 1, device)
    ds1 = ds_us(D_1).to(device)
    _, _, D_2 = get_graph_params("../mesh_ds", 2, device)
    ds2 = ds_us(D_2).to(device)
    can_ds_vert_dir = data_dir.replace('pkl', 'can_ds_vert')
    ds_vert_dir = data_dir.replace('pkl', 'ds_vert')
    os.makedirs(can_ds_vert_dir, exist_ok=True)
    os.makedirs(ds_vert_dir, exist_ok=True)
    for seq_name in seq_name_list:
        if args.dataset_type == "samp" and not seq_name + '.pkl' in seq_file_list:
            continue
        if args.seq_name is not None and seq_name != args.seq_name:
            continue
        if args.save_video or args.visualize:
            save_seq_dir = os.path.join(output_dir, seq_name)
            os.makedirs(save_seq_dir, exist_ok=True)
            output_image_dir = os.path.join(save_seq_dir, 'vis')
            os.makedirs(output_image_dir, exist_ok=True)
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window(visible=False)
            down_sample_level = 2
            # tpose for getting face_arr
            tpose_mesh_path = os.path.join(args.tpose_mesh_dir, "mesh_{}.obj".format(down_sample_level))
            faces_arr = trimesh.load(tpose_mesh_path, process=False).faces
        print("Test scene: {}".format(seq_name))
        if args.dataset_type == "samp":
            seq_file = seq_name + '.pkl'
            betas, full_poses, full_trans, fps = load_smap_motion(os.path.join(data_dir, seq_file))
            all_vertices_can = []
            all_vertices = []
            # import pdb;pdb.set_trace()
            for i in tqdm(range(len(full_poses))):
                semantic_poses = get_semantic_poses(betas, full_poses, full_trans, i)
                torch_params = {}
                torch_params['betas'] = torch.tensor(semantic_poses['betas'], dtype=torch.float32).to(device)
                torch_params['global_orient'] = torch.tensor(semantic_poses['global_orient'], dtype=torch.float32).to(device)
                torch_params['transl'] = torch.tensor(semantic_poses['transl'], dtype=torch.float32).to(device)
                torch_params['body_pose'] = torch.tensor(semantic_poses['body_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['jaw_pose'] = torch.tensor(semantic_poses['jaw_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['leye_pose'] = torch.tensor(semantic_poses['leye_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['reye_pose'] = torch.tensor(semantic_poses['reye_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['left_hand_pose'] = torch.tensor(semantic_poses['left_hand_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['right_hand_pose'] = torch.tensor(semantic_poses['right_hand_pose'], dtype=torch.float32).to(device).flatten()
                torch_params['expression'] = torch.tensor(semantic_poses['expression'], dtype=torch.float32).to(device).flatten()
                body_model.reset_params(**torch_params)
                with torch.no_grad():
                    body_model_output = body_model(return_verts=True)
                vertices = body_model_output.vertices.squeeze()
                pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
                vertices_can = vertices - pelvis
                vertices_can = ds2(ds1(vertices_can.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
                vertices_ds =  ds2(ds1(vertices.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
                all_vertices_can.append(vertices_can.cpu().numpy())
                all_vertices.append(vertices_ds.cpu().numpy())
            np.save(str(os.path.join(can_ds_vert_dir, seq_file.replace('pkl', 'npy'))), np.array(all_vertices_can))
            np.save(str(os.path.join(ds_vert_dir, seq_file.replace('pkl', 'npy'))), np.array(all_vertices))
            verts_can = torch.tensor(np.array(all_vertices_can)).to(device).to(torch.float32)
        elif args.dataset_type == "amass":
            verts_can = torch.tensor(np.load(os.path.join(data_dir, seq_name + "_verts_can_ds2.npy"))).to(device).to(torch.float32)
            all_vertices_can = verts_can.cpu().numpy()
        # Loop over video frames to get predictions
        verts_can_batch = verts_can[::jump_step]
        if fix_ori:
            verts_can_batch = du.normalize_orientation(verts_can_batch, associated_joints, device)
        if verts_can_batch.shape[0] > max_frame:
            verts_can_batch = verts_can_batch[:max_frame]

        mask = torch.zeros(1, max_frame, device=device)
        mask[0, :verts_can_batch.shape[0]] = 1
        verts_can_padding = torch.zeros(max_frame - verts_can_batch.shape[0], *verts_can_batch.shape[1:], device=device)
        verts_can_batch = torch.cat((verts_can_batch, verts_can_padding), dim=0)

        z = torch.tensor(np.random.normal(0, 1, (max_frame, 256)).astype(np.float32)).to(device)

        with torch.no_grad():
            posa_out = model.posa.decoder(z, verts_can_batch)
            if decoder_mode == 0:
                pr_cf = posa_out.unsqueeze(0)
            else:
                pr_cf = model.decoder(posa_out, mask)

        pred = pr_cf.squeeze()
        pred = pred[:(int)(mask.sum())]
        cur_output_path = os.path.join(output_dir, seq_name + ".npy")
        if save_probability:
            softmax = torch.nn.Softmax(dim=2)
            pred_npy = softmax(pred).detach().cpu().numpy()
        else:
            pred_npy = torch.argmax(pred, dim=-1).unsqueeze(-1).detach().cpu().numpy()
        np.save(cur_output_path, pred_npy)
        if args.save_video or args.visualize:
            for i in range(len(pred)):
                mesh = vis_utils.show_sample(all_vertices_can[i*jump_step], pred_npy[i,:,0], faces_arr, True)[0]
                R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0.))
                mesh.rotate(R, center=(0, 0, 0))
                visualizer.add_geometry(mesh)
                visualizer.poll_events()
                visualizer.update_renderer()
                visualizer.capture_screen_image(
                    os.path.join(output_image_dir, "frame_{:04d}.png".format(i*jump_step)))
                visualizer.remove_geometry(mesh)
                o3d.io.write_triangle_mesh("frame_{:04d}.ply".format(i*jump_step), mesh, write_vertex_colors=True)