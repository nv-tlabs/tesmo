import os
import json
import pickle

import smplx

import torch

import numpy as np

from tqdm import tqdm

from utils import get_graph_params, ds_us


def pickle_amass_vertices(input_path, output_path, output_name, body_models, device, num_pca_comps=6):
    # import pdb; pdb.set_trace()
    parameters = np.load(input_path)

    gender = str(parameters["gender"])
    # if gender is byte string, convert to string
    if gender not in body_models:
        gender = str(gender, encoding='utf-8')
    body_model = body_models[gender]
    betas = parameters["betas"][:10]
    target_fps = 20
    fps = parameters['mocap_framerate']
    down_sample = int(fps / target_fps)
    # print(list(parameters.keys()))
    # import pdb; pdb.set_trace()
    
    frames = []
    for fId in range(0, len(parameters["poses"]), down_sample):
        frame = {}
        frame["root_orient"] = parameters['poses'][fId:fId+1, :3]
        frame["trans"] = parameters['trans'][fId:fId+1]
        frame["pose_body"] = parameters['poses'][fId:fId+1, 3:66]
        frame["pose_hand"] = parameters['poses'][fId:fId+1, 66:]
        frames.append(frame)
    
    
    _, _, D_1 = get_graph_params("mesh_ds", 1, device)
    ds1 = ds_us(D_1).to(device)
    _, _, D_2 = get_graph_params("mesh_ds", 2, device)
    ds2 = ds_us(D_2).to(device)
    
    all_vertices = []
    all_vertices_can = []
    all_vertices_ds2 = []
    all_vertices_can_ds2 = []
    torch_params = {}
    # import pdb; pdb.set_trace()
    torch_params['betas'] = torch.tensor(betas, dtype=torch.float32).to(device).unsqueeze(0)
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        torch_params['global_orient'] = torch.tensor(frame['root_orient'], dtype=torch.float32).to(device)
        torch_params['transl'] = torch.tensor(frame['trans'], dtype=torch.float32).to(device)
        torch_params['body_pose'] = torch.tensor(frame['pose_body'], dtype=torch.float32).to(device).flatten()
        torch_params['left_hand_pose'] = torch.tensor(frame['pose_hand'][:, :45], dtype=torch.float32).to(device).flatten()
        torch_params['right_hand_pose'] = torch.tensor(frame['pose_hand'][:, 45:], dtype=torch.float32).to(device).flatten()
        body_model.reset_params(**torch_params)
        body_model_output = body_model(return_verts=True)
        vertices = body_model_output.vertices.squeeze()
        pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
        vertices_can = vertices - pelvis
        all_vertices.append(vertices.detach().cpu().numpy())
        all_vertices_can.append(vertices_can.detach().cpu().numpy())
        vertices_ds2 = ds2(ds1(vertices.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
        vertices_can_ds2 = ds2(ds1(vertices_can.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
        all_vertices_ds2.append(vertices_ds2.detach().cpu().numpy())
        all_vertices_can_ds2.append(vertices_can_ds2.detach().cpu().numpy())
    
    all_vertices = np.array(all_vertices)
    all_vertices_can = np.array(all_vertices_can)
    all_vertices_ds2 = np.array(all_vertices_ds2)
    all_vertices_can_ds2 = np.array(all_vertices_can_ds2)
    
    np.save(os.path.join(output_path, output_name + "_verts.npy"), all_vertices)
    np.save(os.path.join(output_path, output_name + "_verts_can.npy"), all_vertices_can)
    np.save(os.path.join(output_path, output_name + "_verts_ds2.npy"), all_vertices_ds2)
    np.save(os.path.join(output_path, output_name + "_verts_can_ds2.npy"), all_vertices_can_ds2)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seq_id", type=int, default=0)
    # parser.add_argument("input_path",
    #                     type=str)
    # parser.add_argument("output_path",
    #                     type=str)
    # parser.add_argument("output_name",
    #                     type=str)
    args = parser.parse_args()
    amass_root = '/ps/project/datasets/AMASS/amass_april_2019/'
    file_lists = ['../priorMDM/dataset/HumanML3D/summon/seq_list_0128.txt']
    annotation_file = '../priorMDM/dataset/HumanML3D/annotations.json'
    output_dir = '../priorMDM/dataset/HumanML3D/summon/ds_verts'    
    
    os.makedirs(output_dir, exist_ok=True)
                 
    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")
    print()    
    
    smplx_models = "../priorMDM/body_models/smplx_models/"
    
    seq_list = []
    for list_f in file_lists:
        with open(list_f, 'r') as f:
            file_names = [s.strip() for s in f.readlines()]
            seq_list.extend(file_names)
    
    annotation = json.load(open(annotation_file, 'r'))
    
    
    model_params = dict(model_path=smplx_models,
                        model_type='smplx',
                        ext='npz',
                        # num_pca_comps=6,
                        use_pca=False,
                        batch_size=1,
                        num_betas=10,
                        flat_hand_mean=True)
    print(model_params)
    body_models = {}
    for gender in ['male', 'female', 'neutral']:
        body_models[gender] = smplx.create(gender=gender, **model_params).to(device)
    
    failed_cnt = 0
    output_seq_list = []
    seq = seq_list[args.seq_id]
    input_path = os.path.join(amass_root, annotation[seq]['path'] + '.npz')
    if os.path.exists(input_path):
        pickle_amass_vertices(input_path, output_dir, seq, body_models, device)
