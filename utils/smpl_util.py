import os
from human_body_prior.body_model.body_model import BodyModel
import torch
import numpy as np

def load_body_models(support_base_dir='./thirdparty/HumanML3D/body_models', surface_model_type = "smplh", device='cuda:0'):
    surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, 'male', 'model.npz')
    surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "female", 'model.npz')
    dmpl_fname = None
    num_dmpls = None
    num_expressions = None
    num_betas = 16

    male_bm = BodyModel(surface_model_male_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname).to(device)
    female_bm = BodyModel(surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)
    bm_dict = {'male': male_bm, 'female': female_bm}
    # bm_dict = {'male': male_bm}

    return bm_dict

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3
    # betas: BS X 16
    # gender: BS

    # print('aa_rot_rep.shape, root_trans.shape: ', aa_rot_rep.shape, root_trans.shape)
    # aa_rot_rep.shape, root_trans.shape:  torch.Size([4, 181, 22, 3]) torch.Size([4, 196, 3])
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device)  # BS X T X 30 X 3
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2)  # BS X T X 52 X 3

    aa_rot_rep = aa_rot_rep.reshape(bs * num_steps, -1, 3)  # (BS*T) X n_joints X 3
    if betas is None:
        betas_batch = torch.zeros(bs, 16).to(aa_rot_rep.device)
        betas = betas_batch[:, None, :].repeat(1, num_steps, 1).reshape(bs * num_steps, -1)  # (BS*T) X 16
    
    if gender is None: # defaulted to male
        gender = ['male' for i in range(bs)]

    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist()  # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3)  # (BS*T) X 3
    smpl_betas = betas  # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :]  # (BS*T) X 3
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63)  # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90)  # (BS*T) X 90

    B = smpl_trans.shape[0]  # (BS*T)

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=int) * -1
    
        
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=int)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue

        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                       betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)

        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map]  # (BS*T) X 22 X 3

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map]  # (BS*T) X 6890 X 3

    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3)  # BS X T X 22 X 3
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3)  # BS X T X 6890 X 3

    mesh_faces = pred_body.f
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces