import random
import torch
import numpy as np
from data_loaders.amass.tools import collate_tensor_with_padding

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    # print('notnone_batches: ', notnone_batches[0])

    databatch = [b['inp'] for b in notnone_batches]
        
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    # print('databatch device: ', databatchTensor.device)

    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'input_motion': databatchTensor}}

    # TODO: the below is useless.
    if 'is_transition' in notnone_batches[0]:
        is_transition_batch = torch.stack([b['is_transition']for b in notnone_batches])
        cond['y'].update({'is_transition': is_transition_batch})

    if 'length_transition' in notnone_batches[0]:
        length_transition = [b['length_transition'] for b in notnone_batches]
        cond['y'].update({'length_transition': length_transition})

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})


    if 'other_motion' in notnone_batches[0]:
        other_motion = [b['other_motion'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'other_motion': other_motion})

    if 'person_id' in notnone_batches[0]:
        textbatch = [b['person_id'] for b in notnone_batches]
        cond['y'].update({'person_id': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'action_cat' in notnone_batches[0]:
        action_cat = torch.stack([b['action_cat']for b in notnone_batches])
        action_cat_mask = torch.stack([b['action_cat_mask']for b in notnone_batches])
        act_cat_list = [b['act_cat_list']for b in notnone_batches]
        cond['y'].update({'action_cat': action_cat})
        cond['y'].update({'action_cat_mask': action_cat_mask})
        cond['y'].update({'act_cat_list': act_cat_list})

    if 'bps_data' in notnone_batches[0]:
        other_motion = [b['bps_data'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'bps_data': other_motion})
    if 'betas' in notnone_batches[0]:
        other_motion = [b['betas'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'betas': other_motion})
    if 'move_to_zero_trans' in notnone_batches[0]:
        other_motion = [b['move_to_zero_trans'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'move_to_zero_trans': other_motion})
    if 'recover_rot_quat' in notnone_batches[0]:
        other_motion = [b['recover_rot_quat'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'recover_rot_quat': other_motion})
    if 'local_rot_6d' in notnone_batches[0]:
        textbatch = [b['local_rot_6d'] for b in notnone_batches]
        textbatch = collate_tensors(textbatch)
        cond['y'].update({'local_rot_6d': textbatch})
    if 'root_trans' in notnone_batches[0]:
        textbatch = [b['root_trans'] for b in notnone_batches]
        textbatch = collate_tensors(textbatch)
        cond['y'].update({'root_trans': textbatch})

    if 'start_goal' in notnone_batches[0]:
        other_motion = [b['start_goal'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'start_goal': other_motion})
    if 'seq_name' in notnone_batches[0]:
        textbatch = [b['seq_name'] for b in notnone_batches]
        cond['y'].update({'seq_name': textbatch})
    if 'gender' in notnone_batches[0]:
        textbatch = [b['gender'] for b in notnone_batches]
        cond['y'].update({'gender': textbatch})
    if 'seq_len' in notnone_batches[0]:
        textbatch = [b['seq_len'] for b in notnone_batches]
        cond['y'].update({'seq_len': textbatch})
    if 'inpainting_mask' in notnone_batches[0] and notnone_batches[0]['inpainting_mask'] != []:
        inpainting_mask = [b['inpainting_mask'] for b in notnone_batches]
        inpainting_mask = collate_tensors(inpainting_mask)
        cond['y'].update({'inpainting_mask': inpainting_mask})
        # cond['y'].update({'inpainted_motion': motion})
    if 'lhand_rhand_mask' in notnone_batches[0] and notnone_batches[0]['lhand_rhand_mask'] != []:
        lhand_rhand_mask = [b['lhand_rhand_mask'] for b in notnone_batches]
        lhand_rhand_mask = collate_tensors(lhand_rhand_mask)
        cond['y'].update({'lhand_rhand_mask': lhand_rhand_mask})
        # cond['y'].update({'inpainted_motion': motion})
    if 'inpainted_motion' in notnone_batches[0] and notnone_batches[0]['inpainted_motion'] != []:
        databatch = [b['inpainted_motion'] for b in notnone_batches]
        inpainted_motion = collate_tensors(databatch)
        cond['y'].update({'inpainted_motion': inpainted_motion})
    # import pdb;pdb.set_trace()
    if 'scene' in notnone_batches[0] and notnone_batches[0]['scene'] != []:
        databatch = [b['scene'] for b in notnone_batches]
        scene = collate_tensors(databatch)
        cond['y'].update({'scene': scene})
    
    if 'obj_point_data' in notnone_batches[0] and notnone_batches[0]['obj_point_data'] != []:
        databatch = [b['obj_point_data'] for b in notnone_batches]
        scene = collate_tensors(databatch)
        cond['y'].update({'obj_point_data': scene})
    
    if 'object' in notnone_batches[0]:
        # TODO: move this into getitem
        # hasattr()
        for keys in notnone_batches[0]['object'].sdf_dict.keys():
            if isinstance(notnone_batches[0]['object'].sdf_dict[keys], np.ndarray) :
                databatch = [torch.from_numpy(b['object'].sdf_dict[keys].copy()).float() for b in notnone_batches]
            elif isinstance(notnone_batches[0]['object'].sdf_dict[keys], np.float64):
                databatch = [torch.from_numpy(b['object'].sdf_dict[keys].copy()[None]).float() for b in notnone_batches]
            else:
                databatch = [torch.Tensor(b['object'].sdf_dict[keys]).float() for b in notnone_batches] # int, float;
            scene = collate_tensors(databatch)
            cond['y'].update({'sdf_'+keys: scene})
    
    if 'transform_mat' in notnone_batches[0]:
        databatch = [b['transform_mat'] for b in notnone_batches]
        trans_mat = collate_tensors(databatch)
        cond['y'].update({'transform_mat': trans_mat})

    if 'obj_transform_trans' in notnone_batches[0]:
        databatch = [b['obj_transform_trans'] for b in notnone_batches]
        trans_mat = collate_tensors(databatch)
        cond['y'].update({'obj_transform_trans': trans_mat})

    if 'obj_transform_rotation' in notnone_batches[0]:
        databatch = [b['obj_transform_rotation'] for b in notnone_batches]
        trans_mat = collate_tensors(databatch)
        cond['y'].update({'obj_transform_rotation': trans_mat})

    if 'obj_transform_scale' in notnone_batches[0]:
        databatch = [b['obj_transform_scale'] for b in notnone_batches]
        trans_mat = collate_tensors(databatch)
        cond['y'].update({'obj_transform_scale': trans_mat})

    if 'floor_map_easy_hard_kind' in notnone_batches[0]:
        databatch = [b['floor_map_easy_hard_kind'] for b in notnone_batches]
        easy_hard_kind = collate_tensors(databatch)
        cond['y'].update({'floor_map_easy_hard_kind': easy_hard_kind})
    
    if 'ori_humanml_motion' in notnone_batches[0]:
        other_motion = [b['ori_humanml_motion'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'ori_humanml_motion': other_motion})

    if 'obj_bps' in notnone_batches[0]:
        other_motion = [b['obj_bps'] for b in notnone_batches]
        other_motion = collate_tensors(other_motion)
        cond['y'].update({'obj_bps': other_motion})

    if 'obj_id' in notnone_batches[0]: # store the load object name
        textbatch = [b['obj_id'] for b in notnone_batches]
        cond['y'].update({'obj_id': textbatch})

    if 'scene_mesh' in notnone_batches[0]: # store the load object name
        scene_mesh_batch = [b['scene_mesh'] for b in notnone_batches]
        cond['y'].update({'scene_mesh': scene_mesh_batch})
    
    for tmp_ke in ['word_embeddings', 'pos_one_hots', 'sent_lens']:
        if tmp_ke in notnone_batches[0]:
            textbatch = [b[tmp_ke] for b in notnone_batches]
            trans_mat = collate_tensors(textbatch)
            cond['y'].update({tmp_ke: trans_mat})
    
    # import pdb;pdb.set_trace()

    return motion, cond

def t2m_collate_eval(batch):
    
    if len(batch[0]) >= 9:
        # print(type(batch[0][8]))
        if isinstance(batch[0][8], np.ndarray):
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'scene': torch.from_numpy(b[8]),
            } for b in batch]
            
        elif isinstance(batch[0][8], list):
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'scene': torch.from_numpy(b[8][0]),
                'object': b[8][1]
            } for b in batch]
        else:
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'object': b[8], # do nothing.
            } for b in batch]
    else: # on sdf or scene map in the batch.
        adapted_batch = [{
            'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'is_transition': torch.zeros(1), # just for eval not really needed
        } for b in batch]


    # TODO: add different condition or other needed data.
    for i, b in enumerate(batch):
        if len(b[7]) > 1:
            adapted_batch[i]['transform_mat'] = torch.from_numpy(b[7][1]) # canonicalization transform to world CSmatrix 

        if len(b[7])>0 and len(b[7]) > 3:
            # adapted_batch[i]['start_goal'] = torch.from_numpy(b[7][0])
            adapted_batch[i]['obj_transform_trans'] = torch.from_numpy(b[7][2])
            adapted_batch[i]['obj_transform_rotation'] = torch.from_numpy(b[7][3])

            if len(b[7]) >= 5:
                adapted_batch[i]['obj_id'] = b[7][4]
            if len(b[7]) >= 6:
                if isinstance(b[7][5], float):
                    adapted_batch[i]['obj_transform_scale'] = torch.from_numpy(np.array([b[7][5]])[None])
                else:
                    # print(b[7][5], 'is numpy')
                    adapted_batch[i]['obj_transform_scale'] = torch.from_numpy(b[7][5][None])

        # add word_embeddings, pos_one_hots, _, sent_lens
        if len(b) >= 10 and isinstance(b[9], list): # human-object interaction
            adapted_batch[i]['obj_point_data'] = torch.from_numpy(b[9][0])
        
        if len(b) >= 10 and isinstance(b[7], list): # human-object interaction
            adapted_batch[i]['scene_mesh'] = b[7][0]

        if len(b) >= 10:
            adapted_batch[i]['floor_map_easy_hard_kind'] = torch.from_numpy(np.array([b[9]]))
        
        if len(b) >= 11:
            adapted_batch[i]['ori_humanml_motion'] = torch.from_numpy(b[10])

        adapted_batch[i]['word_embeddings'] = torch.from_numpy(b[0])
        adapted_batch[i]['pos_one_hots'] = torch.from_numpy(b[1])
        adapted_batch[i]['sent_lens'] = torch.from_numpy(np.array([b[3]])) # token length
        
        # import pdb;pdb.set_trace()

    return collate(adapted_batch)

def t2m_collate(batch): # TODO: This is what we need to modify.
    # batch.sort(key=lambda x: x[3], reverse=True)
    # import pdb;pdb.set_trace()
    if len(batch[0]) >= 9:
        # print(type(batch[0][8]))
        if isinstance(batch[0][8], np.ndarray):
            # for b in batch:
            #     print(b[2], b[8].shape, type(b[8]))
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'scene': torch.from_numpy(b[8]),
                # 'scene': torch.from_numpy(b[8]) if isinstance(b[8], np.ndarray) else torch.from_numpy(b[8].cpu().numpy()), # FIXME: why bps feature contains tensor?
                # 'scene': torch.from_numpy(b[8]) if isinstance(b[8], np.ndarray) else torch.from_numpy(b[8].cpu().numpy()), # FIXME: why bps feature contains tensor?
            } for b in batch]
            
        elif isinstance(batch[0][8], list):
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'scene': torch.from_numpy(b[8][0]),
                'object': b[8][1]
            } for b in batch]
        else:
            adapted_batch = [{
                'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
                'text': b[2], #b[0]['caption']
                'tokens': b[6],
                'lengths': b[5],
                'is_transition': torch.zeros(1), # just for eval not really needed
                'object': b[8], # do nothing.
            } for b in batch]
    else: # on sdf or scene map in the batch.
        adapted_batch = [{
            'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'is_transition': torch.zeros(1), # just for eval not really needed
        } for b in batch]


    # TODO: add different condition or other needed data.
    # import pdb;pdb.set_trace()
    for i, b in enumerate(batch):
        if len(b[7]) > 1:
            adapted_batch[i]['transform_mat'] = torch.from_numpy(b[7][1]) # canonicalization transform to world CSmatrix 

        if len(b[7])>0 and len(b[7]) > 3:
            # adapted_batch[i]['start_goal'] = torch.from_numpy(b[7][0])
            adapted_batch[i]['obj_transform_trans'] = torch.from_numpy(b[7][2])
            adapted_batch[i]['obj_transform_rotation'] = torch.from_numpy(b[7][3])

            if len(b[7]) >= 5:
                adapted_batch[i]['obj_id'] = b[7][4]
            if len(b[7]) >= 6:
                if isinstance(b[7][5], float):
                    adapted_batch[i]['obj_transform_scale'] = torch.from_numpy(np.array([b[7][5]])[None])
                else:
                    adapted_batch[i]['obj_transform_scale'] = torch.from_numpy(b[7][5][None])

    return collate(adapted_batch)


# an adapter to our collate func
# def t2m_collate(batch):
#     # batch.sort(key=lambda x: x[3], reverse=True)
#     adapted_batch = [{
#         'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
#         'text': b[2], #b[0]['caption']
#         'tokens': b[6],
#         'lengths': b[5],
#         'is_transition': torch.zeros(1), # just for eval not really needed
#         'bps_data': [] if b[7][0] == [] else torch.from_numpy(b[7][0]),
#         'start_goal': [] if b[7][1] == [] else torch.from_numpy(b[7][1]),
#         'seq_name': [] if b[7][2] == [] else b[7][2],
#         'move_to_zero_trans': [] if b[7][3] == [] else torch.from_numpy(b[7][3]),
#         'recover_rot_quat': [] if b[7][4] == [] else torch.from_numpy(b[7][4]),
#         'betas': [] if b[7][5] == [] else torch.from_numpy(b[7][5]),
#         'gender': [] if b[7][6] == [] else b[7][6],
#         'seq_len': [] if b[7][7] == [] else b[7][7],
#         'local_rot_6d': [] if b[7][8] == [] else torch.from_numpy(b[7][8]),
#         'root_trans': [] if b[7][9] == [] else torch.from_numpy(b[7][9]),
#         'inpainting_mask': [] if b[7][10] == [] else torch.from_numpy(b[7][10]),
#         'lhand_rhand_mask': [] if b[7][11] == [] else torch.from_numpy(b[7][11]),
#         'inpainted_motion': [] if b[7][12] == [] else torch.from_numpy(b[7][12]),
#     } for b in batch]
#     return collate(adapted_batch)


def babel_eval_collate(batch):
    try:
        adapted_batch = [{
            'inp': torch.from_numpy(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'is_transition': torch.from_numpy(b[7]),
        } for b in batch]
    except TypeError:
        print(5)
    return collate(adapted_batch)

def pw3d_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'other_motion': torch.tensor(b[0].T).float().unsqueeze(1),
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'person_id': b[3],
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

from enum import IntEnum

class motion_type(IntEnum):
    MOTION_0 = 0
    MOTION_1 = 1
    MOTION_0_W_T = 2
    MOTION_1_W_T = 3

def pad_sample_with_zeros(sample, vector_len):
    # pad inp, change lenghts, and pad is transition
    n_feats, _, seq_len = sample['inp'].shape
    len_to_pad = vector_len-seq_len
    torch.zeros_like(sample['inp'])
    is_transition_padding = torch.zeros(len_to_pad)
    inp_padding = torch.zeros((n_feats, 1, len_to_pad))
    sample['inp'] = torch.cat((sample['inp'], inp_padding), dim=2)
    sample['is_transition'] = torch.cat((sample['is_transition'], is_transition_padding))
    return sample

def babel_collate(batch):
    from data_loaders.amass.tools import collate_pairs_and_text
    batch = collate_pairs_and_text(batch)
    bs = len(batch['motion_feats'])
    adapted_batch = []
    for ii in range(bs):
        adapted_batch.append({
            'inp': batch['motion_feats'][ii].permute(1, 0).unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': batch['text'][ii],
            'lengths': batch['length'][ii],
            'is_transition': batch['is_transition'][ii]})
    return collate(adapted_batch)
