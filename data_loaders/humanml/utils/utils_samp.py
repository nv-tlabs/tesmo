import numpy as np
import matplotlib.pyplot as plt
import trimesh
from data_loaders.humanml.scripts.motion_process import process_file
from scipy.spatial.transform import Rotation as R
from data_loaders.humanml.scripts.motion_process import face_joint_indx,uniform_skeleton,tgt_offsets
from data_loaders.humanml.common.quaternion import qbetween_np, quaternion_to_matrix_np, qeuler
from data_loaders.humanml.utils.get_scene_bps import ObjectScene # this is used to get SDF
import torch
import os
import pickle
import skimage
import glob
from tqdm import tqdm
import random

from datetime import datetime

### TODO important to change, used for canonicalization. 
GROUND_PLAN='xy' # for SAMP 
# GROUND_PLAN='xz' # for 3D FRONT SCENE

def get_current_time():
    # Get the current date and time
    current_datetime = datetime.now()

    # Print the current date and time
    print("Current Date and Time:", current_datetime)

    # Extract the date and time components separately
    current_date = current_datetime.date()
    # current_time = current_datetime.time()

    # Print the date and time components
    print("Current Date:", current_date)
    # print("Current Time:", current_time)
    return current_date

# trans_matrix = np.array([[1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]]) # old one.
trans_matrix = np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 1.0, 0.0]]) # !!! this make the orientation with no flip 
# trans_matrix_new = np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, -1.0, 0.0]])
joints_num = 22

# SAMP test split.
#  "armchair019", "chair_mo019", "highstool019", "reebokstep019", "sofa019", "table019", "lie_down_19" 


# MAX_DISTANCE=1.5
class Myclass():
    pass

def get_test_obj_name(id_name):
    
    test_dict = {
        'chair_mo019_stageII': 
        ['409ae7cf-1cbe-4d88-8936-80d192c168b5', 
         '9706_stool_wood'],
        'armchair019_stageII':
        ['3ab2e104-be5b-4617-a849-306ccb29c0ac',
         '19d3b4f9-b76c-4205-981b-977181f0da38',
         '81d1ecd3-9a1c-4ee2-be0e-de9c6c121ed3'],
        'highstool019_stageII':
        ['0f92470c-b703-49fb-b317-fb3e570363aa'],
        'reebokstep020_stageII': 
        ['af727019-da70-40fb-87c4-97870423fd65'],

    }
    assert id_name in test_dict.keys()
    return test_dict[id_name]


def load_motion(amass_npz_fname, start_frame=0, end_frame=None, viz=False, return_all=False, recalculate=False):
    '''
    load all npz files in the amass_npz_fname folder
    merge all the data into one file
    '''
    # import pdb;pdb.set_trace()
    all_merge_file = amass_npz_fname + '_merge.pkl'
    print(f'load motion {all_merge_file}')

    if os.path.exists(all_merge_file) and not recalculate:
        with open(all_merge_file, 'rb') as f:
            bdata = pickle.load(f)
            all_verts_np = bdata['vertices']
            all_joints_np = bdata['joints']
            faces = bdata['faces']
    else:
        all_files = sorted(glob.glob(os.path.join(amass_npz_fname, '*.pkl')))
        
        if int(os.path.basename(all_files[0]).split('.')[0]) == 0:
            all_files = all_files[start_frame:] #20fps

        all_verts_list = []
        all_joints_list = []
        faces = None
        
        for one_file in tqdm(all_files):
            with open(one_file, 'rb') as f:
                bdata = pickle.load(f)
                all_verts_list.append(bdata['body_model_output.vertices'].detach().cpu().numpy())
                all_joints_list.append(bdata['body_model_output.joints'].detach().cpu().numpy())
                if faces is None:
                    faces = bdata['faces']
        all_verts_np = np.concatenate(all_verts_list, axis=0)
        all_joints_np = np.concatenate(all_joints_list, axis=0)
    
        with open(all_merge_file, 'wb') as f:
            pickle.dump({'vertices': all_verts_np,
                        'joints': all_joints_np,
                        'faces': faces}, f)
    if viz:
        vis_body_trans_root(0, all_verts_np, faces) # frame id
    
    if return_all:
        return all_verts_np, all_joints_np, faces

    return all_verts_np

def get_id_list(data_dir, is_train=True, id_frame=None, size=None, post_fix=''):
    new_id_list = glob.glob(os.path.join(data_dir, '*.npy'))
    
    train_split = os.path.join(data_dir, f'train{post_fix}.txt')

    # import pdb;pdb.set_trace()
    # TODO: change the test file.
    test_split = os.path.join(data_dir, f'test{post_fix}.txt')
    # test_split = os.path.join(data_dir, 'test_from_train.txt')
    # import pdb;pdb.set_trace()
    if not os.path.exists(train_split):
        print('save to train and test split to ', train_split)
        # random.shuffle(id_list)
        train_list = [new_id_list[i] for i in range(int(len(new_id_list) * 0.8))]
        test_list = [new_id_list[i] for i in range(int(len(new_id_list) * 0.8), len(new_id_list))]
        
        with open(train_split, 'w') as f:
            f.writelines([one+'\n' for one in train_list])
        with open(test_split, 'w') as f:
            f.writelines([one+'\n' for one in test_list])
    else:
        print('load train and test split from ', train_split, test_split)
        with open(train_split, 'r') as f:
            train_list = f.readlines()
            train_list = [one[:-1] for one in train_list]
        with open(test_split, 'r') as f:
            test_list = f.readlines()
            test_list = [one[:-1] for one in test_list]

    if is_train:
        id_list = train_list
    else:
        new_id_list = test_list
        # import pdb;pdb.set_trace()
        if id_frame is None and size is not None:
            id_list = [one for one in new_id_list if one in id_list][:size]
        elif id_frame is not None:
            id_list = [one for one in new_id_list if one in id_list][id_frame:id_frame+1]
        else:
            id_list = new_id_list

    if 0: # only extract the specific file for debug.
        new_name_list = []
        for one in train_list:
            if '001499' in one or '000492' in one:
                new_name_list.append(one)
        id_list = new_name_list
    
    return id_list


# finetune the data augmentation. is better than training from scratch.
def get_restore_data(restore_data_dir, process_path_name, data_dict, new_name_list, length_list, half_hard_case=False):

    print('load exampls from ', os.path.join(restore_data_dir, process_path_name.split('.')[0]))
    sub_files = glob.glob(os.path.join(restore_data_dir, process_path_name.split('.')[0], '*.pkl'))
    for one_files in sub_files:
        with open(one_files, 'rb') as f:
            data = pickle.load(f)
            dict_name = os.path.basename(one_files).split('.')[0]
            if np.isnan(data['motion']).sum() > 0:
                print('skip :', one_files)
                import pdb;pdb.set_trace()
                continue
            
            ## check hard or easy case.
            if half_hard_case:
                floor_plane_mask_img = data['scene'][:, :].squeeze()
                pose_seq_np = data['motion'].copy()
                length = data['length']
                xs, ys = pose_seq_np[:length, 0], pose_seq_np[:length, 1]
                ys = ys * -1
                H, W = floor_plane_mask_img.shape[:2]
                grid_x, grid_y = np.floor(xs / FLOOR_PLANE_GRID_SCALE).astype(np.int32)+128, np.floor(ys / FLOOR_PLANE_GRID_SCALE).astype(np.int32)+128
                
                kind = get_easy_hard_kind(floor_plane_mask_img, (grid_x[0], grid_y[0]), (grid_x[-1], grid_y[-1]))
                if kind == 0:
                    if easy_cnt >= size//2:
                    # if easy_cnt >= 0:
                        continue
                    easy_cnt += 1
                else:
                    if hard_cnt >= size //2:
                    # if hard_cnt >= size:
                        continue
                    hard_cnt += 1

            data_dict[dict_name] = data
            new_name_list.append(dict_name)
            length_list.append(data['motion'].shape[0])

    return data_dict, new_name_list, length_list
    
def check_point_inside_bbox(point, bbox):
    # check whether a point is inside a bbox;
    # point: [x, y, z]
    # bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
    if point[0] > bbox[0] and point[0] < bbox[3] and point[1] > bbox[1] and point[1] < bbox[4] and point[2] > bbox[2] and point[2] < bbox[5]:
        return True
    else:
        return False

# transition label; 
def get_contact_label(input_joints, obj_list, vis=False, max_distance=1.5):
        # load the object;
        # obj_list = glob.glob(os.path.join(args.object, f'{os.path.basename(pkl_path).split(".")[0]}/*/*'))

        obj_path = obj_list[0]
        obj_mesh = trimesh.load(obj_path, process=False)
        obj_center = np.mean(obj_mesh.vertices, 0)
        xmin, ymin, zmin = obj_mesh.vertices.min(0)
        xmax, ymax, zmax = obj_mesh.vertices.max(0)

        frame_number = input_joints.shape[0]

        dists_list = []
        frame_id = []
        inside_list = []

        for fId in range(0, frame_number):

            joints = input_joints[fId]
            # calculate the eculidean distance between joints and object center.
            dist = np.linalg.norm((joints-obj_center)[:,:2], axis=1).mean() # xy is the ground, z is the height.

            dists_list.append(dist)
            frame_id.append(fId)

            # import pdb;pdb.set_trace()
            inside_list.append(check_point_inside_bbox(joints[0], [xmin, ymin, zmin, xmax, ymax, zmax]))
            # get the near label for it.
        
        if vis:

            # visualize the value of distance over time;
            x_values = range(len(dists_list))

            # Plot the distance values
            plt.plot(x_values, dists_list)
            plt.plot(x_values, inside_list, 'r')
            # Set labels and title
            plt.xlabel('Time')
            plt.ylabel('Distance')
            plt.title('Distance over Time')

        # Display the plot
        # plt.show()
        # assign the interaction label;
        minum_index = np.argmin(dists_list)
        minum_dist = dists_list[minum_index]
        label_list = []

        # use chamfer distance to compute the distance between human mesh and chair mesh;
        # torch.cdist(human_mesh_vertices, chair_mesh_vertices)
        # consider different body part's movement;

        sit = np.array(inside_list) * (np.array(dists_list) < minum_dist+0.20) * np.arange(len(dists_list)) 
        available_list = (np.array(dists_list) < max_distance) * np.arange(len(dists_list)) 
        return sit, inside_list, available_list


def load_mannual_text_dict():
    text_dict = {
        "highstool_stageII": "A person sits down and crosses their right leg over their left leg. #A/DET person/NOUN sits/VERB down/ADP and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool001_stageII": "A person sits down and raises their right leg to rest on the foot of the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool002_stageII": "A person sits down and stretches out both legs.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool003_stageII": "A person sits down and raises both legs to rest on the foot of the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB both/DET legs/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool004_stageII": "A person sits down and raises their right leg to rest on the foot of the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool005_stageII": "A person sits down, raises their right leg, and leans forward.#A/DET person/NOUN sits/VERB down/ADP, raises/VERB their/DET right/ADJ leg/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool006_stageII": "A person sits down and raises both feet to rest on the foot of the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool007_stageII": "A person sits down and raises both feet to rest on the foot of the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool008_stageII": "A person sits down, leans to the right side, and crosses their right leg over their left leg.#A/DET person/NOUN sits/VERB down/ADP, leans/VERB to/ADP the/DET right/ADJ side/NOUN, and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool009_stageII": "A person sits down, crosses their legs on the foot of the chair, and leans forward.#A/DET person/NOUN sits/VERB down/ADP, crosses/VERB their/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool010_stageII": "A person sits down and stretches out both legs.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool011_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward.#A/DET person/NOUN sits/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool012_stageII": "A person sits down, places both legs on the foot of the chair, and sits straight.#A/DET person/NOUN sits/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "highstool013_stageII": "A person sits down, places both legs on the foot of the chair, and rests.#A/DET person/NOUN sits/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ rests/VERB#0.0#0.0",
        "highstool014_stageII": "A person sits down and raises their right leg to place it on the chair.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT place/VERB it/PRON on/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool015_stageII": "A person sits down, bends over, and rests elbows on legs.#A/DET person/NOUN sits/VERB down/ADP, bends/VERB over/ADP, and/CCONJ rests/VERB elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "highstool016_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward.#A/DET person/NOUN sits/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool017_stageII": "A person sits down, places their right leg on the left leg.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ places/VERB their/DET right/ADJ leg/NOUN on/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool018_stageII": "A person sits down and rests.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ rests/VERB#0.0#0.0",
        "highstool019_stageII": "A person sits down and sits straight.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep_stageII": "A person sits down and stretches out both legs.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "reebokstep001_stageII": "A person sits down and stretches out their left leg.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep002_stageII": "A person sits down, stretches out both legs straight, and sits straight.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep003_stageII": "A person sits down, stretches out both legs, and leans forward.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "reebokstep004_stageII": "A person sits down and rests with elbows on legs.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep005_stageII": "A person sits down, slightly stretches out both legs, and rests with elbows on legs.#A/DET person/NOUN sits/VERB down/ADP, slightly/RB stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep006_stageII": "A person sits down, crosses their legs, and sits straight.#A/DET person/NOUN sits/VERB down/ADP, crosses/VERB their/DET legs/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep007_stageII": "A person sits down, stretches the left leg straight, crosses the right leg over the left, and leans back.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ, and/CCONJ leans/VERB back/ADP#0.0#0.0",
        "reebokstep008_stageII": "A person sits down, stretches the left leg straight, places the right leg on the chair, and embraces the right leg.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep009_stageII": "A person sits down, stretches out the left leg, and looks to the right back.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB to/ADP the/DET right/ADJ back/NOUN#0.0#0.0",
        "reebokstep011_stageII": "A person sits down and stretches out both legs straight.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep012_stageII": "A person sits down, stretches out the left leg, and looks down.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep013_stageII": "A person sits down and stretches out both legs straight.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep014_stageII": "A person sits down and stretches out both legs straight.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep015_stageII": "A person sits down, stretches out the left leg, places the right leg on the chair, and embraces the right leg.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep016_stageII": "A person sits down and uses legs to support elbows, resting the head.#A/DET person/NOUN sits/VERB down/ADP and/CCONJ uses/VERB legs/NOUN to/PRT support/VERB elbows/NOUN, resting/VERB the/DET head/NOUN#0.0#0.0",
        "reebokstep017_stageII": "A person sits down, stretches out the left leg, and looks down.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep018_stageII": "A person sits down, stretches out the left leg, and crosses the right leg over the left leg.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep020_stageII": "A person sits down shyly and closely.#A/DET person/NOUN sits/VERB down/ADP shyly/RB and/CCONJ closely/RB#0.0#0.0",
        "reebokstep021_stageII": "A person sits down, stretches out both legs with slightly bent knees.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN with/ADP slightly/RB bent/VBN knees/NOUN#0.0#0.0",
        "reebokstep022_stageII": "A person sits down, stretches out the left leg, and looks down.#A/DET person/NOUN sits/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",

        "armchair001_stageII": "A person sits down and places hands on the arm of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair002_stageII": "A person sits down, stretches out his right leg, and puts hands on the arm of the chair.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET right/ADJ leg/NOUN, and/CCONJ put/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair003_stageII": "A person sits down, places his right leg over the left leg, and raises his right hand close to his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN close/ADV to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair004_stageII": "A person sits down, uses his right hand to pull the left leg onto the right leg, and then places both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET right/ADJ hand/NOUN to/ADP pull/VERB the/DET left/ADJ leg/NOUN onto/ADP the/DET right/ADJ leg/NOUN, and/CCONJ then/ADV place/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair005_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair006_stageII": "A person sits down, places his right leg over the left leg, and uses his left hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET left/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair007_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair008_stageII": "A person sits down, leans back, and lets his hands hang down.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ let/VERB his/DET hands/NOUN hang/VERB down/ADV#0.0#0.0",
        "armchair009_stageII": "A person sits down, stretches out his left leg, and raises his right hand to his head.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair010_stageII": "A person sits down, leans back, stretches his right leg, and raises his left hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET right/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET left/ADJ hand/NOUN#0.0#0.0",
        "armchair011_stageII": "A person sits down, stretches out his left leg, and uses his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair012_stageII": "A person sits down, stretches out his left leg, raises his right hand, and looks at it as if watching a phone.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, raise/VERB his/DET right/ADJ hand/NOUN, and/CCONJ look/VERB at/ADP it/PRON as/SCONJ if/ADP watching/VERB a/DET phone/NOUN#0.0#0.0",
        "armchair013_stageII": "A person sits down, raises his head up, and places his two hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET head/NOUN up/ADV, and/CCONJ place/VERB his/DET two/NUM hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair014_stageII": "A person sits down, crosses his legs, and watches a phone in his right hand.#a/DET person/NOUN sit/VERB down/ADP, cross/VERB his/DET legs/NOUN, and/CCONJ watch/VERB a/DET phone/NOUN in/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "armchair015_stageII": "A person sits down, places his right leg over his left leg, and uses his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB the/DET head/NOUN#0.0#0.0",
        "armchair016_stageII": "A person sits down, leans forward, and uses both legs to support two elbows.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ use/VERB both/DET legs/NOUN to/ADP support/VERB two/NUM elbows/NOUN#0.0#0.0",
        "armchair017_stageII": "A person sits down, leans back, stretches his arms, and places them under his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET arms/NOUN, and/CCONJ place/VERB them/PRON under/ADP his/DET head/NOUN#0.0#0.0",
        "armchair018_stageII": "A person sits down, places his elbow on the arm of the chair, and spreads his hands.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET elbow/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, and/CCONJ spread/VERB his/DET hands/NOUN#0.0#0.0",
        "armchair019_stageII": "A person sits down, places his right leg over the left leg, crosses his hands, and places them on the right arm of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, cross/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET right/ADJ arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair_stageII": "A person sits down, stretches out both legs, and places hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP both/DET legs/NOUN, and/CCONJ place/VERB hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "chair_mo_stageII": "A person sits down and places two elbows on the legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB two/NUM elbows/NOUN on/ADP the/DET legs/NOUN#0.0#0.0",
        "chair_mo001_stageII": "A person sits down, leans forward, and stretches out his left leg.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ stretch/VERB out/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo002_stageII": "A person sits down, stretches out his legs, and crosses his arms.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo003_stageII": "A person sits down and then plays with his phone.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ then/ADV play/VERB with/ADP his/DET phone/NOUN#0.0#0.0",
        "chair_mo004_stageII": "A person sits down, places his hands close, and looks around.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET hands/NOUN close/ADV, and/CCONJ look/VERB around/ADV#0.0#0.0",
        "chair_mo005_stageII": "A person sits down and raises two hands, putting them together.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo006_stageII": "A person sits down and raises two hands, putting them together.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo007_stageII": "A person sits down, leans back, and crosses his arms.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo008_stageII": "A person sits down, leans back, stretches out his legs, and places two hands under his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ place/VERB two/NUM hands/NOUN under/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo009_stageII": "A person sits down, places his right leg on the chair, and hugs his right leg.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ hug/VERB his/DET right/ADJ leg/NOUN#0.0#0.0",
        "chair_mo010_stageII": "A person sits down, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADV his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo011_stageII": "A person sits down, raises his hands, and places them on the back of his head.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET back/NOUN of/AD his/DET head/NOUN#0.0#0.0",
        "chair_mo012_stageII": "A person sits down, leans left, and hangs down his left hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB left/ADV, and/CCONJ hang/VERB down/ADP his/DET left/ADJ hand/NOUN#0.0#0.0",
        "chair_mo013_stageII": "A person sits down, leans back, and raises his hands to the back of his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ raise/VERB his/DET hands/NOUN to/ADP the/DET back/NOUN of/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo014_stageII": "A person sits down and sits on his left leg.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ sit/VERB on/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo015_stageII": "A person sits down, uses his legs to support his elbows, and uses his hands to support his head.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET legs/NOUN to/ADP support/VERB his/DET elbows/NOUN, and/CCONJ use/VERB his/DET hands/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo016_stageII": "A person sits down and raises his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo017_stageII": "A person sits down, leans right, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo018_stageII": "A person sits down and crosses his hands, putting them together on his legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ cross/VERB his/DET hands/NOUN, putting/VERB them/PRON together/ADV on/ADP his/DET legs/NOUN#0.0#0.0",
        "chair_mo019_stageII": "A person sits down, leans right, and places his left hand on his body.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, and/CCONJ place/VERB his/DET left/ADJ hand/NOUN on/ADP his/DET body/NOUN#0.0#0.0"
    }

    walk_to_sit_text_dict = {
        "highstool_stageII": "A person walks and sits down and crosses their right leg over their left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool001_stageII": "A person walks and sits down and raises their right leg to rest on the foot of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool002_stageII": "A person walks and sits down and stretches out both legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool003_stageII": "A person walks and sits down and raises both legs to rest on the foot of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET legs/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool004_stageII": "A person walks and sits down and raises their right leg to rest on the foot of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool005_stageII": "A person walks and sits down, raises their right leg, and leans forward.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raises/VERB their/DET right/ADJ leg/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool006_stageII": "A person walks and sits down and raises both feet to rest on the foot of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool007_stageII": "A person walks and sits down and raises both feet to rest on the foot of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool008_stageII": "A person walks and sits down, leans to the right side, and crosses their right leg over their left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, leans/VERB to/ADP the/DET right/ADJ side/NOUN, and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool009_stageII": "A person walks and sits down, crosses their legs on the foot of the chair, and leans forward.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool010_stageII": "A person walks and sits down and stretches out both legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool011_stageII": "A person walks and sits down, places their right leg on the foot of the chair, and leans forward.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool012_stageII": "A person walks and sits down, places both legs on the foot of the chair, and sits straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "highstool013_stageII": "A person walks and sits down, places both legs on the foot of the chair, and rests.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ rests/VERB#0.0#0.0",
        "highstool014_stageII": "A person walks and sits down and raises their right leg to place it on the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT place/VERB it/PRON on/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool015_stageII": "A person walks and sits down, bends over, and rests elbows on legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, bends/VERB over/ADP, and/CCONJ rests/VERB elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "highstool016_stageII": "A person walks and sits down, places their right leg on the foot of the chair, and leans forward.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool017_stageII": "A person walks and sits down, places their right leg on the left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool018_stageII": "A person walks and sits down and rests.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ rests/VERB#0.0#0.0",
        "highstool019_stageII": "A person walks and sits down and sits straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep_stageII": "A person walks and sits down and stretches out both legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "reebokstep001_stageII": "A person walks and sits down and stretches out their left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep002_stageII": "A person walks and sits down, stretches out both legs straight, and sits straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep003_stageII": "A person walks and sits down, stretches out both legs, and leans forward.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "reebokstep004_stageII": "A person walks and sits down and rests with elbows on legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep005_stageII": "A person walks and sits down, slightly stretches out both legs, and rests with elbows on legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, slightly/RB stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep006_stageII": "A person walks and sits down, crosses their legs, and sits straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep007_stageII": "A person walks and sits down, stretches the left leg straight, crosses the right leg over the left, and leans back.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ, and/CCONJ leans/VERB back/ADP#0.0#0.0",
        "reebokstep008_stageII": "A person walks and sits down, stretches the left leg straight, places the right leg on the chair, and embraces the right leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep009_stageII": "A person walks and sits down, stretches out the left leg, and looks to the right back.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB to/ADP the/DET right/ADJ back/NOUN#0.0#0.0",
        "reebokstep011_stageII": "A person walks and sits down and stretches out both legs straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep012_stageII": "A person walks and sits down, stretches out the left leg, and looks down.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep013_stageII": "A person walks and sits down and stretches out both legs straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep014_stageII": "A person walks and sits down and stretches out both legs straight.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep015_stageII": "A person walks and sits down, stretches out the left leg, places the right leg on the chair, and embraces the right leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep016_stageII": "A person walks and sits down and uses legs to support elbows, resting the head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ uses/VERB legs/NOUN to/PRT support/VERB elbows/NOUN, resting/VERB the/DET head/NOUN#0.0#0.0",
        "reebokstep017_stageII": "A person walks and sits down, stretches out the left leg, and looks down.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep018_stageII": "A person walks and sits down, stretches out the left leg, and crosses the right leg over the left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep020_stageII": "A person walks and sits down shyly and closely.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP shyly/RB and/CCONJ closely/RB#0.0#0.0",
        "reebokstep021_stageII": "A person walks and sits down, stretches out both legs with slightly bent knees.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN with/ADP slightly/RB bent/VBN knees/NOUN#0.0#0.0",
        "reebokstep022_stageII": "A person walks and sits down, stretches out the left leg, and looks down.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",

        "armchair001_stageII": "A person walks and sits down and places hands on the arm of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ place/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair002_stageII": "A person walks and sits down, stretches out his right leg, and puts hands on the arm of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET right/ADJ leg/NOUN, and/CCONJ put/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair003_stageII": "A person walks and sits down, places his right leg over the left leg, and raises his right hand close to his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN close/ADV to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair004_stageII": "A person walks and sits down, uses his right hand to pull the left leg onto the right leg, and then places both hands on the arms of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, use/VERB his/DET right/ADJ hand/NOUN to/ADP pull/VERB the/DET left/ADJ leg/NOUN onto/ADP the/DET right/ADJ leg/NOUN, and/CCONJ then/ADV place/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair005_stageII": "A person walks and sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair006_stageII": "A person walks and sits down, places his right leg over the left leg, and uses his left hand to support his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET left/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair007_stageII": "A person walks and sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair008_stageII": "A person walks and sits down, leans back, and lets his hands hang down.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ let/VERB his/DET hands/NOUN hang/VERB down/ADV#0.0#0.0",
        "armchair009_stageII": "A person walks and sits down, stretches out his left leg, and raises his right hand to his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair010_stageII": "A person walks and sits down, leans back, stretches his right leg, and raises his left hand.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET right/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET left/ADJ hand/NOUN#0.0#0.0",
        "armchair011_stageII": "A person walks and sits down, stretches out his left leg, and uses his right hand to support his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair012_stageII": "A person walks and sits down, stretches out his left leg, raises his right hand, and looks at it as if watching a phone.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, raise/VERB his/DET right/ADJ hand/NOUN, and/CCONJ look/VERB at/ADP it/PRON as/SCONJ if/ADP watching/VERB a/DET phone/NOUN#0.0#0.0",
        "armchair013_stageII": "A person walks and sits down, raises his head up, and places his two hands on the arms of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raise/VERB his/DET head/NOUN up/ADV, and/CCONJ place/VERB his/DET two/NUM hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair014_stageII": "A person walks and sits down, crosses his legs, and watches a phone in his right hand.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, cross/VERB his/DET legs/NOUN, and/CCONJ watch/VERB a/DET phone/NOUN in/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "armchair015_stageII": "A person walks and sits down, places his right leg over his left leg, and uses his right hand to support his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB the/DET head/NOUN#0.0#0.0",
        "armchair016_stageII": "A person walks and sits down, leans forward, and uses both legs to support two elbows.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ use/VERB both/DET legs/NOUN to/ADP support/VERB two/NUM elbows/NOUN#0.0#0.0",
        "armchair017_stageII": "A person walks and sits down, leans back, stretches his arms, and places them under his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET arms/NOUN, and/CCONJ place/VERB them/PRON under/ADP his/DET head/NOUN#0.0#0.0",
        "armchair018_stageII": "A person walks and sits down, places his elbow on the arm of the chair, and spreads his hands.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET elbow/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, and/CCONJ spread/VERB his/DET hands/NOUN#0.0#0.0",
        "armchair019_stageII": "A person walks and sits down, places his right leg over the left leg, crosses his hands, and places them on the right arm of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, cross/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET right/ADJ arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair_stageII": "A person walks and sits down, stretches out both legs, and places hands on the arms of the chair.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP both/DET legs/NOUN, and/CCONJ place/VERB hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "chair_mo_stageII": "A person walks and sits down and places two elbows on the legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ place/VERB two/NUM elbows/NOUN on/ADP the/DET legs/NOUN#0.0#0.0",
        "chair_mo001_stageII": "A person walks and sits down, leans forward, and stretches out his left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ stretch/VERB out/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo002_stageII": "A person walks and sits down, stretches out his legs, and crosses his arms.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo003_stageII": "A person walks and sits down and then plays with his phone.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ then/ADV play/VERB with/ADP his/DET phone/NOUN#0.0#0.0",
        "chair_mo004_stageII": "A person walks and sits down, places his hands close, and looks around.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET hands/NOUN close/ADV, and/CCONJ look/VERB around/ADV#0.0#0.0",
        "chair_mo005_stageII": "A person walks and sits down and raises two hands, putting them together.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo006_stageII": "A person walks and sits down and raises two hands, putting them together.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo007_stageII": "A person walks and sits down, leans back, and crosses his arms.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo008_stageII": "A person walks and sits down, leans back, stretches out his legs, and places two hands under his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ place/VERB two/NUM hands/NOUN under/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo009_stageII": "A person walks and sits down, places his right leg on the chair, and hugs his right leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ hug/VERB his/DET right/ADJ leg/NOUN#0.0#0.0",
        "chair_mo010_stageII": "A person walks and sits down, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADV his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo011_stageII": "A person walks and sits down, raises his hands, and places them on the back of his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raise/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET back/NOUN of/AD his/DET head/NOUN#0.0#0.0",
        "chair_mo012_stageII": "A person walks and sits down, leans left, and hangs down his left hand.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB left/ADV, and/CCONJ hang/VERB down/ADP his/DET left/ADJ hand/NOUN#0.0#0.0",
        "chair_mo013_stageII": "A person walks and sits down, leans back, and raises his hands to the back of his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ raise/VERB his/DET hands/NOUN to/ADP the/DET back/NOUN of/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo014_stageII": "A person walks and sits down and sits on his left leg.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ sit/VERB on/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo015_stageII": "A person walks and sits down, uses his legs to support his elbows, and uses his hands to support his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, use/VERB his/DET legs/NOUN to/ADP support/VERB his/DET elbows/NOUN, and/CCONJ use/VERB his/DET hands/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo016_stageII": "A person walks and sits down and raises his right hand to support his head.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo017_stageII": "A person walks and sits down, leans right, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB right/ADV, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo018_stageII": "A person walks and sits down and crosses his hands, putting them together on his legs.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ cross/VERB his/DET hands/NOUN, putting/VERB them/PRON together/ADV on/ADP his/DET legs/NOUN#0.0#0.0",
        "chair_mo019_stageII": "A person walks and sits down, leans right, and places his left hand on his body.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB right/ADV, and/CCONJ place/VERB his/DET left/ADJ hand/NOUN on/ADP his/DET body/NOUN#0.0#0.0"
    }

    stand_to_sit_text_dict = {

        "highstool_stageII": "A person sits down and crosses their right leg over their left leg.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool001_stageII": "A person sits down and raises their right leg to rest on the foot of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool002_stageII": "A person sits down and stretches out both legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool003_stageII": "A person sits down and raises both legs to rest on the foot of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET legs/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool004_stageII": "A person sits down and raises their right leg to rest on the foot of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool005_stageII": "A person sits down, raises their right leg, and leans forward.#a/DET person/NOUN sit/VERB down/ADP, raises/VERB their/DET right/ADJ leg/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool006_stageII": "A person sits down and raises both feet to rest on the foot of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool007_stageII": "A person sits down and raises both feet to rest on the foot of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool008_stageII": "A person sits down, leans to the right side, and crosses their right leg over their left leg.#a/DET person/NOUN sit/VERB down/ADP, leans/VERB to/ADP the/DET right/ADJ side/NOUN, and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool009_stageII": "A person sits down, crosses their legs on the foot of the chair, and leans forward.#a/DET person/NOUN sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool010_stageII": "A person sits down and stretches out both legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "highstool011_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool012_stageII": "A person sits down, places both legs on the foot of the chair, and sits straight.#a/DET person/NOUN sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "highstool013_stageII": "A person sits down, places both legs on the foot of the chair, and rests.#a/DET person/NOUN sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ rests/VERB#0.0#0.0",
        "highstool014_stageII": "A person sits down and raises their right leg to place it on the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT place/VERB it/PRON on/ADP the/DET chair/NOUN#0.0#0.0",
        "highstool015_stageII": "A person sits down, bends over, and rests elbows on legs.#a/DET person/NOUN sit/VERB down/ADP, bends/VERB over/ADP, and/CCONJ rests/VERB elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "highstool016_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "highstool017_stageII": "A person sits down, places their right leg on the left leg.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "highstool018_stageII": "A person sits down and rests.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ rests/VERB#0.0#0.0",
        "highstool019_stageII": "A person sits down and sits straight.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep_stageII": "A person sits down and stretches out both legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN#0.0#0.0",
        "reebokstep001_stageII": "A person sits down and stretches out their left leg.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP their/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep002_stageII": "A person sits down, stretches out both legs straight, and sits straight.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep003_stageII": "A person sits down, stretches out both legs, and leans forward.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ leans/VERB forward/ADP#0.0#0.0",
        "reebokstep004_stageII": "A person sits down and rests with elbows on legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep005_stageII": "A person sits down, slightly stretches out both legs, and rests with elbows on legs.#a/DET person/NOUN sit/VERB down/ADP, slightly/RB stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN#0.0#0.0",
        "reebokstep006_stageII": "A person sits down, crosses their legs, and sits straight.#a/DET person/NOUN sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN, and/CCONJ sits/VERB straight/ADJ#0.0#0.0",
        "reebokstep007_stageII": "A person sits down, stretches the left leg straight, crosses the right leg over the left, and leans back.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ, and/CCONJ leans/VERB back/ADP#0.0#0.0",
        "reebokstep008_stageII": "A person sits down, stretches the left leg straight, places the right leg on the chair, and embraces the right leg.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep009_stageII": "A person sits down, stretches out the left leg, and looks to the right back.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB to/ADP the/DET right/ADJ back/NOUN#0.0#0.0",
        "reebokstep011_stageII": "A person sits down and stretches out both legs straight.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep012_stageII": "A person sits down, stretches out the left leg, and looks down.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep013_stageII": "A person sits down and stretches out both legs straight.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep014_stageII": "A person sits down and stretches out both legs straight.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ#0.0#0.0",
        "reebokstep015_stageII": "A person sits down, stretches out the left leg, places the right leg on the chair, and embraces the right leg.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN#0.0#0.0",
        "reebokstep016_stageII": "A person sits down and uses legs to support elbows, resting the head.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ uses/VERB legs/NOUN to/PRT support/VERB elbows/NOUN, resting/VERB the/DET head/NOUN#0.0#0.0",
        "reebokstep017_stageII": "A person sits down, stretches out the left leg, and looks down.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "reebokstep018_stageII": "A person sits down, stretches out the left leg, and crosses the right leg over the left leg.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN#0.0#0.0",
        "reebokstep020_stageII": "A person sits down shyly and closely.#a/DET person/NOUN sit/VERB down/ADP shyly/RB and/CCONJ closely/RB#0.0#0.0",
        "reebokstep021_stageII": "A person sits down, stretches out both legs with slightly bent knees.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN with/ADP slightly/RB bent/VBN knees/NOUN#0.0#0.0",
        "reebokstep022_stageII": "A person sits down, stretches out the left leg, and looks down.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP#0.0#0.0",
        "armchair001_stageII": "A person sits down and places hands on the arm of the chair.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair002_stageII": "A person sits down, stretches out his right leg, and puts hands on the arm of the chair.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET right/ADJ leg/NOUN, and/CCONJ put/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair003_stageII": "A person sits down, places his right leg over the left leg, and raises his right hand close to his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN close/ADV to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair004_stageII": "A person sits down, uses his right hand to pull the left leg onto the right leg, and then places both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET right/ADJ hand/NOUN to/ADP pull/VERB the/DET left/ADJ leg/NOUN onto/ADP the/DET right/ADJ leg/NOUN, and/CCONJ then/ADV place/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair005_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair006_stageII": "A person sits down, places his right leg over the left leg, and uses his left hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET left/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair007_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair008_stageII": "A person sits down, leans back, and lets his hands hang down.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ let/VERB his/DET hands/NOUN hang/VERB down/ADV#0.0#0.0",
        "armchair009_stageII": "A person sits down, stretches out his left leg, and raises his right hand to his head.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP his/DET head/NOUN#0.0#0.0",
        "armchair010_stageII": "A person sits down, leans back, stretches his right leg, and raises his left hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET right/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET left/ADJ hand/NOUN#0.0#0.0",
        "armchair011_stageII": "A person sits down, stretches out his left leg, and uses his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "armchair012_stageII": "A person sits down, stretches out his left leg, raises his right hand, and looks at it as if watching a phone.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, raise/VERB his/DET right/ADJ hand/NOUN, and/CCONJ look/VERB at/ADP it/PRON as/SCONJ if/ADP watching/VERB a/DET phone/NOUN#0.0#0.0",
        "armchair013_stageII": "A person sits down, raises his head up, and places his two hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET head/NOUN up/ADV, and/CCONJ place/VERB his/DET two/NUM hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair014_stageII": "A person sits down, crosses his legs, and watches a phone in his right hand.#a/DET person/NOUN sit/VERB down/ADP, cross/VERB his/DET legs/NOUN, and/CCONJ watch/VERB a/DET phone/NOUN in/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "armchair015_stageII": "A person sits down, places his right leg over his left leg, and uses his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB the/DET head/NOUN#0.0#0.0",
        "armchair016_stageII": "A person sits down, leans forward, and uses both legs to support two elbows.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ use/VERB both/DET legs/NOUN to/ADP support/VERB two/NUM elbows/NOUN#0.0#0.0",
        "armchair017_stageII": "A person sits down, leans back, stretches his arms, and places them under his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET arms/NOUN, and/CCONJ place/VERB them/PRON under/ADP his/DET head/NOUN#0.0#0.0",
        "armchair018_stageII": "A person sits down, places his elbow on the arm of the chair, and spreads his hands.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET elbow/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, and/CCONJ spread/VERB his/DET hands/NOUN#0.0#0.0",
        "armchair019_stageII": "A person sits down, places his right leg over the left leg, crosses his hands, and places them on the right arm of the chair.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, cross/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET right/ADJ arm/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "armchair_stageII": "A person sits down, stretches out both legs, and places hands on the arms of the chair.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP both/DET legs/NOUN, and/CCONJ place/VERB hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN#0.0#0.0",
        "chair_mo_stageII": "A person sits down and places two elbows on the legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB two/NUM elbows/NOUN on/ADP the/DET legs/NOUN#0.0#0.0",
        "chair_mo001_stageII": "A person sits down, leans forward, and stretches out his left leg.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ stretch/VERB out/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo002_stageII": "A person sits down, stretches out his legs, and crosses his arms.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo003_stageII": "A person sits down and then plays with his phone.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ then/ADV play/VERB with/ADP his/DET phone/NOUN#0.0#0.0",
        "chair_mo004_stageII": "A person sits down, places his hands close, and looks around.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET hands/NOUN close/ADV, and/CCONJ look/VERB around/ADV#0.0#0.0",
        "chair_mo005_stageII": "A person sits down and raises two hands, putting them together.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo006_stageII": "A person sits down and raises two hands, putting them together.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV#0.0#0.0",
        "chair_mo007_stageII": "A person sits down, leans back, and crosses his arms.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ cross/VERB his/DET arms/NOUN#0.0#0.0",
        "chair_mo008_stageII": "A person sits down, leans back, stretches out his legs, and places two hands under his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ place/VERB two/NUM hands/NOUN under/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo009_stageII": "A person sits down, places his right leg on the chair, and hugs his right leg.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ hug/VERB his/DET right/ADJ leg/NOUN#0.0#0.0",
        "chair_mo010_stageII": "A person sits down, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADV his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo011_stageII": "A person sits down, raises his hands, and places them on the back of his head.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET back/NOUN of/AD his/DET head/NOUN#0.0#0.0",
        "chair_mo012_stageII": "A person sits down, leans left, and hangs down his left hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB left/ADV, and/CCONJ hang/VERB down/ADP his/DET left/ADJ hand/NOUN#0.0#0.0",
        "chair_mo013_stageII": "A person sits down, leans back, and raises his hands to the back of his head.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ raise/VERB his/DET hands/NOUN to/ADP the/DET back/NOUN of/ADP his/DET head/NOUN#0.0#0.0",
        "chair_mo014_stageII": "A person sits down and sits on his left leg.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ sit/VERB on/ADP his/DET left/ADJ leg/NOUN#0.0#0.0",
        "chair_mo015_stageII": "A person sits down, uses his legs to support his elbows, and uses his hands to support his head.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET legs/NOUN to/ADP support/VERB his/DET elbows/NOUN, and/CCONJ use/VERB his/DET hands/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo016_stageII": "A person sits down and raises his right hand to support his head.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN#0.0#0.0",
        "chair_mo017_stageII": "A person sits down, leans right, stretches out his left leg, and hangs down his right hand.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADP his/DET right/ADJ hand/NOUN#0.0#0.0",
        "chair_mo018_stageII": "A person sits down and crosses his hands, putting them together on his legs.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ cross/VERB his/DET hands/NOUN, putting/VERB them/PRON together/ADV on/ADP his/DET legs/NOUN#0.0#0.0",
        "chair_mo019_stageII": "A person sits down, leans right, and places his left hand on his body.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, and/CCONJ place/VERB his/DET left/ADJ hand/NOUN on/ADP his/DET body/NOUN#0.0#0.0"
    }

    stand_sit_stand_text_dict = {
        "highstool_stageII": "A person sits down and crosses their right leg over their left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool001_stageII": "A person sits down and raises their right leg to rest on the foot of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool002_stageII": "A person sits down and stretches out both legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool003_stageII": "A person sits down and raises both legs to rest on the foot of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET legs/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool004_stageII": "A person sits down and raises their right leg to rest on the foot of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool005_stageII": "A person sits down, raises their right leg, and leans forward, then stand up.#a/DET person/NOUN sit/VERB down/ADP, raises/VERB their/DET right/ADJ leg/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool006_stageII": "A person sits down and raises both feet to rest on the foot of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool007_stageII": "A person sits down and raises both feet to rest on the foot of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool008_stageII": "A person sits down, leans to the right side, and crosses their right leg over their left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, leans/VERB to/ADP the/DET right/ADJ side/NOUN, and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool009_stageII": "A person sits down, crosses their legs on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool010_stageII": "A person sits down and stretches out both legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool011_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool012_stageII": "A person sits down, places both legs on the foot of the chair, and sits straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool013_stageII": "A person sits down, places both legs on the foot of the chair, and rests, then stand up.#a/DET person/NOUN sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ rests/VERB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool014_stageII": "A person sits down and raises their right leg to place it on the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT place/VERB it/PRON on/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool015_stageII": "A person sits down, bends over, and rests elbows on legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP, bends/VERB over/ADP, and/CCONJ rests/VERB elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool016_stageII": "A person sits down, places their right leg on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool017_stageII": "A person sits down, places their right leg on the left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool018_stageII": "A person sits down and rests, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ rests/VERB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool019_stageII": "A person sits down and sits straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep_stageII": "A person sits down and stretches out both legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep001_stageII": "A person sits down and stretches out their left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep002_stageII": "A person sits down, stretches out both legs straight, and sits straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep003_stageII": "A person sits down, stretches out both legs, and leans forward, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep004_stageII": "A person sits down and rests with elbows on legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep005_stageII": "A person sits down, slightly stretches out both legs, and rests with elbows on legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP, slightly/RB stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep006_stageII": "A person sits down, crosses their legs, and sits straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep007_stageII": "A person sits down, stretches the left leg straight, crosses the right leg over the left, and leans back, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ, and/CCONJ leans/VERB back/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep008_stageII": "A person sits down, stretches the left leg straight, places the right leg on the chair, and embraces the right leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep009_stageII": "A person sits down, stretches out the left leg, and looks to the right back, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB to/ADP the/DET right/ADJ back/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep011_stageII": "A person sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep012_stageII": "A person sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep013_stageII": "A person sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep014_stageII": "A person sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep015_stageII": "A person sits down, stretches out the left leg, places the right leg on the chair, and embraces the right leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep016_stageII": "A person sits down and uses legs to support elbows, resting the head, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ uses/VERB legs/NOUN to/PRT support/VERB elbows/NOUN, resting/VERB the/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep017_stageII": "A person sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep018_stageII": "A person sits down, stretches out the left leg, and crosses the right leg over the left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep020_stageII": "A person sits down shyly and closely, then stand up.#a/DET person/NOUN sit/VERB down/ADP shyly/RB and/CCONJ closely/RB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep021_stageII": "A person sits down, stretches out both legs with slightly bent knees, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN with/ADP slightly/RB bent/VBN knees/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep022_stageII": "A person sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair001_stageII": "A person sits down and places hands on the arm of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair002_stageII": "A person sits down, stretches out his right leg, and puts hands on the arm of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET right/ADJ leg/NOUN, and/CCONJ put/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair003_stageII": "A person sits down, places his right leg over the left leg, and raises his right hand close to his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN close/ADV to/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair004_stageII": "A person sits down, uses his right hand to pull the left leg onto the right leg, and then places both hands on the arms of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET right/ADJ hand/NOUN to/ADP pull/VERB the/DET left/ADJ leg/NOUN onto/ADP the/DET right/ADJ leg/NOUN, and/CCONJ then/ADV place/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair005_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair006_stageII": "A person sits down, places his right leg over the left leg, and uses his left hand to support his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET left/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair007_stageII": "A person sits down, places his right leg over the left leg, and puts both hands on the arms of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair008_stageII": "A person sits down, leans back, and lets his hands hang down, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ let/VERB his/DET hands/NOUN hang/VERB down/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair009_stageII": "A person sits down, stretches out his left leg, and raises his right hand to his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair010_stageII": "A person sits down, leans back, stretches his right leg, and raises his left hand, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET right/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET left/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair011_stageII": "A person sits down, stretches out his left leg, and uses his right hand to support his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair012_stageII": "A person sits down, stretches out his left leg, raises his right hand, and looks at it as if watching a phone, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, raise/VERB his/DET right/ADJ hand/NOUN, and/CCONJ look/VERB at/ADP it/PRON as/SCONJ if/ADP watching/VERB a/DET phone/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair013_stageII": "A person sits down, raises his head up, and places his two hands on the arms of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET head/NOUN up/ADV, and/CCONJ place/VERB his/DET two/NUM hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair014_stageII": "A person sits down, crosses his legs, and watches a phone in his right hand, then stand up.#a/DET person/NOUN sit/VERB down/ADP, cross/VERB his/DET legs/NOUN, and/CCONJ watch/VERB a/DET phone/NOUN in/ADP his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair015_stageII": "A person sits down, places his right leg over his left leg, and uses his right hand to support his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB the/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair016_stageII": "A person sits down, leans forward, and uses both legs to support two elbows, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ use/VERB both/DET legs/NOUN to/ADP support/VERB two/NUM elbows/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair017_stageII": "A person sits down, leans back, stretches his arms, and places them under his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET arms/NOUN, and/CCONJ place/VERB them/PRON under/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair018_stageII": "A person sits down, places his elbow on the arm of the chair, and spreads his hands, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET elbow/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, and/CCONJ spread/VERB his/DET hands/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair019_stageII": "A person sits down, places his right leg over the left leg, crosses his hands, and places them on the right arm of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, cross/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET right/ADJ arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair_stageII": "A person sits down, stretches out both legs, and places hands on the arms of the chair, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP both/DET legs/NOUN, and/CCONJ place/VERB hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo_stageII": "A person sits down and places two elbows on the legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ place/VERB two/NUM elbows/NOUN on/ADP the/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo001_stageII": "A person sits down, leans forward, and stretches out his left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo002_stageII": "A person sits down, stretches out his legs, and crosses his arms, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ cross/VERB his/DET arms/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo003_stageII": "A person sits down and then plays with his phone, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ then/ADV play/VERB with/ADP his/DET phone/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo004_stageII": "A person sits down, places his hands close, and looks around, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET hands/NOUN close/ADV, and/CCONJ look/VERB around/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo005_stageII": "A person sits down and raises two hands, putting them together, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo006_stageII": "A person sits down and raises two hands, putting them together, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo007_stageII": "A person sits down, leans back, and crosses his arms, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ cross/VERB his/DET arms/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo008_stageII": "A person sits down, leans back, stretches out his legs, and places two hands under his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ place/VERB two/NUM hands/NOUN under/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo009_stageII": "A person sits down, places his right leg on the chair, and hugs his right leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ hug/VERB his/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo010_stageII": "A person sits down, stretches out his left leg, and hangs down his right hand, then stand up.#a/DET person/NOUN sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADV his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo011_stageII": "A person sits down, raises his hands, and places them on the back of his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, raise/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET back/NOUN of/AD his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo012_stageII": "A person sits down, leans left, and hangs down his left hand, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB left/ADV, and/CCONJ hang/VERB down/ADP his/DET left/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo013_stageII": "A person sits down, leans back, and raises his hands to the back of his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ raise/VERB his/DET hands/NOUN to/ADP the/DET back/NOUN of/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo014_stageII": "A person sits down and sits on his left leg, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ sit/VERB on/ADP his/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo015_stageII": "A person sits down, uses his legs to support his elbows, and uses his hands to support his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP, use/VERB his/DET legs/NOUN to/ADP support/VERB his/DET elbows/NOUN, and/CCONJ use/VERB his/DET hands/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo016_stageII": "A person sits down and raises his right hand to support his head, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo017_stageII": "A person sits down, leans right, stretches out his left leg, and hangs down his right hand, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADP his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo018_stageII": "A person sits down and crosses his hands, putting them together on his legs, then stand up.#a/DET person/NOUN sit/VERB down/ADP and/CCONJ cross/VERB his/DET hands/NOUN, putting/VERB them/PRON together/ADV on/ADP his/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo019_stageII": "A person sits down, leans right, and places his left hand on his body, then stand up.#a/DET person/NOUN sit/VERB down/ADP, lean/VERB right/ADV, and/CCONJ place/VERB his/DET left/ADJ hand/NOUN on/ADP his/DET body/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0"
    }

    walk_sit_stand_text_dict = {
        "highstool_stageII": "A person walks and sits down and crosses their right leg over their left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool001_stageII": "A person walks and sits down and raises their right leg to rest on the foot of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool002_stageII": "A person walks and sits down and stretches out both legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool003_stageII": "A person walks and sits down and raises both legs to rest on the foot of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET legs/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool004_stageII": "A person walks and sits down and raises their right leg to rest on the foot of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool005_stageII": "A person walks and sits down, raises their right leg, and leans forward, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raises/VERB their/DET right/ADJ leg/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool006_stageII": "A person walks and sits down and raises both feet to rest on the foot of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool007_stageII": "A person walks and sits down and raises both feet to rest on the foot of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB both/DET feet/NOUN to/PRT rest/VERB on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool008_stageII": "A person walks and sits down, leans to the right side, and crosses their right leg over their left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, leans/VERB to/ADP the/DET right/ADJ side/NOUN, and/CCONJ crosses/VERB their/DET right/ADJ leg/NOUN over/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool009_stageII": "A person walks and sits down, crosses their legs on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool010_stageII": "A person walks and sits down and stretches out both legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool011_stageII": "A person walks and sits down, places their right leg on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool012_stageII": "A person walks and sits down, places both legs on the foot of the chair, and sits straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool013_stageII": "A person walks and sits down, places both legs on the foot of the chair, and rests, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB both/DET legs/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ rests/VERB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool014_stageII": "A person walks and sits down and raises their right leg to place it on the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raises/VERB their/DET right/ADJ leg/NOUN to/PRT place/VERB it/PRON on/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool015_stageII": "A person walks and sits down, bends over, and rests elbows on legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, bends/VERB over/ADP, and/CCONJ rests/VERB elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool016_stageII": "A person walks and sits down, places their right leg on the foot of the chair, and leans forward, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP the/DET foot/NOUN of/ADP the/DET chair/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool017_stageII": "A person walks and sits down, places their right leg on the left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, places/VERB their/DET right/ADJ leg/NOUN on/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool018_stageII": "A person walks and sits down and rests, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ rests/VERB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "highstool019_stageII": "A person walks and sits down and sits straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep_stageII": "A person walks and sits down and stretches out both legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep001_stageII": "A person walks and sits down and stretches out their left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP their/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep002_stageII": "A person walks and sits down, stretches out both legs straight, and sits straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep003_stageII": "A person walks and sits down, stretches out both legs, and leans forward, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ leans/VERB forward/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep004_stageII": "A person walks and sits down and rests with elbows on legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep005_stageII": "A person walks and sits down, slightly stretches out both legs, and rests with elbows on legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, slightly/RB stretches/VERB out/ADP both/DET legs/NOUN, and/CCONJ rests/VERB with/ADP elbows/NOUN on/ADP legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep006_stageII": "A person walks and sits down, crosses their legs, and sits straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, crosses/VERB their/DET legs/NOUN, and/CCONJ sits/VERB straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep007_stageII": "A person walks and sits down, stretches the left leg straight, crosses the right leg over the left, and leans back, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ, and/CCONJ leans/VERB back/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep008_stageII": "A person walks and sits down, stretches the left leg straight, places the right leg on the chair, and embraces the right leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB the/DET left/ADJ leg/NOUN straight/ADJ, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep009_stageII": "A person walks and sits down, stretches out the left leg, and looks to the right back, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB to/ADP the/DET right/ADJ back/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep011_stageII": "A person walks and sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep012_stageII": "A person walks and sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep013_stageII": "A person walks and sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep014_stageII": "A person walks and sits down and stretches out both legs straight, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ stretches/VERB out/ADP both/DET legs/NOUN straight/ADJ, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep015_stageII": "A person walks and sits down, stretches out the left leg, places the right leg on the chair, and embraces the right leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, places/VERB the/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ embraces/VERB the/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep016_stageII": "A person walks and sits down and uses legs to support elbows, resting the head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ uses/VERB legs/NOUN to/PRT support/VERB elbows/NOUN, resting/VERB the/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep017_stageII": "A person walks and sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep018_stageII": "A person walks and sits down, stretches out the left leg, and crosses the right leg over the left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ crosses/VERB the/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep020_stageII": "A person walks and sits down shyly and closely, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP shyly/RB and/CCONJ closely/RB, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep021_stageII": "A person walks and sits down, stretches out both legs with slightly bent knees, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP both/DET legs/NOUN with/ADP slightly/RB bent/VBN knees/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "reebokstep022_stageII": "A person walks and sits down, stretches out the left leg, and looks down, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretches/VERB out/ADP the/DET left/ADJ leg/NOUN, and/CCONJ looks/VERB down/ADP, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair001_stageII": "A person walks and sits down and places hands on the arm of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ place/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair002_stageII": "A person walks and sits down, stretches out his right leg, and puts hands on the arm of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET right/ADJ leg/NOUN, and/CCONJ put/VERB hands/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair003_stageII": "A person walks and sits down, places his right leg over the left leg, and raises his right hand close to his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN close/ADV to/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair004_stageII": "A person walks and sits down, uses his right hand to pull the left leg onto the right leg, and then places both hands on the arms of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, use/VERB his/DET right/ADJ hand/NOUN to/ADP pull/VERB the/DET left/ADJ leg/NOUN onto/ADP the/DET right/ADJ leg/NOUN, and/CCONJ then/ADV place/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair005_stageII": "A person walks and sits down, places his right leg over the left leg, and puts both hands on the arms of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair006_stageII": "A person walks and sits down, places his right leg over the left leg, and uses his left hand to support his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET left/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair007_stageII": "A person walks and sits down, places his right leg over the left leg, and puts both hands on the arms of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, and/CCONJ put/VERB both/DET hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair008_stageII": "A person walks and sits down, leans back, and lets his hands hang down, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ let/VERB his/DET hands/NOUN hang/VERB down/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair009_stageII": "A person walks and sits down, stretches out his left leg, and raises his right hand to his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair010_stageII": "A person walks and sits down, leans back, stretches his right leg, and raises his left hand, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET right/ADJ leg/NOUN, and/CCONJ raise/VERB his/DET left/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair011_stageII": "A person walks and sits down, stretches out his left leg, and uses his right hand to support his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair012_stageII": "A person walks and sits down, stretches out his left leg, raises his right hand, and looks at it as if watching a phone, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, raise/VERB his/DET right/ADJ hand/NOUN, and/CCONJ look/VERB at/ADP it/PRON as/SCONJ if/ADP watching/VERB a/DET phone/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair013_stageII": "A person walks and sits down, raises his head up, and places his two hands on the arms of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raise/VERB his/DET head/NOUN up/ADV, and/CCONJ place/VERB his/DET two/NUM hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair014_stageII": "A person walks and sits down, crosses his legs, and watches a phone in his right hand, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, cross/VERB his/DET legs/NOUN, and/CCONJ watch/VERB a/DET phone/NOUN in/ADP his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair015_stageII": "A person walks and sits down, places his right leg over his left leg, and uses his right hand to support his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP his/DET left/ADJ leg/NOUN, and/CCONJ use/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB the/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair016_stageII": "A person walks and sits down, leans forward, and uses both legs to support two elbows, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ use/VERB both/DET legs/NOUN to/ADP support/VERB two/NUM elbows/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair017_stageII": "A person walks and sits down, leans back, stretches his arms, and places them under his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB his/DET arms/NOUN, and/CCONJ place/VERB them/PRON under/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair018_stageII": "A person walks and sits down, places his elbow on the arm of the chair, and spreads his hands, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET elbow/NOUN on/ADP the/DET arm/NOUN of/ADP the/DET chair/NOUN, and/CCONJ spread/VERB his/DET hands/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair019_stageII": "A person walks and sits down, places his right leg over the left leg, crosses his hands, and places them on the right arm of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN over/ADP the/DET left/ADJ leg/NOUN, cross/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET right/ADJ arm/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "armchair_stageII": "A person walks and sits down, stretches out both legs, and places hands on the arms of the chair, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP both/DET legs/NOUN, and/CCONJ place/VERB hands/NOUN on/ADP the/DET arms/NOUN of/ADP the/DET chair/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo_stageII": "A person walks and sits down and places two elbows on the legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ place/VERB two/NUM elbows/NOUN on/ADP the/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo001_stageII": "A person walks and sits down, leans forward, and stretches out his left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB forward/ADV, and/CCONJ stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo002_stageII": "A person walks and sits down, stretches out his legs, and crosses his arms, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ cross/VERB his/DET arms/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo003_stageII": "A person walks and sits down and then plays with his phone, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ then/ADV play/VERB with/ADP his/DET phone/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo004_stageII": "A person walks and sits down, places his hands close, and looks around, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET hands/NOUN close/ADV, and/CCONJ look/VERB around/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo005_stageII": "A person walks and sits down and raises two hands, putting them together, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo006_stageII": "A person walks and sits down and raises two hands, putting them together, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB two/NUM hands/NOUN, putting/VERB them/PRON together/ADV, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo007_stageII": "A person walks and sits down, leans back, and crosses his arms, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ cross/VERB his/DET arms/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo008_stageII": "A person walks and sits down, leans back, stretches out his legs, and places two hands under his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, stretch/VERB out/ADP his/DET legs/NOUN, and/CCONJ place/VERB two/NUM hands/NOUN under/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo009_stageII": "A person walks and sits down, places his right leg on the chair, and hugs his right leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, place/VERB his/DET right/ADJ leg/NOUN on/ADP the/DET chair/NOUN, and/CCONJ hug/VERB his/DET right/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo010_stageII": "A person walks and sits down, stretches out his left leg, and hangs down his right hand, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADV his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo011_stageII": "A person walks and sits down, raises his hands, and places them on the back of his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, raise/VERB his/DET hands/NOUN, and/CCONJ place/VERB them/PRON on/ADP the/DET back/NOUN of/AD his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo012_stageII": "A person walks and sits down, leans left, and hangs down his left hand, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB left/ADV, and/CCONJ hang/VERB down/ADP his/DET left/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo013_stageII": "A person walks and sits down, leans back, and raises his hands to the back of his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB back/ADV, and/CCONJ raise/VERB his/DET hands/NOUN to/ADP the/DET back/NOUN of/ADP his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo014_stageII": "A person walks and sits down and sits on his left leg, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ sit/VERB on/ADP his/DET left/ADJ leg/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo015_stageII": "A person walks and sits down, uses his legs to support his elbows, and uses his hands to support his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, use/VERB his/DET legs/NOUN to/ADP support/VERB his/DET elbows/NOUN, and/CCONJ use/VERB his/DET hands/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo016_stageII": "A person walks and sits down and raises his right hand to support his head, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ raise/VERB his/DET right/ADJ hand/NOUN to/ADP support/VERB his/DET head/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo017_stageII": "A person walks and sits down, leans right, stretches out his left leg, and hangs down his right hand, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB right/ADV, stretch/VERB out/ADP his/DET left/ADJ leg/NOUN, and/CCONJ hang/VERB down/ADP his/DET right/ADJ hand/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo018_stageII": "A person walks and sits down and crosses his hands, putting them together on his legs, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP and/CCONJ cross/VERB his/DET hands/NOUN, putting/VERB them/PRON together/ADV on/ADP his/DET legs/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0",
        "chair_mo019_stageII": "A person walks and sits down, leans right, and places his left hand on his body, then stand up.#a/DET person/NOUN walk/VERB and/CCONJ sit/VERB down/ADP, lean/VERB right/ADV, and/CCONJ place/VERB his/DET left/ADJ hand/NOUN on/ADP his/DET body/NOUN, then/ADV stand/VERB up/ADP#0.0#0.0"

    }

    # 'stand_sit_stand': 'a person sits on a chair behind them and then stands back up.#a/DET person/NOUN sit/VERB on/ADP a/DET chair/NOUN behind/ADP them/PRON and/CCONJ then/ADV stand/VERB back/ADV up/ADV#0.0#0.0',
    #         'walk_to_sit_to_stand': 'a person walks, and sits down on an object, and then stands back up.#a/DET person/NOUN walk/VERB, sit/VERB down/ADP on/ADP an/DET object/NOUN, and/CCONJ then/ADV stand/VERB back/ADV up/ADV#0.0#0.0',
    # add highstool, reebok;
    return text_dict, walk_to_sit_text_dict, stand_to_sit_text_dict, stand_sit_stand_text_dict, walk_sit_stand_text_dict

def load_mannual_dict():
    
    motion_cfg1 = {
        "armchair_stageII": {"start": 64, "stand_start": 275, "sit_start": 336, "sit_end": 547, "stand_end": 590, "end": 726},
        "armchair001_stageII": {"start": 97, "stand_start": 337, "sit_start": 424, "sit_end": 612, "stand_end": 646, "end": 804},
        "armchair002_stageII": {"start": 86, "stand_start": 237, "sit_start": 294, "sit_end": 511, "stand_end": 541, "end": 657},
        "armchair003_stageII": {"start": 160, "stand_start": 340, "sit_start": 394, "sit_end": 671, "stand_end": 700, "end": 800},
        "armchair004_stageII": {"start": 130, "stand_start": 330, "sit_start": 390, "sit_end": 680, "stand_end": 722, "end": 821},
        "armchair005_stageII": {"start": 230, "stand_start": 330, "sit_start": 380, "sit_end": 650, "stand_end": 690, "end": 750},
        "armchair006_stageII": {"start": 48, "stand_start": 198, "sit_start": 254, "sit_end": 578, "stand_end": 608, "end": 720},
        "armchair007_stageII": {"start": 86, "stand_start": 325, "sit_start": 387, "sit_end": 584, "stand_end": 615, "end": 740},
        "armchair008_stageII": {"start": 100, "stand_start": 196, "sit_start": 248, "sit_end": 434, "stand_end": 459, "end": 549},
        "armchair009_stageII": {"start": 53, "stand_start": 199, "sit_start": 246, "sit_end": 437, "stand_end": 460, "end": 500},
        "armchair010_stageII": {"start": 100, "stand_start": 207, "sit_start": 253, "sit_end": 506, "stand_end": 547, "end": 674},
        "armchair011_stageII": {"start": 181, "stand_start": 264, "sit_start": 335, "sit_end": 509, "stand_end": 548, "end": 650},
        "armchair012_stageII": {"start": 180, "stand_start": 275, "sit_start": 377, "sit_end": 737, "stand_end": 788, "end": 860},
        "armchair013_stageII": {"start": 252, "stand_start": 339, "sit_start": 390, "sit_end": 668, "stand_end": 700, "end": 762},
        "armchair014_stageII": {"start": 200, "stand_start": 400, "sit_start": 444, "sit_end": 709, "stand_end": 737, "end": 870},
        "armchair015_stageII": {"start": 200, "stand_start": 367, "sit_start": 437, "sit_end": 672, "stand_end": 736, "end": 860},
        "armchair016_stageII": {"start": 296, "stand_start": 415, "sit_start": 493, "sit_end": 641, "stand_end": 674, "end": 790},
        "armchair017_stageII": {"start": 363, "stand_start": 440, "sit_start": 520, "sit_end": 870, "stand_end": 910, "end": 1000},
        "armchair018_stageII": {"start": 198, "stand_start": 312, "sit_start": 360, "sit_end": 537, "stand_end": 567, "end": 690},
        "armchair019_stageII": {"start": 135, "stand_start": 268, "sit_start": 340, "sit_end": 570, "stand_end": 609, "end": 710},
        

        "chair_mo_stageII": {'start': 170, 'stand_start': 320, 'sit_start': 420, 'sit_end': 690, 'stand_end': 750, 'end': 850},
        "chair_mo001_stageII": {'start': 100, 'stand_start': 300, 'sit_start': 420, 'sit_end': 580, 'stand_end': 650, 'end': 750},
        "chair_mo002_stageII": {'start': 100, 'stand_start': 210, 'sit_start': 320, 'sit_end': 580, 'stand_end': 660, 'end': 750},
        "chair_mo003_stageII": {'start': 200, 'stand_start': 330, 'sit_start': 430, 'sit_end': 630, 'stand_end': 720, 'end': 800},
        "chair_mo004_stageII": {'start': 120, 'stand_start': 330, 'sit_start': 430, 'sit_end': 600, 'stand_end': 680, 'end': 760},
        "chair_mo005_stageII": {'start': 210, 'stand_start': 350, 'sit_start': 460, 'sit_end': 660, 'stand_end': 860, 'end': 950},
        "chair_mo006_stageII": {'start': 100, 'stand_start': 360, 'sit_start': 470, 'sit_end': 1060, 'stand_end': 1100, 'end': 1180},
        "chair_mo007_stageII": {'start': 120, 'stand_start': 320, 'sit_start': 400, 'sit_end': 720, 'stand_end': 810, 'end': 870},

        "chair_mo008_stageII": {'start': 140, 'stand_start': 230, 'sit_start': 320, 'sit_end': 650, 'stand_end': 760, 'end': 880},
        "chair_mo009_stageII": {'start': 110, 'stand_start': 220, 'sit_start': 380, 'sit_end': 560, 'stand_end': 710, 'end': 800},
        "chair_mo010_stageII": {'start': 150, 'stand_start': 260, 'sit_start': 370, 'sit_end': 730, 'stand_end': 810, 'end': 890},
        "chair_mo011_stageII": {'start': 110, 'stand_start': 450, 'sit_start': 540, 'sit_end': 830, 'stand_end': 940, 'end': 990},
        "chair_mo012_stageII": {'start': 90, 'stand_start': 180, 'sit_start': 300, 'sit_end': 570, 'stand_end': 650, 'end': 720},
        "chair_mo013_stageII": {'start': 130, 'stand_start': 340, 'sit_start': 480, 'sit_end': 700, 'stand_end': 830, 'end': 1000},
        "chair_mo014_stageII": {'start': 150, 'stand_start': 200, 'sit_start': 380, 'sit_end': 650, 'stand_end': 720, 'end': 780},
        "chair_mo015_stageII": {'start': 150, 'stand_start': 340, 'sit_start': 480, 'sit_end': 620, 'stand_end': 700, 'end': 760},
        "chair_mo016_stageII": {'start': 140, 'stand_start': 280, 'sit_start': 380, 'sit_end': 670, 'stand_end': 800, 'end': 900},
        "chair_mo017_stageII": {'start': 140, 'stand_start': 230, 'sit_start': 370, 'sit_end': 580, 'stand_end': 690, 'end': 780},
        "chair_mo018_stageII": {'start': 150, 'stand_start': 420, 'sit_start': 500, 'sit_end': 750, 'stand_end': 780, 'end': 860},
        "chair_mo019_stageII": {'start': 130, 'stand_start': 260, 'sit_start': 430, 'sit_end': 550, 'stand_end': 600, 'end': 660},

    }
    motion_cfg = { # fbx is 30 fps;
            # "armchair009_stageII": {"start": 48, "stand_start": 198, "sit_start": 254, "sit_end": 578, "stand_end": 608, "end": 720},
            # "armchair016_stageII": {"start": 86, "stand_start": 325, "sit_start": 387, "sit_end": 584, "stand_end": 615, "end": 740},
            # "armchair017_stageII": {"start": 100, "stand_start": 196, "sit_start": 248, "sit_end": 434, "stand_end": 459, "end": 549},
            #  new labelled arm data:
            
            # ### ! needs to get new motion data; || old version is already good.
            ### new labelled data:
            "highstool001_stageII": {"start": 100, "stand_start": 317, "sit_start": 360, "sit_end": 580, "stand_end": 620, "end": 810},
            "highstool002_stageII": {"start": 100, "stand_start": 380, "sit_start": 420, "sit_end": 700, "stand_end": 725, "end": 850},
            "highstool003_stageII": {"start": 200, "stand_start": 360, "sit_start": 427, "sit_end": 695, "stand_end": 740, "end": 850},
            "highstool004_stageII": {"start": 100, "stand_start": 203, "sit_start": 285, "sit_end": 555, "stand_end": 590, "end": 690},
            "highstool005_stageII": {"start": 120, "stand_start": 200, "sit_start": 260, "sit_end": 480, "stand_end": 523, "end": 652},
            "highstool006_stageII": {"start": 170, "stand_start": 323, "sit_start": 378, "sit_end": 575, "stand_end": 620, "end": 690},
            "highstool007_stageII": {"start": 130, "stand_start": 230, "sit_start": 315, "sit_end": 568, "stand_end": 610, "end": 712},
            "highstool009_stageII": {"start": 198, "stand_start": 312, "sit_start": 388, "sit_end": 600, "stand_end": 643, "end": 780},
            "highstool010_stageII": {"start": 150, "stand_start": 290, "sit_start": 340, "sit_end": 678, "stand_end": 713, "end": 848},
            "highstool011_stageII": {"start": 190, "stand_start": 280, "sit_start": 345, "sit_end": 558, "stand_end": 620, "end": 730},
            "highstool012_stageII": {"start": 150, "stand_start": 280, "sit_start": 340, "sit_end": 670, "stand_end": 708, "end": 810},
            "highstool013_stageII": {"start": 160, "stand_start": 360, "sit_start": 430, "sit_end": 620, "stand_end": 690, "end": 770},
            "highstool014_stageII": {"start": 150, "stand_start": 373, "sit_start": 430, "sit_end": 666, "stand_end": 720, "end": 820},
            "highstool015_stageII": {"start": 180, "stand_start": 290, "sit_start": 350, "sit_end": 600, "stand_end": 660, "end": 760},
            "highstool016_stageII": {"start": 490, "stand_start": 552, "sit_start": 605, "sit_end": 834, "stand_end": 873, "end": 950},
            "highstool017_stageII": {"start": 117, "stand_start": 185, "sit_start": 250, "sit_end": 820, "stand_end": 870, "end": 940},
            "highstool018_stageII": {"start": 200, "stand_start": 310, "sit_start": 390, "sit_end": 700, "stand_end": 775, "end": 870},
            "highstool019_stageII": {"start": 150, "stand_start": 270, "sit_start": 320, "sit_end": 555, "stand_end": 605, "end": 691},
            "highstool_stageII": {"start": 200, "stand_start": 315, "sit_start": 380, "sit_end": 546, "stand_end": 600, "end": 680},
            
            "reebokstep_stageII": {"start": 85, "stand_start": 210, "sit_start": 280, "sit_end": 560, "stand_end": 600, "end": 700},
            "reebokstep001_stageII": {"start": 100, "stand_start": 176, "sit_start": 256, "sit_end": 402, "stand_end": 452, "end": 640},
            "reebokstep002_stageII": {"start": 160, "stand_start": 230, "sit_start": 312, "sit_end": 704, "stand_end": 748, "end": 860},
            "reebokstep003_stageII": {"start": 100, "stand_start": 210, "sit_start": 260, "sit_end": 496, "stand_end": 550, "end": 650},
            "reebokstep004_stageII": {"start": 110, "stand_start": 320, "sit_start": 380, "sit_end": 670, "stand_end": 740, "end": 850},
            "reebokstep005_stageII": {"start": 86, "stand_start": 280, "sit_start": 350, "sit_end": 834, "stand_end": 881, "end": 1000},
            "reebokstep006_stageII": {"start": 159, "stand_start": 337, "sit_start": 390, "sit_end": 750, "stand_end": 814, "end": 900},
            "reebokstep007_stageII": {"start": 160, "stand_start": 238, "sit_start": 320, "sit_end": 700, "stand_end": 747, "end": 865},
            "reebokstep008_stageII": {"start": 320, "stand_start": 420, "sit_start": 478, "sit_end": 850, "stand_end": 890, "end": 1020},
            "reebokstep009_stageII": {"start": 440, "stand_start": 586, "sit_start": 630, "sit_end": 956, "stand_end": 1020, "end": 1122},
            "reebokstep011_stageII": {"start": 300, "stand_start": 394, "sit_start": 451, "sit_end": 803, "stand_end": 847, "end": 960},
            "reebokstep012_stageII": {"start": 93, "stand_start": 241, "sit_start": 320, "sit_end": 589, "stand_end": 640, "end": 770},
            "reebokstep013_stageII": {"start": 140, "stand_start": 370, "sit_start": 427, "sit_end": 690, "stand_end": 768, "end": 905},
            "reebokstep014_stageII": {"start": 60, "stand_start": 220, "sit_start": 283, "sit_end": 629, "stand_end": 682, "end": 750},
            "reebokstep015_stageII": {"start": 70, "stand_start": 209, "sit_start": 276, "sit_end": 641, "stand_end": 715, "end": 822},
            "reebokstep016_stageII": {"start": 47, "stand_start": 264, "sit_start": 340, "sit_end": 600, "stand_end": 694, "end": 784},
            "reebokstep017_stageII": {"start": 111, "stand_start": 280, "sit_start": 340, "sit_end": 590, "stand_end": 640, "end": 700},
            "reebokstep018_stageII": {"start": 74, "stand_start": 261, "sit_start": 318, "sit_end": 620, "stand_end": 685, "end": 790},
            "reebokstep020_stageII": {"start": 87, "stand_start": 240, "sit_start": 310, "sit_end": 496, "stand_end": 542, "end": 634},
            "reebokstep021_stageII": {"start": 57, "stand_start": 314, "sit_start": 368, "sit_end": 698, "stand_end": 738, "end": 813},
            "reebokstep022_stageII": {"start": 100, "stand_start": 290, "sit_start": 350, "sit_end": 612, "stand_end": 653, "end": 760},

            ### add sit on sofa
            # new label.
            "sofa_stageII": {'start': 120, 'stand_start': 250, 'sit_start': 387, 'sit_end': 470, 'stand_end': 570, 'end': 670},
            "sofa001_stageII": {'start': 130, 'stand_start': 280, 'sit_start': 420, 'sit_end': 600, 'stand_end': 670, 'end': 770},
            "sofa002_stageII": {'start': 140, 'stand_start': 250, 'sit_start': 390, 'sit_end': 540, 'stand_end': 640, 'end': 700},
            "sofa003_stageII": {'start': 130, 'stand_start': 240, 'sit_start': 320, 'sit_end': 450, 'stand_end': 490, 'end': 590},
            "sofa004_stageII": {'start': 120, 'stand_start': 350, 'sit_start': 430, 'sit_end': 570, 'stand_end': 630, 'end': 750},
            "sofa005_stageII": {'start': 120, 'stand_start': 230, 'sit_start': 320, 'sit_end': 480, 'stand_end': 560, 'end': 630},
            "sofa006_stageII": {'start': 120, 'stand_start': 280, 'sit_start': 370, 'sit_end': 500, 'stand_end': 560, 'end': 720},
            "sofa007_stageII": {'start': 100, 'stand_start': 250, 'sit_start': 350, 'sit_end': 470, 'stand_end': 590, 'end': 670},
            "sofa008_stageII": {'start': 141, 'stand_start': 225, 'sit_start': 340, 'sit_end': 430, 'stand_end': 520, 'end': 630},
            "sofa009_stageII": {'start': 120, 'stand_start': 220, 'sit_start': 350, 'sit_end': 460, 'stand_end': 560, 'end': 700},
            "sofa010_stageII": {'start': 90, 'stand_start': 210, 'sit_start': 350, 'sit_end': 550, 'stand_end': 620, 'end': 690},
            "sofa011_stageII": {'start': 110, 'stand_start': 360, 'sit_start': 440, 'sit_end': 540, 'stand_end': 620, 'end': 700},
            "sofa012_stageII": {'start': 130, 'stand_start': 290, 'sit_start': 370, 'sit_end': 460, 'stand_end': 500, 'end': 700},
            "sofa013_stageII": {'start': 140, 'stand_start': 230, 'sit_start': 330, 'sit_end': 470, 'stand_end': 550, 'end': 640},
            "sofa014_stageII": {'start': 110, 'stand_start': 230, 'sit_start': 550, 'sit_end': 710, 'stand_end': 770, 'end': 870},
            "sofa015_stageII": {'start': 140, 'stand_start': 680, 'sit_start': 760, 'sit_end': 890, 'stand_end': 1000, 'end': 1180},
            "sofa016_stageII": {'start': 130, 'stand_start': 630, 'sit_start': 700, 'sit_end': 810, 'stand_end': 870, 'end': 970},
            "sofa017_stageII": {'start': 140, 'stand_start': 440, 'sit_start': 520, 'sit_end': 640, 'stand_end': 685, 'end': 900},
            "sofa018_stageII": {'start': 200, 'stand_start': 370, 'sit_start': 480, 'sit_end': 605, 'stand_end': 740, 'end': 840},
            "sofa019_stageII": {'start': 110, 'stand_start': 580, 'sit_start': 700, 'sit_end': 950, 'stand_end': 1020, 'end': 1100},
            "sofa020_stageII": {'start': 110, 'stand_start': 170, 'sit_start': 270, 'sit_end': 330, 'stand_end': 400, 'end': 480},

            ### lie down
            "lie_down_stageII": {"start": 101, "stand_start": 193, "sit_start": 312, "sit_end": 479, "stand_end": 626, "end": 730},
            "lie_down_2_stageII": {"start": 200, "stand_start": 348, "sit_start": 480, "sit_end": 713, "stand_end": 820, "end": 900},
            "lie_down_3_stageII": {"start": 130, "stand_start": 290, "sit_start": 410, "sit_end": 636, "stand_end": 751, "end": 830},
            "lie_down_4_stageII": {"start": 116, "stand_start": 170, "sit_start": 310, "sit_end": 560, "stand_end": 670, "end": 810},
            "lie_down_7_stageII": {"start": 200, "stand_start": 290, "sit_start": 426, "sit_end": 700, "stand_end": 840, "end": 900},
            "lie_down_8_stageII": {"start": 130, "stand_start": 268, "sit_start": 400, "sit_end": 834, "stand_end": 940, "end": 1020},
            "lie_down_9_stageII": {"start": 160, "stand_start": 254, "sit_start": 400, "sit_end": 590, "stand_end": 680, "end": 761},
            "lie_down_10_stageII": {"start": 50, "stand_start": 132, "sit_start": 300, "sit_end": 546, "stand_end": 666, "end": 740},
            "lie_down_11_stageII": {"start": 130, "stand_start": 270, "sit_start": 440, "sit_end": 610, "stand_end": 765, "end": 846},
            "lie_down_12_stageII": {"start": 100, "stand_start": 87, "sit_start": 343, "sit_end": 592, "stand_end": 744, "end": 820},
            "lie_down_13_stageII": {"start": 164, "stand_start": 320, "sit_start": 456, "sit_end": 682, "stand_end": 787, "end": 839},
            "lie_down_14_stageII": {"start": 81, "stand_start": 213, "sit_start": 348, "sit_end": 524, "stand_end": 686, "end": 738},
            "lie_down_15_stageII": {"start": 96, "stand_start": 190, "sit_start": 320, "sit_end": 879, "stand_end": 970, "end": 1047},
            "lie_down_16_stageII": {"start": 72, "stand_start": 188, "sit_start": 317, "sit_end": 778, "stand_end": 901, "end": 1000},
            "lie_down_17_stageII": {"start": 118, "stand_start": 280, "sit_start": 425, "sit_end": 835, "stand_end": 947, "end": 1000},
            "lie_down_18_stageII": {"start": 115, "stand_start": 213, "sit_start": 401, "sit_end": 907, "stand_end": 1042, "end": 1124},
            "lie_down_19_stageII": {"start": 106, "stand_start": 162, "sit_start": 333, "sit_end": 660, "stand_end": 775, "end": 850},
            "lie_down_20_stageII": {"start": 140, "stand_start": 246, "sit_start": 398, "sit_end": 630, "stand_end": 766, "end": 857},
            "lie_down_21_stageII": {"start": 41, "stand_start": 121, "sit_start": 256, "sit_end": 476, "stand_end": 568, "end": 634},
            "lie_down_22_stageII": {"start": 80, "stand_start": 175, "sit_start": 327, "sit_end": 554, "stand_end": 690, "end": 774},
            "lie_down_23_stageII": {"start": 88, "stand_start": 178, "sit_start": 290, "sit_end": 529, "stand_end": 637, "end": 680},
            "lie_down_24_stageII": {"start": 130, "stand_start": 272, "sit_start": 412, "sit_end": 608, "stand_end": 729, "end": 817},
            "lie_down_25_stageII": {"start": 150, "stand_start": 259, "sit_start": 420, "sit_end": 628, "stand_end": 761, "end": 838},
            "lie_down_26_stageII": {"start": 42, "stand_start": 132, "sit_start": 266, "sit_end": 569, "stand_end": 689, "end": 764},
            "lie_down_27_stageII": {"start": 94, "stand_start": 138, "sit_start": 278, "sit_end": 464, "stand_end": 611, "end": 689},
            "lie_down_28_stageII": {"start": 141, "stand_start": 223, "sit_start": 342, "sit_end": 699, "stand_end": 789, "end": 860},
            "lie_down_29_stageII": {"start": 130, "stand_start": 241, "sit_start": 339, "sit_end": 573, "stand_end": 706, "end": 774},
            "lie_down_30_stageII": {"start": 57, "stand_start": 162, "sit_start": 311, "sit_end": 562, "stand_end": 726, "end": 796},
            "lie_down_31_stageII": {"start": 48, "stand_start": 157, "sit_start": 285, "sit_end": 415, "stand_end": 555, "end": 651},
            "lie_down_32_stageII": {"start": 141, "stand_start": 261, "sit_start": 414, "sit_end": 639, "stand_end": 731, "end": 801},
            "lie_down_33_stageII": {"start": 67, "stand_start": 173, "sit_start": 300, "sit_end": 487, "stand_end": 620, "end": 685},
            "lie_down_34_stageII": {"start": 135, "stand_start": 266, "sit_start": 368, "sit_end": 646, "stand_end": 738, "end": 809},
        }

    motion_cfg.update(motion_cfg1)

    ### load previous labeled data:
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'SAMP_interaction', 'body_meshes', 'avaliable_frame.txt'))
    with open(input_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
    all_lines = [line.strip() for line in lines]
    # motion_cfg = {}

    # ! use the old start and end frames.
    for line in all_lines:
        if line.split(',')[0] not in motion_cfg.keys() or 'chair_mo00' in line.split(',')[0] or 'chair_mo_stageII' in line.split(',')[0]:
            # print(line.split(',')[0])
            continue
        # old ones are the 20 fps frames;
        motion_cfg[line.split(',')[0]]['start'] = int(int(line.split(',')[1]) * 30 / 20.0)
        motion_cfg[line.split(',')[0]]['end'] = min(int(int(line.split(',')[2])* 30 / 20.0), motion_cfg[line.split(',')[0]]['end'])

    # ------------------ load the new labelled data ------------------
    # import pdb;pdb.set_trace()
    cfg_fps = 30
    target_fps = 20
    for key, value in motion_cfg.items():
        start_frame = value['start']
        for k, v in value.items():
            value[k] = int((v - start_frame) * target_fps / cfg_fps)
    

    return motion_cfg

def get_contact_label_from_dict(input_joints, interaction_still_idx, max_distance=1.5): 
    dists_list = []

    obj_center = input_joints[interaction_still_idx][0]
    frame_number = input_joints.shape[0]

    for fId in range(0, frame_number):
        joints = input_joints[fId][0]
        # calculate the eculidean distance between joints and object center.
        dist = np.linalg.norm((joints-obj_center)[:2], axis=0) # .mean() # xy is the ground, z is the height.
        dists_list.append(dist)
    
    available_list = (np.array(dists_list) < max_distance) * np.arange(len(dists_list)) 
    return available_list

def get_interaction_text_description(kind='sit'):

    if kind in ['sit', 'lie']:
        text_dict = {
            'walk_to_sit': 'a person walks, and sits down on an object.#a/DET person/NOUN walk/VERB, sit/VERB down/ADP on/ADP an/DET object/NOUN#0.0#0.0',
            'sit_from_stand': 'a person stands and then sits on a chair behind them. #a/DET person/NOUN stand/VERB and/CCONJ then/ADV sit/VERB on/ADP a/DET chair/NOUN behind/ADP them/PRON#0.0#0.0',
            'sit_to_stand': 'a person is sitting on a chair and then stands up.#a/DET person/NOUN be/VERB sit/VERB on/ADP a/DET chair/NOUN and/CCONJ then/ADV stand/VERB up/ADV#0.0#0.0',
            'stand_sit_stand': 'a person sits on a chair behind them and then stands back up.#a/DET person/NOUN sit/VERB on/ADP a/DET chair/NOUN behind/ADP them/PRON and/CCONJ then/ADV stand/VERB back/ADV up/ADV#0.0#0.0',
            'walk_to_sit_to_stand': 'a person walks, and sits down on an object, and then stands back up.#a/DET person/NOUN walk/VERB, sit/VERB down/ADP on/ADP an/DET object/NOUN, and/CCONJ then/ADV stand/VERB back/ADV up/ADV#0.0#0.0',
        }
        all_sit_keys = list(text_dict.keys())
        for key in all_sit_keys:
            text_dict[key.replace('sit', 'lie')] = text_dict[key].replace('sit', 'lie').replace('chair', 'sofa')
    else:
        raise NotImplementedError

    return text_dict

def get_data_augmentaion_on_motion(sit_label_list, inside_object_list, available_list, text_dict, aug_num=100,
                                kind='sit', min_motion_len=40, max_motion_len=196, action_type=-1, 
                                use_comprehensive_text=False, stand_to_sit=None, walk_to_sit=None):
    if kind == 'sit': # TODO: needs to be relabelled. 

        num_kind = len(text_dict)
        video_length = len(inside_object_list)
        new_start_f_list = []
        new_end_f_list = []
        new_text_list = []

        # only consider a single sit motion in the video.
        # import pdb;pdb.set_trace()
        sit_index = np.nonzero(sit_label_list)
        inside_index = np.nonzero(inside_object_list)

        sit_start = sit_index[0][0]
        sit_end = sit_index[0][-1]
        # transition_start = inside_index[0][0]
        # transition_end = inside_index[0][-1]

        sit_start = sit_start + 10
        transition_start = sit_start - 20 # 1 s transition time.

        sit_end = sit_end - 10
        # sit_end + 20 (almost leave the chair) | 10 with standing.
        transition_end = sit_end + 10 # make sure the transition frame leave the chair any space; [make the frame is within stand - walk]

        print('transition_start', transition_start, 'sit_start', sit_start, 'sit_end', sit_end, 'transition_end', transition_end)
        # print(sit_start, transition_start, transition_end, sit_end)
        kind_list = range(num_kind)

        if action_type == 0:
            kind_list = [0]
        elif action_type == 1:
            kind_list = [0,1]
        elif action_type == 2:
            kind_list = [2]
        
        
        # kind_list = [1,2,3,4]
        print('training data: -----------------------')
        print('action type: ', action_type)
        print('kind list: ', kind_list)
        
        min_start = 0
        for tmp_i in range(sit_start, 0, -1):
            if available_list[tmp_i] == 0:
                min_start = tmp_i+1
                break
        
        max_end = video_length-1
        for tmp_i in range(sit_end, video_length):
            if available_list[tmp_i] == 0:
                max_end = tmp_i-1
                break
        print(min_start, max_end)

        all_use_list = []
        for i in kind_list:
            for j in range(aug_num):
                # try: # diversity get large.
                if True: # bug-free from 1216
                    # print(i, j)
                    # print(sit_start, sit_end, video_length)
                    # print(available_list)
                    if i == 0: 
                        # import pdb;pdb.set_trace()
                        # min_start = available_list.nonzero()[0].min()
                        # this would definitely a person is sitting.
                        # new_end_f = np.random.randint(sit_start, min(min(sit_end, transition_start+max_motion_len), max_end), 1) # leave the space for transition
                        # new_start_f = np.random.randint(max(new_end_f-max_motion_len, min_start), min(transition_start, new_end_f-min_motion_len), 1)
                        
                        valid_samples = np.arange(sit_start, min(min(sit_end, transition_start+max_motion_len), max_end), 8)
                        new_end_f = np.random.choice(valid_samples, 1)

                        valid_samples = np.arange(max(new_end_f-max_motion_len, min_start), min(transition_start, new_end_f-min_motion_len), 8)
                        new_start_f = np.random.choice(valid_samples, 1)

                        if [new_start_f, new_end_f] in all_use_list:
                            continue
                        else:
                            all_use_list.append([new_start_f, new_end_f])
                        
                        new_start_f_list.append(new_start_f)
                        new_end_f_list.append(new_end_f)
                        if use_comprehensive_text:
                            new_text_list = new_text_list + walk_to_sit
                        else:
                            new_text_list = new_text_list + [text_dict['walk_to_sit']] 

                    elif i == 1:
                        # new_start_f = np.random.randint(transition_start, sit_start, 1)
                        # new_element = max(min_motion_len, sit_start-new_start_f)
                        # length = np.random.randint(new_element, min(max_motion_len, sit_end-new_start_f), 1)
                        # new_end_f = new_start_f + length

                        valid_sample = np.arange(transition_start, sit_start, 8)
                        new_start_f = np.random.choice(valid_sample, 1)

                        new_element = max(min_motion_len, sit_start-new_start_f)
                        length_sample = np.arange(new_element, min(max_motion_len, sit_end-new_start_f), 8)
                        length = np.random.choice(length_sample, 1)
                        
                        new_end_f = new_start_f + length

                        if [new_start_f, new_end_f] in all_use_list:
                            continue
                        else:
                            all_use_list.append([new_start_f, new_end_f])

                        new_start_f_list.append(new_start_f)
                        new_end_f_list.append(new_end_f)
                        if use_comprehensive_text:
                            new_text_list = new_text_list + stand_to_sit
                        else:
                            new_text_list = new_text_list + [text_dict['sit_from_stand']]

                    elif i == 2:
                        # sit end is the person leaves the chair
                        # it's already the standing 
                        # new_end_f = np.random.randint(transition_end, min(max_end, min(video_length, transition_end+max_motion_len)), 1)
                        # new_start_f = np.random.randint(max(min(new_end_f-max_motion_len, sit_end-20), sit_start+20), min(new_end_f-min_motion_len, sit_end-10), 1) # transition time = 60 frames
                        
                        valid_sample = np.arange(transition_end, min(max_end, min(video_length, transition_end+max_motion_len)), 8)
                        new_end_f = np.random.choice(valid_sample, 1)

                        valid_sample = np.arange(max(min(new_end_f-max_motion_len, sit_end-20), sit_start+20), min(new_end_f-min_motion_len, sit_end-10), 8)
                        new_start_f = np.random.choice(valid_sample, 1)
                        
                        if [new_start_f, new_end_f] in all_use_list:
                            continue
                        else:
                            all_use_list.append([new_start_f, new_end_f])
                            
                        new_start_f_list.append(new_start_f)
                        new_end_f_list.append(new_end_f)
                        print(new_start_f, new_end_f)
                        new_text_list = new_text_list + [text_dict['sit_to_stand']] # sit to stand/walk


                    elif i == 3: # transition = -30 frames;
                        # if transition_start >= sit_start:
                        #     continue

                        if sit_end-sit_start >  max_motion_len:
                            # print(f'sit too long, sit_start{sit_start}-sit_end{sit_end} over {max_motion_len}')
                            continue
                        elif sit_end-sit_start ==  max_motion_len:
                            delta = 0
                            new_start_f = sit_start
                            new_end_f = sit_end
                            if j == 0:
                                new_start_f_list.append(new_start_f)
                                new_end_f_list.append(new_end_f)
                                new_text_list = new_text_list + [text_dict['stand_sit_stand']] 
                            continue

                        valid_sample = np.arange(0, max_motion_len - (sit_end-sit_start), 8)
                        delta = np.random.choice(valid_sample, 1)

                        new_start_f = sit_start - delta
                        
                        valid_sample = np.arange(0, max_motion_len - (sit_end-sit_start) - delta, 8)
                        delta2 = np.random.choice(valid_sample, 1)
                        # delta2 = np.random.randint(0, max_motion_len - (sit_end-sit_start) - delta, 1)
                        
                        new_end_f = sit_end + delta2
                        
                        if [new_start_f, new_end_f] in all_use_list:
                            continue
                        else:
                            all_use_list.append([new_start_f, new_end_f])
                            
                        new_start_f_list.append(new_start_f)
                        new_end_f_list.append(new_end_f)
                        new_text_list = new_text_list + [text_dict['stand_sit_stand']] 
                        length = new_end_f - new_start_f
                        
                    elif i == 4:
                        if sit_end-transition_start > max_motion_len: #or sit_end-max_motion_len < 0:
                            # print(f'sit too long, sit_start{transition_start}-sit_end{sit_end} over {max_motion_len}')
                            continue
                        elif sit_end-transition_start ==  max_motion_len:
                            delta = 0
                            new_start_f = transition_start
                            new_end_f = sit_end
                            if j == 0:
                                new_start_f_list.append(new_start_f)
                                new_end_f_list.append(new_end_f)
                                new_text_list = new_text_list + [text_dict['stand_sit_stand']] 
                            continue


                        shift = max_motion_len - (sit_end-transition_start)
                        # tmp = np.random.randint(0, shift, 1)
                        #  
                        valid_sample = np.arange(0, shift, 8)
                        tmp = np.random.choice(valid_sample, 1)
                        new_start_f = transition_start - tmp

                        # left = np.random.randint(0, shift-tmp, 1)
                        valid_sample = np.arange(0, shift-tmp, 8)
                        left = np.random.choice(valid_sample, 1)

                        new_end_f = sit_end+left
                        
                        if [new_start_f, new_end_f] in all_use_list:
                            continue
                        else:
                            all_use_list.append([new_start_f, new_end_f])
                            
                        new_start_f_list.append(new_start_f)
                        new_end_f_list.append(new_end_f)
                        new_text_list = new_text_list + [text_dict['walk_to_sit_to_stand']] # walk to sit to stand/walk
                
        
        return [one.item() for one in new_start_f_list], [one.item() for one in new_end_f_list], new_text_list
                
    else:
        raise NotImplementedError
# '''


def get_data_augmentaion_on_motion_mannual_label(action_status_dict, available_list, text_dict, aug_num=100,
                                kind='sit', min_motion_len=40, max_motion_len=196, action_type=-1,
                                use_comprehensive_text=False, stand_to_sit=None, walk_to_sit=None,
                                walk_to_stand=None, stand_to_stand=None):
    new_start_f_list = []
    new_end_f_list = []
    new_text_list = []
    
    transition_start = action_status_dict['stand_start']
    sit_start = action_status_dict[f'sit_start'] # all use this key.
    sit_end = action_status_dict[f'sit_end']
    transition_end = action_status_dict['stand_end']

    video_length = min(action_status_dict['end'], len(available_list))
    
    # import pdb;pdb.set_trace()
    print('available_list: ', len(available_list))
    print('mannul labled: transition_start', transition_start, f'{kind}_start', sit_start, f'{kind}_end', sit_end, 'transition_end', transition_end, 'video_length', video_length)
    # print(sit_start, transition_start, transition_end, sit_end)

    num_kind = len(text_dict)
    kind_list = range(num_kind)

    if action_type == 0:
        kind_list = [0]
    elif action_type == 1:
        kind_list = [0,1]
    elif action_type == 2: # only sit down.
        kind_list = [2]
    elif action_type == 3: # only one action is possible.
        kind_list = [0,1,2]
    elif action_type == 4: # only sit down with a very near distance.
        kind_list = [1]
    elif action_type == 5:
        kind_list = [3, 4]
    
    # kind_list = [1,2,3,4]
    print('training data: -----------------------')
    print('action type: ', action_type)
    print('kind list: ', kind_list)
    
    min_start = 0
    for tmp_i in range(sit_start, 0, -1):
        if available_list[tmp_i] == 0:
            min_start = tmp_i+1
            break
    
    max_end = video_length-1
    for tmp_i in range(sit_end, video_length):
        if available_list[tmp_i] == 0:
            max_end = tmp_i-1
            break
    print(min_start, max_end, 'in [{}, {}]'.format(0, max_end))
    print('video length: ', video_length)

    all_use_list = []
    for i in kind_list:
        for j in range(aug_num):
            if True: 
                if i == 0: 
                    # import pdb;pdb.set_trace()
                    # min_start = available_list.nonzero()[0].min()
                    # this would definitely a person is sitting.
                    # new_end_f = np.random.randint(sit_start, min(min(sit_end, transition_start+max_motion_len), max_end), 1) # leave the space for transition
                    valid_samples = np.arange(sit_start, min(min(sit_end, transition_start+max_motion_len), max_end), 8)
                    new_end_f = np.random.choice(valid_samples, 1)

                    # print(max(new_end_f-max_motion_len, min_start), new_end_f, min_start, sit_start, min(transition_start, new_end_f-min_motion_len), transition_start)

                    valid_samples = np.arange(max(new_end_f-max_motion_len, min_start), min(transition_start, new_end_f-min_motion_len), 8)

                    if valid_samples.shape[0] == 0: # ! this should not happen.
                        print(1111)
                        continue

                    new_start_f = np.random.choice(valid_samples, 1)

                    if [new_start_f, new_end_f] in all_use_list:
                        continue
                    else:
                        all_use_list.append([new_start_f, new_end_f])
                    new_start_f_list.append(new_start_f)
                    new_end_f_list.append(new_end_f)
                    # new_text_list = new_text_list + [text_dict[f'walk_to_{kind}']]
                    if use_comprehensive_text:
                        new_text_list = new_text_list + walk_to_sit
                    else:
                        new_text_list = new_text_list + [text_dict[f'walk_to_{kind}']]  

                elif i == 1: # start from the very near frames. 2 seconds-0.5 seconds.
                    # ! in pratical use: people start from the standing pose.

                    # new_start_f = np.random.randint(transition_start, sit_start, 1)
                    # new_element = max(min_motion_len, sit_start-new_start_f)
                    # length = np.random.randint(new_element, min(max_motion_len, sit_end-new_start_f), 1)

                    # valid_sample = np.arange(transition_start, sit_start, 8) # sit_start is too late for label.
                    valid_sample = np.arange(max(0, transition_start-10), transition_start+20, 4)
                    new_start_f = np.random.choice(valid_sample, 1)

                    new_element = max(min_motion_len, sit_start-new_start_f)
                    length_sample = np.arange(new_element, min(max_motion_len, sit_end-new_start_f), 8)
                    length = np.random.choice(length_sample, 1)
                    
                    new_end_f = new_start_f + length

                    if [new_start_f, new_end_f] in all_use_list:
                        continue
                    else:
                        all_use_list.append([new_start_f, new_end_f])

                    new_start_f_list.append(new_start_f)
                    new_end_f_list.append(new_end_f)
                    # new_text_list = new_text_list + [text_dict[f'{kind}_from_stand']]
                    if use_comprehensive_text:
                        new_text_list = new_text_list + stand_to_sit
                    else:
                        new_text_list = new_text_list + [text_dict[f'{kind}_from_stand']]

                elif i == 2:
                    # sit end is the person leaves the chair
                    # it's already the standing 
                    # ! the data is not complete | chair_mo003, chair007
                    if transition_end > min(max_end, min(video_length, transition_end+max_motion_len)):
                        print('wrong', transition_end, min(max_end, min(video_length, transition_end+max_motion_len)))
                        continue

                    # new_end_f = np.random.randint(transition_end, min(max_end, min(video_length, transition_end+max_motion_len)), 1)
                    # new_start_f = np.random.randint(max(min(new_end_f-max_motion_len, sit_end), sit_start), min(new_end_f-min_motion_len, sit_end), 1) # transition time = 60 frames
                    
                    valid_sample = np.arange(transition_end, min(max_end, min(video_length, transition_end+max_motion_len)), 8)
                    new_end_f = np.random.choice(valid_sample, 1)

                    valid_sample = np.arange(max(min(new_end_f-max_motion_len, sit_end), sit_start), min(new_end_f-min_motion_len, sit_end), 8)
                    new_start_f = np.random.choice(valid_sample, 1)
                    
                    if [new_start_f, new_end_f] in all_use_list:
                        continue
                    else:
                        all_use_list.append([new_start_f, new_end_f])

                    new_start_f_list.append(new_start_f)
                    new_end_f_list.append(new_end_f)
                    new_text_list = new_text_list + [text_dict[f'{kind}_to_stand']] # sit to stand/walk
                
                ### ! need to be checked whether this is available.
                elif i == 3: # transition = -30 frames;

                    if sit_end+20-transition_start+10 >  max_motion_len:
                        continue
                    elif sit_end+20-transition_start+10 ==  max_motion_len:
                        delta = 0
                        new_start_f = transition_start-10+1
                        new_end_f = sit_end+20
                        if j == 0:
                            new_start_f_list.append(new_start_f)
                            new_end_f_list.append(new_end_f)
                            
                            if use_comprehensive_text:
                                new_text_list = new_text_list + walk_to_stand
                            else:
                                new_text_list = new_text_list + [text_dict[f'walk_to_{kind}_to_stand']] 
                        continue

                    # delta = np.random.randint(0, max_motion_len - (sit_end-sit_start), 1)
                    valid_sample = np.arange(0, max_motion_len - (sit_end+20-transition_start+10), 4)
                    delta = np.random.choice(valid_sample, 1)

                    new_start_f = sit_start - delta
                    
                    valid_sample = np.arange(0, max_motion_len - (sit_end+20-transition_start+10) - delta, 4)
                    delta2 = np.random.choice(valid_sample, 1)
                    # delta2 = np.random.randint(0, max_motion_len - (sit_end-sit_start) - delta, 1)
                    
                    new_end_f = sit_end+20 + delta2
                    
                    if [new_start_f, new_end_f] in all_use_list:
                        continue
                    else:
                        all_use_list.append([new_start_f, new_end_f])
                    
                    new_start_f_list.append(new_start_f)
                    new_end_f_list.append(new_end_f)
                    if use_comprehensive_text:
                        new_text_list = new_text_list + walk_to_stand
                    else:
                        new_text_list = new_text_list + [text_dict[f'walk_to_{kind}_to_stand']] 

                    length = new_end_f - new_start_f + 1
                    # print(i, length)

                elif i == 4:
                    if sit_end+20-transition_start > max_motion_len: #or sit_end-max_motion_len < 0:
                        # print(f'sit too long, sit_start{transition_start}-sit_end{sit_end} over {max_motion_len}')
                        continue
                    elif sit_end+20-transition_start ==  max_motion_len:
                        delta = 0
                        new_start_f = transition_start+1
                        new_end_f = sit_end+20
                        if j == 0:
                            new_start_f_list.append(new_start_f)
                            new_end_f_list.append(new_end_f)
                            if use_comprehensive_text:
                                new_text_list = new_text_list + stand_to_stand
                            else:
                                new_text_list = new_text_list + [text_dict[f'stand_{kind}_stand']] # walk to sit to stand/walk
                        continue

                    shift = max_motion_len - (sit_end+20-transition_start)
                    # tmp = np.random.randint(0, shift, 1)
                    #  
                    valid_sample = np.arange(0, shift, 4)
                    tmp = np.random.choice(valid_sample, 1)
                    new_start_f = transition_start + 10 - tmp

                    # left = np.random.randint(0, shift-tmp, 1)
                    valid_sample = np.arange(0, shift-tmp, 4)
                    left = np.random.choice(valid_sample, 1)
                    new_end_f = sit_end+20+left
                    
                    length = new_end_f - new_start_f + 1
                    if [new_start_f, new_end_f] in all_use_list:
                        continue
                    else:
                        all_use_list.append([new_start_f, new_end_f])

                    new_start_f_list.append(new_start_f)
                    new_end_f_list.append(new_end_f)
                    if use_comprehensive_text:
                        new_text_list = new_text_list + stand_to_stand
                    else:
                        new_text_list = new_text_list + [text_dict[f'stand_{kind}_stand']] # walk to sit to stand/walk

    return [one.item() if isinstance(one, np.ndarray) and one.size == 1 else one for one in new_start_f_list], \
        [one.item() if isinstance(one, np.ndarray) and one.size == 1 else one for one in new_end_f_list], new_text_list       
    


def add_entire_motion(data_dict, length_list, name_list, \
    name, n_motion, scene_data_process, scene_transform_mat, text_data, opt, restore_dir=None, restore=False):

    # ! this is used for add our mannually written text description.
    ## scene_data_process: sdf list.
    if hasattr(opt, 'indicator') and opt.indicator: 
        indicator = np.zeros((n_motion.shape[0], 1))
        indicator[[0, -1], :] = 1.0
        n_motion = np.concatenate((n_motion, indicator), -1)
    
    if not isinstance(text_data, list):    
        def text_to_data(line):
            text_dict = {}
            line_split = line.strip().split('#')
            caption = line_split[0]
            tokens = line_split[1].split(' ')
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag

            text_dict['caption'] = caption
            text_dict['tokens'] = tokens
            return text_dict

        text_dict = [text_to_data(text_data)]
    else:
        text_dict = text_data

    data_dict[name] = {'motion': n_motion,
                        'length': len(n_motion),
                        'text': text_dict,
                        'scene': scene_data_process, 
                        'tranform_mat': scene_transform_mat}
    length_list.append(len(n_motion))
    name_list.append(name)

    if restore:
        # for each motion sequence, we save each object pairs.
        sub_dir = os.path.basename(name).split('_')[0]
        os.makedirs(os.path.join(restore_dir, sub_dir), exist_ok=True)
        print('save to ', os.path.join(restore_dir, sub_dir, os.path.basename(name)+'.pkl'))
        with open(os.path.join(restore_dir, sub_dir, os.path.basename(name)+'.pkl'), 'wb') as f:
            pickle.dump(data_dict[name], f)

    return data_dict, length_list, name_list


def add_subset_motion(data_dict, length_list, name_list, name, motion, scene_data_process, scene_transform_mat,
                      text_data, opt, restore_dir=None, restore=False, max_motion_length=196):
    
    bias = random.randint(0, len(motion) - max_motion_length)
    
    motion_sub = motion[bias: bias + max_motion_length]
    motion_sub[:, [0,1]] = motion_sub[:, [0,1]] - motion_sub[0, [0,1]]

    # change the motion trajectory.
    cos_theta = motion_sub[:, 2]
    sin_theta = motion_sub[:, 3]
    rot_mat = np.zeros((len(cos_theta), 2, 2))
    rot_mat[:, 0, 0] = cos_theta
    rot_mat[:, 0, 1] = sin_theta
    rot_mat[:, 1, 0] = -sin_theta
    rot_mat[:, 1, 1] = cos_theta

    motion_before = motion_sub[:, [0,1]].copy()
    motion_sub[:, [0,1]] = np.matmul(np.linalg.inv(rot_mat[0, ]), motion_before.T).T

    rot_mat[:] = np.matmul(np.linalg.inv(rot_mat[0,]), rot_mat[:])
    motion_sub[:, 2] = rot_mat[:, 0, 0]
    motion_sub[:, 3] = rot_mat[:, 0, 1]

    # TODO: visualize the image and trajectory.

    # vis_traj_on_scene(motion_sub, scene_data_process, 'debug_results/transform.png', yaxis_up=True)                                    
    data_dict[name] = {'motion': motion_sub,
                        'length': max_motion_length,
                        'text': text_data,
                        'scene': scene_data_process,
                        'tranform_mat': scene_transform_mat,
                        }
    length_list.append(len(motion_sub))
    name_list.append(name)

    return data_dict, length_list, name_list

# this only used for SAMP: the motion and the object is in the xy plane.
def canonicalize_poses_to_object_space(input_joints, transform_mat, obj_transl_xz, obj_rot_mat): 

    # poses_joints: B, J, 3; torch.Tensor;
    # output: torch.Tensor
    # import pdb;pdb.set_trace()

    B, J, _ = input_joints.shape
    poses_joints = input_joints.reshape(-1, 3) # 3, J, B
    # TODO: add transl and rotation sampling, and coordinate system synchronization.

    if isinstance(transform_mat, np.ndarray):
        transform_mat = torch.from_numpy(transform_mat).float().to(input_joints.device)
        
    # canocalize body -> world body -> canonize object space.      
    # import pdb;pdb.set_trace()                         
    rot_mat, transl = transform_mat[:3, :3], transform_mat[:, 3]
    world_poses = (torch.inverse(rot_mat) @ poses_joints.T).T + transl # 
    
    # this in xz plane coordinate system -> xy plane
    world_poses = (torch.inverse(torch.from_numpy(trans_matrix).float().to(input_joints.device)) @ world_poses.T).T

    if isinstance(obj_rot_mat, np.ndarray):
        obj_rot_mat = torch.from_numpy(obj_rot_mat).float().to(input_joints.device)
        obj_transl_xz = torch.from_numpy(obj_transl_xz).float().to(input_joints.device)

    new_world_poses = (torch.inverse(obj_rot_mat) @ (world_poses - obj_transl_xz[None, :]).T).T

    return new_world_poses.reshape(B, J, 3)


def canonicalize_poses_to_object_space_th(input_joints, transform_mat, obj_transl_xz, obj_rot_mat, original_scene_ground_plane=GROUND_PLAN):

    # poses_joints: B, N, J, 3; torch.Tensor;
    # output: torch.Tensor

    B, N, J, _ = input_joints.shape
    poses_joints = input_joints.reshape(B, -1, 3) # 3, J, B
    # TODO: add transl and rotation sampling, and coordinate system synchronization.

    # canocalize body -> world body -> canonize object space.      
    # import pdb;pdb.set_trace()                         
    rot_mat, transl = transform_mat[:, :3, :3].float(), transform_mat[:, :, 3].float()
    # world_poses = (torch.inverse(rot_mat) @ poses_joints.T).T + transl # 
    world_poses = torch.bmm(torch.inverse(rot_mat), poses_joints.permute(0, 2, 1)).permute(0, 2, 1) + transl[:, None].repeat((1, N*J, 1))
    
    # import pdb;pdb.set_trace()
    # save_sample_poses(world_poses.detach().cpu().numpy().reshape(N,J,-1), 'debug_results/collision_guidance_object', 'world_cs_xzplane')
    
    if original_scene_ground_plane == 'xy': # ! used for SAMP, default !!!
        # this in xz plane coordinate system -> xy plane
        # import pdb;pdb.set_trace()
        b_trans_mat = torch.inverse(torch.from_numpy(trans_matrix).float().to(input_joints.device)).repeat((B, 1, 1))
        world_poses = torch.bmm(b_trans_mat, world_poses.permute(0,2,1)).permute(0, 2, 1)
        # save_sample_poses(world_poses.detach().cpu().numpy().reshape(N,J,-1), 'debug_results/collision_guidance_object', 'world_cs_xyplane')

    # TODO: normalize scale from body poses; 

    # new_world_poses = (torch.inverse(obj_rot_mat) @ (world_poses - obj_transl_xz[None, :]).T).T
    new_world_poses = torch.bmm(torch.inverse(obj_rot_mat[:, :3, :3]), (world_poses - obj_transl_xz[:,None].repeat((1, N*J, 1))).permute(0, 2, 1)).permute(0,2,1)

    return new_world_poses.reshape(B, N, J, 3)


def visualize_interactive_scene(scene_sdf_volume, sdf_dict, input_poses, save_dir, postfix=''):
    pass
    # visualize sdf volume
    # import pdb;pdb.set_trace()
    # vertices, faces, normals, _ = skimage.measure.marching_cubes(scene_sdf_volume, level=0)
    # vertices = vertices / sdf_dict['dim'] * 2 - 1
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    # sphere = trimesh.creation.uv_sphere(radius=0.02)
    # poses = np.tile(np.eye(4), (2, 1, 1))
    # poses[0, :3, 3] = np.array([1, 1, 1])
    # poses[1, :3, 3] = -np.array([1, 1, 1])
    # sphere.apply_transform(poses[0])

    # mesh.export(os.path.join(save_dir, 'scene.obj'))
    # sphere.export(os.path.join(save_dir, 'sphere.obj'))

    # visualize a voxel. 
    scale = sdf_dict['scale']
    center = sdf_dict['centroid']
    dim = sdf_dict['dim']

    grid_min = scale * np.ones(3) * -1

    # numpy process.
    def get_vertices_from_sdf_volume(sdf_volume, voxel_size, grid_min):
        contact_index = (sdf_volume <= 0).nonzero()
        contact_v_x = (contact_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
        contact_v_y = (contact_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
        contact_v_z = (contact_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
        contact_v = np.stack([contact_v_x, contact_v_y, contact_v_z], -1)
        
        out_index = (sdf_volume > 0).nonzero()
        out_v_x = (out_index[0] * 1.0 + 0.5) * voxel_size[0] + grid_min[0]
        out_v_y = (out_index[1] * 1.0 + 0.5)* voxel_size[1] + grid_min[1]
        out_v_z = (out_index[2] * 1.0 + 0.5)* voxel_size[2] + grid_min[2]
        out_v = np.stack([out_v_x, out_v_y, out_v_z], -1)

        
        return contact_v, out_v

    # import pdb;pdb.set_trace()
    voxel_size = np.ones(3) * 2 * scale / dim 

    contact_v, out_v = get_vertices_from_sdf_volume(scene_sdf_volume, voxel_size, grid_min)
    contact_v += center 
    out_v += center

    out_mesh = trimesh.Trimesh(contact_v, process=False)
    template_save_fn = os.path.join(save_dir, f'in_sdf_sample_{postfix}.ply') 
    out_mesh.export(template_save_fn, vertex_normal=False) # export_ply

    sample_num = int(out_v.shape[0] / 1000)
    sample_out_v = out_v[np.random.choice(out_v.shape[0], sample_num, replace=False)]
    out_mesh = trimesh.Trimesh(sample_out_v, process=False)
    template_save_fn = os.path.join(save_dir, f'out_sdf_sample_0.001_{postfix}.ply') 
    out_mesh.export(template_save_fn, vertex_normal=False) # export_ply
        
    if input_poses is not None:
        # export poses
        for i in range(input_poses.shape[0]):
            if i % 20 == 0:
                poses_mesh = trimesh.Trimesh(vertices=input_poses[i], process=False)
                poses_mesh.export(os.path.join(save_dir, f'poses_{i}_{postfix}.obj'))

def export_body_meshes(input_poses, body_face, save_dir):
    print('save to ', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(input_poses.shape[0]):
        poses_mesh = trimesh.Trimesh(vertices=input_poses[i], faces=body_face, process=False)
        poses_mesh.export(os.path.join(save_dir, f'frame_{i:03d}.obj'))

def save_sample_poses(input_poses, save_dir, post_fix):
    for i in range(input_poses.shape[0]):
        if i % 20 == 0 or i == input_poses.shape[0]-1:
            poses_mesh = trimesh.Trimesh(vertices=input_poses[i], process=False)
            poses_mesh.export(os.path.join(save_dir, f'poses_{i}_{post_fix}.obj'))

def canonicalize_motion_and_scene_new(pose_seq_np, scene_data_input, return_transform=True):

    # pose_seq_np: (68, 55, 3)
    # scene_data_input: (1, point number, 3)
    # ! body and scene are in xz plane, y is the height

    # in xz plane coord /
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)#(68, 55, 3) # face orientation is wrong.

    # ! this is the key problem.
    pose_seq_np_n[..., 2] = pose_seq_np_n[..., 2]

    source_data = pose_seq_np_n[:, :joints_num]#(68, 22, 3)
    motion, ground_positions, positions, l_velocity = process_file(source_data, 0.002)

    # 获取AMASS到HumanML3D的偏移和旋转
    positions=source_data
    positions, scale_rt = uniform_skeleton(positions, tgt_offsets)
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1]) # translation.
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    
    root_quat_init = qbetween_np(forward_init, target) # rotation;
    root_init_matrix = quaternion_to_matrix_np(root_quat_init)  # rotation matrix.
    root_init_euler = qeuler(torch.tensor(root_quat_init), 'xzy', epsilon=0, deg=True)

    # transform the obj or the sdf and grad volume, which represented as a set of point cloud.

    # TODO: add scene transformation.
    # new_scene_data_process = root_init_matrix @ (scene_data_input + root_pose_init_xz[:, np.newaxis, np.newaxis])

    # import pdb;pdb.set_trace()
    if return_transform:
        transform_mat = np.concatenate((root_init_matrix[0], root_pose_init_xz[:, np.newaxis]), axis=1)
        return motion, None, transform_mat
    else:
        return motion, None
