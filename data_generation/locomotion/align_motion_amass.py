import os
import sys
import numpy as np
from pathlib import Path
import json
from PIL import Image
import trimesh
from tqdm import tqdm
import pandas as pd
import random
import pickle as pkl
import time
from p_tqdm import p_map, p_uimap
import glob

from babel_tools import load_motion
from utils.smplx_util import SMPLX_Util
from sklearn.neighbors import KDTree
from utils.geo_utils import is_point_in_cuboid, make_M_from_tqs, create_unit_bbox, create_vector, rotate_2D_points_along_z_axis
import utils.configuration as config
 
##############################
## Constants
#############################
FLOOR_HEIGHT = 0. # the height of floor plane
FLOOR_PLANE_GRID_SIZE = 256 # size of floor plane grids (square)
FLOOR_PLANE_GRID_SCALE = 6.2*2/FLOOR_PLANE_GRID_SIZE # scale of floor plane grid (metre)
CHECK_INTERVAL = 5
NUM_SAMPLE_PROPOSAL = 1
MAX_TRY = 100
NUM_SCENES = 10

########################################
## align class
########################################
class ActionAlign():
    def __init__(
        self, 
        # annotations: dict,
        # instance_to_semantic: dict,
        # label_mapping: dict,
        # scene_path: str, 
        # static_scene: trimesh.PointCloud, 
        # static_scene_label: np.ndarray,
        # static_scene_trans: np.ndarray,
        floor_plane_mask: np.array,
        body_vertices: np.ndarray, 
        joints_traj: np.ndarray,
        scale: float=FLOOR_PLANE_GRID_SCALE,
    ):
        """ Action Align father class
        """
        self.action = None
        self.floor_plane_mask = floor_plane_mask
        self.scale = scale
        # self.annotations = annotations
        # self.instance_to_semantic = instance_to_semantic
        # self.label_mapping = label_mapping

        # self.scene_id = scene_path.split('/')[-2]
        # self.static_scene = static_scene
        # self.static_scene_label = static_scene_label
        # self.static_scene_trans = static_scene_trans
        self.body_vertices = body_vertices
        self.joints_traj = joints_traj

        # self.scene_occupancy = self.get_scene_occupancy(static_scene_label)
        # self.floor_occupancy = self.get_scene_floor_occupancy(static_scene_label)
        # self.wall_occupancy = self.get_scene_wall_occupancy(static_scene_label)

        lf_traj, rf_traj = self.get_smpl_foots_keypoints_trajectory(body_vertices)
        self.left_foot_traj = lf_traj
        self.right_foot_traj = rf_traj
        lf_p_traj, rf_p_traj = self.get_smpl_foots_trajectory(body_vertices)
        self.left_p_foot_traj = lf_p_traj
        self.right_p_foot_traj = rf_p_traj
        
    def get_smpl_foots_keypoints_trajectory(self, body_vertices):
        lf_index = [3216, 3226, 3387]
        rf_index = [6617, 6624, 6787]
        return body_vertices[:, lf_index], body_vertices[:, rf_index]
    
    def get_smpl_foots_trajectory(self, body_vertices):
        with open('dataset/data/smpl_vert_segmentation.json', 'r') as f:
            body_segmentation = json.load(f)
        lf_index = body_segmentation['leftFoot'] 
        rf_index = body_segmentation['rightFoot']
        return body_vertices[:, lf_index], body_vertices[:, rf_index]
    
    def get_butt_verts(self, f: int=0):
        """ Get butt vertices of f-th frame

        Args:
            f: the index of frame

        Return:
            The butt vertices coordinates in 3D
        """
        return SMPLX_Util.get_butt_verts(self.body_vertices[f])
    
    def get_knee_verts(self, f: int=0):
        """ Get knee vertices of f-th frame

        Args:
            f: the index of frame

        Return:
            The knee vertices coordinates in 3D
        """
        return SMPLX_Util.get_knee_verts(self.body_vertices[f])
    
    def get_right_hand_position(self, f: int=0):
        """ Get right hand position of f-th frame

        Args:
            f: the index of frame

        Return:
            The right hand position coordinates in 3D
        """
        return SMPLX_Util.get_right_hand(self.body_vertices[f]).mean(axis=0)
    
    def get_body_orient(self, f: int=0, xy: bool=False):
        """ Get body orientation of f-th frame

        Args:
            f: the index of frame
        
        Return:
            The body orient
        """
        orient3D = SMPLX_Util.get_body_orient(self.joints_traj[f])

        if xy == True:
            return orient3D[0:2] / np.linalg.norm(orient3D[0:2])
        return orient3D
    
    def get_scene_occupancy(self, scene_labels: np.ndarray):
        """ Get occupied scene indices, objects occupancy (without floor, ceiling, unlabeled)

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the occupied space
        """
        ## containing floor, floor mat, ceiling, and unlabled vertices
        free_space_indices = (scene_labels == self.label_mapping['floor']) | (scene_labels == self.label_mapping['floor mat']) | (scene_labels == self.label_mapping['ceiling']) | (scene_labels == 0)
        return np.logical_not(free_space_indices)
    
    def get_scene_floor_occupancy(self, scene_labels: np.ndarray):
        """ Get scene floor indices

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the floor
        """
        floor_space_indices = (scene_labels == self.label_mapping['floor']) | (scene_labels == self.label_mapping['floor mat'])
        return floor_space_indices
    
    def get_scene_wall_occupancy(self, scene_labels: np.ndarray):
        """ Get scene wall indices

        Args:
            scene_labels: scene semantic lable array
        
        Return:
            A bool array, True indicate the wall
        """
        wall_space_indices = (scene_labels == self.label_mapping['wall']) | (scene_labels == self.label_mapping['window'])
        return wall_space_indices
    
    def get_surrounding_height(self, occupied_kdtree: KDTree, verts: np.ndarray, qurey_point_xy: np.ndarray):
        """ Get the height of surrounding floor/object for translating body, since the scannet scene floor may be uneven

        Args:
            occupied_kdtree: the kdtree of 2D floor points
            verts: floor vertices in 3D
            query_point_xy: a query point
        
        Return:
            The height of surrounding floor
        """
        if occupied_kdtree is None:
            return 0
        
        _, indic = occupied_kdtree.query(np.array([[*qurey_point_xy]]), k=20)
        indic = indic[0]
        return verts[indic][:, -1].mean()
    
    def get_valid_interact_object_list(self, related_object_group: list):
        """ Get the list of valid objects

        Args:
            related_object_group: the list of interact obejcts
        
        Return:
            Valid interactive object list and the object occurrence count
        """
        aggregation_file = os.path.join(config.scannet_folder, self.scene_id, self.scene_id + '.aggregation.json')
        segment_file = os.path.join(config.scannet_folder, self.scene_id, self.scene_id + '_vh_clean_2.0.010000.segs.json')

        with open(aggregation_file, 'r') as fp:
            scan_aggregation = json.load(fp)

        with open(segment_file, 'r') as fp:
            segment_info = json.load(fp)
            segment_indices = segment_info['segIndices']
        
        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)  # Add to each segment, its point indices
        
        ## iterate over all object
        all_objects = []
        occurrences = defaultdict(int)
        for object_info in scan_aggregation['segGroups']:
            object_instance_label = object_info['label']
            occurrences[object_instance_label] += 1

            semantic_label = self.label_mapping[ self.instance_to_semantic[object_instance_label] ]
            if semantic_label not in related_object_group: # interact with some selected object categories
                continue

            object_id = object_info['objectId']
            segments = object_info['segments']
            pc_loc = []
            for s in segments:
                pc_loc.extend(segment_indices_dict[s])
            object_pc = pc_loc

            all_objects.append((object_id, object_instance_label, semantic_label, object_pc))
        
        return all_objects, occurrences

    
    @staticmethod
    def calc_Mbbox(model: dict):
        """ Get transformation matrix of bounding box of scan2cad format

        Args:
            model: an instance annotation in scan2cad
        
        Return:
            The transformation matrix of the annotated instance
        """
        trs_obj = model["trs"]
        bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
        center_obj = np.asarray(model["center"], dtype=np.float64)
        trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
        q_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
        scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

        tcenter1 = np.eye(4)
        tcenter1[0:3, 3] = center_obj
        trans1 = np.eye(4)
        trans1[0:3, 3] = trans_obj
        rot1 = np.eye(4)
        rot1[0:3, 0:3] = Q(q_obj).rotation_matrix
        scale1 = np.eye(4)
        scale1[0:3, 0:3] = np.diag(scale_obj)
        bbox1 = np.eye(4)
        bbox1[0:3, 0:3] = np.diag(bbox_obj)
        M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
        return M
    
    @staticmethod
    def get_color_array(length, color: np.ndarray=np.array([64, 64, 64, 255], dtype=np.uint8)):
        color = np.ones((length, 4), dtype=np.uint8) * color
        return color
    
    @staticmethod
    def random_indices(length: int, count: int):
        """ Get random indices of range with length

        Args:
            length: length of the range, i.e. 0, ... ,length - 1
            count: sample count
        
        Return:
            indices list
        """
        indic = list(range(length))
        random.shuffle(indic)
        count = min(len(indic), count)
        return indic[0:count]
    
    def sample_proposal(self):
        pass



########################################
## Walk align class
########################################
class WalkAlign(ActionAlign):
    def __init__(self, *args, **kwargs):
        super(WalkAlign, self).__init__(*args, **kwargs)

        self.action = 'walk'
        #self.interact_object_group = [4, 5, 6, 7, 8, 14, 24, 33, 34] # bed, chair, sofa, table, door, desk, refridgerator, toilet, sink

        #self.TOWARD_ORIENTATION_CONSTRAINT = np.pi / 6
    
    def _detect_valid_for_walk_on_floor_plane(self, floor_plane_mask: np.array, all_points: np.ndarray, scale: float=FLOOR_PLANE_GRID_SCALE, delta: float=0.05):
        """ Detect the points for feet trajectary within the floor plane

        Args:
            floor_plane_mask: 2D floor plane mask
            all_points: all detected points
            scale: scale of the mask
            delta: default is 0.05
        
        Return:
            A bool value
        """
        H, W = floor_plane_mask.shape
        mask_1d = floor_plane_mask.reshape(-1)
        
        xs, ys = all_points[:, 0], all_points[:, 1]
        grid_x, grid_y = np.floor(xs / scale).astype(np.int32), np.floor(ys / scale).astype(np.int32)
        grid_idx = (grid_y * W + grid_x).clip(0, H*W-1)
        valid_points = (grid_x >= 0) & (grid_x < W) & (grid_y >= 0) & (grid_y < W) & (mask_1d[grid_idx] > 0)
        
        neighboring_grid_x = np.stack([grid_x-1, grid_x, grid_x+1, grid_x-1, grid_x+1, grid_x-1, grid_x, grid_x+1], axis=-1)
        neighboring_grid_y = np.stack([grid_y-1, grid_y-1, grid_y-1, grid_y, grid_y, grid_y+1, grid_y+1, grid_y+1], axis=-1)
        neighboring_in = np.stack([
            (xs-grid_x*scale) ** 2 + (ys-grid_y*scale) ** 2 < delta ** 2,
            (ys-grid_y*scale) ** 2 < delta ** 2,
            (xs-(grid_x+1)*scale) ** 2 + (ys-grid_y*scale) ** 2 < delta ** 2,
            (xs-grid_x*scale) ** 2 < delta ** 2,
            (xs-(grid_x+1)*scale) ** 2 < delta ** 2,
            (xs-grid_x*scale) ** 2 + (ys-(grid_y+1)*scale) ** 2 < delta ** 2,
            (ys-(grid_y+1)*scale) ** 2 < delta ** 2,
            (xs-(grid_x+1)*scale) ** 2 + (ys-(grid_y+1)*scale) ** 2 < delta ** 2,
        ], axis=-1)
        neighboring_grid_idx = (neighboring_grid_y * W + neighboring_grid_x).clip(0, H*W-1)
        invalid_neighbors = (neighboring_grid_x >= 0) & (neighboring_grid_x < W) & (neighboring_grid_y >= 0) & (neighboring_grid_y < H) & (mask_1d[neighboring_grid_idx] ==  0) & neighboring_in
        
        valid_points = valid_points & (invalid_neighbors.sum(-1) == 0)
        return valid_points.sum() == len(valid_points)

    def _sample_proposal_for_walk(self, min_x, max_x, min_y, max_y, bin_n: int=12, delta: float=0.5, max_samples=5, per_frame_on_ground=False):
        """ Sample valid position and orientation for walk

        Args:
            object_verts: occupied object vertices
            non_object_verts: non object vertices
            bin_n: orientation bin, default is 72
        
        Returns:
            Valid position of pelvis, Valid rotation of pelvis, 
        """
        
        ## sample grid
        proposed_points = []
        for x in np.arange(min_x, max_x, delta):
            for y in np.arange(min_y, max_y, delta):
                point = np.array([[x, y]])
                grid_x, grid_y = np.floor(x / self.scale).astype(np.int32), np.floor(y / self.scale).astype(np.int32)
                if max((x-min_x)**2, (x-max_x)**2) + max((y-min_y)**2, (y-max_y)**2) < self.traj_len ** 2:
                    continue
                if self.floor_plane_mask[grid_y, grid_x] > 0:
                    proposed_points.append(point.reshape(-1))
                # dist_obj, _ = object_kdtree.query(point, k=1)
                # dist_non_obj, _ = non_object_kdtree.query(point, k=1)
                
                # if floor_occupied_KDtree is not None:
                #     dist_floor, _ = floor_occupied_KDtree.query(point, k=1)
                #     if dist_floor[0][0] > delta:
                #         continue 

                # if dist_obj[0][0] < delta and dist_obj[0][0] > 0.5 * delta and dist_non_obj[0][0] > delta:
                #     proposed_points.append(point.reshape(-1))

        random.shuffle(proposed_points)
        ## filter proposed position
        body_orient_xy_last = self.get_body_orient(-1, xy=True)
        pelvis_xy_rotate = self.joints_traj[0, 0, 0:2] # last frame pelvis as anchor

        valid_trans = []
        valid_orient = []
        debug_body_orient = []
        debug_all_points = []
        for i, xy in enumerate(proposed_points):
            #body_trans_z = self.get_surrounding_height(floor_occupied_KDtree, floor_occupied_verts, xy)
            angles = np.arange(0, np.pi * 2, np.pi * 2 / bin_n)
            np.random.shuffle(angles)
            for angle in angles:
                trans = np.array([*xy, 0], dtype=np.float32)
                ## every frame of body should have no collision with scene, consider foot points
                all_points = np.concatenate((
                    self.left_foot_traj,
                    self.right_foot_traj,
                ), axis=1).reshape(-1, 3)
                #print(self.left_foot_traj.shape, self.right_foot_traj.shape, all_points.shape)
                if per_frame_on_ground:
                    num_frames = len(self.joints_traj)
                    trans = np.repeat(trans.reshape(1, 3), num_frames, axis=0)
                    trans[:, 2] = FLOOR_HEIGHT - all_points.reshape(num_frames, -1, 3)[:,:, 2].min(axis=-1)
                    all_points -= np.array([*pelvis_xy_rotate, 0])
                    all_points[:, 0:2] = rotate_2D_points_along_z_axis(all_points[:, 0:2], angle)
                    all_points = (all_points.reshape(num_frames, -1, 3) + trans[:, None]).reshape(-1, 3)
                else:
                    trans[2] = FLOOR_HEIGHT - all_points[:, 2].min()
                    all_points -= np.array([*pelvis_xy_rotate, 0])
                    all_points[:, 0:2] = rotate_2D_points_along_z_axis(all_points[:, 0:2], angle)
                    all_points += trans
                if not self._detect_valid_for_walk_on_floor_plane(self.floor_plane_mask, all_points):
                    continue
                valid_trans.append(trans)
                valid_orient.append(angle)
                debug_body_orient.append(rotate_2D_points_along_z_axis(body_orient_xy_last, angle))
                debug_all_points.append(all_points)
                if len(valid_trans) >= max_samples:
                    break
            if len(valid_trans) >= max_samples:
                break
        
        return np.array(valid_trans), np.array(valid_orient), np.array(debug_body_orient), np.array(debug_all_points)

    def sample_proposal(self, max_s_per_scene: int=5, use_lang: bool=False):
        """ Sample valid position and orientation for walk action
        """
        proposed_trans = []
        proposed_orient = []
        debug_body_o = []
        debug_feet_p = []
        H, W = self.floor_plane_mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W))
        scene_minx = grid_x[self.floor_plane_mask > 0].min() * self.scale
        scene_maxx = grid_x[self.floor_plane_mask > 0].max() * self.scale
        scene_miny = grid_y[self.floor_plane_mask > 0].min() * self.scale
        scene_maxy = grid_y[self.floor_plane_mask > 0].max() * self.scale
        scene_max_size = np.sqrt((scene_maxx - scene_minx) ** 2+ (scene_maxy - scene_miny) ** 2)
        #print(scene_minx, scene_miny, scene_maxx, scene_maxy)
        
        self.traj_len = np.sqrt(((self.joints_traj[-1, 0, 0:2] - self.joints_traj[0, 0, 0:2]) ** 2).sum())
        if self.traj_len > scene_max_size:
            ## scene is too small
            return proposed_trans, proposed_orient, debug_body_o, debug_feet_p

        #t = time.time()
        [valid_trans, valid_orient, debug_body_orient, debug_feet_points] = self._sample_proposal_for_walk(
            scene_minx, scene_maxx, scene_miny, scene_maxy,
            bin_n=36, max_samples=max_s_per_scene, per_frame_on_ground=True
        )
        ##print(time.time() - t)

        ## select max_s_per_object valid position for each object
        indic = self.random_indices(len(valid_trans), max_s_per_scene)

        
        for i, ind in enumerate(indic):
            proposed_trans.append(valid_trans[ind])
            proposed_orient.append(valid_orient[ind])
            debug_body_o.append(debug_body_orient[ind])
            debug_feet_p.append(debug_feet_points[ind])
    
        return proposed_trans, proposed_orient, debug_body_o, debug_feet_p

def load_mask(fpath):
    if fpath.split('.')[-1] == 'npy':
        mask = np.load(fpath) > 0
    elif fpath.split('.')[-1] == 'png':
        mask = np.array(Image.open(fpath))[:, :, 0] > 0
    H, W = mask.shape[:2]
    assert FLOOR_PLANE_GRID_SIZE % H == 0 and FLOOR_PLANE_GRID_SIZE % W == 0 and H == W
    mask = mask[:, None, :, None].repeat(FLOOR_PLANE_GRID_SIZE//H, axis=1).repeat(FLOOR_PLANE_GRID_SIZE//W, axis=3)
    mask = mask.reshape(FLOOR_PLANE_GRID_SIZE, FLOOR_PLANE_GRID_SIZE)
    return mask
    

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    amass_dir = './HumanML3D/amass_data'
    index_path = './HumanML3D/test_walk_ori_amass_path.txt'
    # index_path = f'{code_dir}/../../dataset/HumanML3D/train_walk_ori_amass_path.txt'
    filter_sequence = pd.read_csv(index_path, header=None, sep=',')    
    #print('Number of sequences: ', len(filter_sequence))
    # vert_segments = json.load('files/smplx_vert_segmentation.json')

    room_kind = 'livingroom'
    save_floor_dir = f'./mime_floor_plan/{room_kind}'
    save_fitting_dir = 'dataset/3dfront_fitting'
    all_room_list = os.listdir(save_floor_dir)
    #print('Number of rooms: ', len(all_room_list))
    info = []
    # idx = 0 # TODO: you need to traverse all elements 
    def _vis(one):
        try:
            align_path = '{}/align_data_obj_v2_test/{}'.format(save_fitting_dir, one[0]).replace('.npy', '.npy')
            npz_file = os.path.join(amass_dir, one[1].replace(' ./pose_data/', '').replace('npy', 'npz')) # remove the space
            start_f = one[2]
            end_f = one[3]
            # load motion sequences;
            (all_verts, all_joints, body_face_template) = load_motion(npz_file, start_f, end_f, return_all=True)
            all_verts = all_verts.detach().cpu().numpy()
            all_joints = all_joints.detach().cpu().numpy()
            align_info = dict(np.load(align_path, allow_pickle=True)[0])
            room_dir = str(align_info['scene'])
            floor_plane_mask = load_mask(os.path.join(room_dir, "floor_plane_mask.npy"))
            obj_mask_path = glob.glob("./atiss_object_mask/*{}/render_obj_bbox.png".format(room_dir.split('_')[-1]))[0]
            obj_mask = load_mask(obj_mask_path)
            floor_plane_mask = floor_plane_mask & (~obj_mask)
            floor_plane_mask_img = (floor_plane_mask.astype(np.uint8) * 255)[:, :, None].repeat(3, axis=-1)
            align = WalkAlign(floor_plane_mask, all_verts, all_joints, scale=FLOOR_PLANE_GRID_SCALE)
            pelvis_xy_rotate = align_info['pelvis_xy_rotate'] # last frame pelvis as anchor
            feet_points = np.concatenate((
                align.left_p_foot_traj,
                align.right_p_foot_traj,
            ), axis=1)
            
            len_traj = len(feet_points)
            feet_points -= np.array([*pelvis_xy_rotate, 0])
            feet_points[:, :, 0:2] = rotate_2D_points_along_z_axis(feet_points[:, :, 0:2].reshape(-1, 2), align_info['rotation'][0]).reshape(len_traj, -1, 2)
            feet_points = (feet_points.reshape(len_traj, -1, 3) + align_info['translation'][0][None, None]).reshape(len_traj, -1, 3)
            xs, ys = feet_points[:, :, 0], feet_points[:, :, 1]
            H, W = floor_plane_mask_img.shape[:2]
            grid_x, grid_y = np.floor(xs / FLOOR_PLANE_GRID_SCALE).astype(np.int32), np.floor(ys / FLOOR_PLANE_GRID_SCALE).astype(np.int32)
            RGB = np.stack([np.arange(0, len_traj, 1) / len_traj * 255, 255 - np.arange(0, len_traj, 1) / len_traj * 255, np.zeros(len_traj)], axis=-1)
            floor_plane_mask_img = floor_plane_mask_img.reshape(-1, 3)
            for i in range(len_traj):
                grid_idx = np.unique(np.clip(grid_y[i], 0, H-1) * W + np.clip(grid_x[i], 0 , W-1))
                floor_plane_mask_img[grid_idx] = RGB[i].astype(np.uint8)
            floor_plane_mask_img = floor_plane_mask_img.reshape(H, W, 3)
            return one[0], floor_plane_mask_img
        except:
            return one[0], None
    
    def _foo(one):
        dump_path = '{}/align_data_obj_v2_test/{}'.format(save_fitting_dir, one[0]).replace('.npy', '.npy')
        if os.path.exists(dump_path):
            return one[0], [], one
        try:
            npz_file = os.path.join(amass_dir, one[1].replace(' ./pose_data/', '').replace('npy', 'npz')) # remove the space
            start_f = one[2]
            end_f = one[3]

            # load motion sequences;
            (all_verts, all_joints, body_face_template) = load_motion(npz_file, start_f, end_f, return_all=True)
            all_verts = all_verts.detach().cpu().numpy()
            all_joints = all_joints.detach().cpu().numpy()

            num_frames = all_verts.shape[0]
            assert end_f - start_f == num_frames, "num_frames:{}, start_f: {}, end_f: {}".format(num_frames, start_f, end_f)
            # load 3D Scans from Scannets or 3D FRONT Dataset;
            
            all_verts = all_verts[0::CHECK_INTERVAL]
            all_joints = all_joints[0::CHECK_INTERVAL]

            # load 2D floor maps: 64->6.2*2 meter
            idx_list = np.arange(len(all_room_list))
            np.random.shuffle(idx_list)
            num_try = 0
            infos = []
            while len(idx_list) > 0 and num_try < MAX_TRY * NUM_SCENES and len(infos) < NUM_SCENES:
                t = time.time()
                room_idx = idx_list[0]
                room_dir = os.path.join(save_floor_dir, all_room_list[room_idx])
                room_mask = Image.open(os.path.join(room_dir, 'room_mask.png')).convert('RGB')
                floor_plane_mask = load_mask(os.path.join(room_dir, "floor_plane_mask.npy"))
                obj_mask_path = glob.glob("./atiss_object_mask/*{}/render_obj_bbox.png".format(room_dir.split('_')[-1]))[0]
                obj_mask = load_mask(obj_mask_path)
                floor_plane_mask = floor_plane_mask & (~obj_mask)
                align = WalkAlign(floor_plane_mask, all_verts, all_joints, scale=FLOOR_PLANE_GRID_SCALE)
                valid_trans, valid_orient, debug_body_orient, debug_feet_points = align.sample_proposal(NUM_SAMPLE_PROPOSAL)
                translation = [trans.max(axis=0) for trans in valid_trans]
                if len(valid_trans) == NUM_SAMPLE_PROPOSAL:
                    # debug_feet_points = debug_feet_points[0].reshape(num_frames, -1, 3)
                    # #print(debug_feet_points.shape)
                    # #print(debug_feet_points.min(axis=1).min(axis=0), debug_feet_points.min(axis=1).max(axis=0))
                    # #print('trans:{}, rotation:{}'.format(valid_trans, valid_orient))
                    # #print('{} <-> {}'.format(idx, room_idx))
                    infos.append(dict(
                        pelvis_xy_rotate=align.joints_traj[0, 0, 0:2],
                        action='walking', # action type
                        motion=npz_file, # motion file path
                        start_f=start_f, # start frame index
                        end_f=end_f, # end frame index
                        scene=room_dir, # scene file path
                        translation_per_frame=valid_trans, # translation (per-frame z-axis translation)
                        translation=translation, # translation
                        rotation=valid_orient, # rotation
                    ))
                    idx_list = idx_list[1:]
                    # break
                else:
                    num_try += 1
                    idx_list = idx_list[1:]
                    print('failed, try again, used time:', time.time() - t)
            return one[0], infos, one
                    # feet_vert_idx = np.array(vert_segments['leftFoot'] + vert_segments['rightFoot']
        except:
            return one[0], [], one
        
        # TODO: make sure all motions body feets are located inside the floor plan. 
    files = [filter_sequence.loc[idx] for idx in range(len(filter_sequence))]
    ones = []
    for one in tqdm(files):
        npz_file = os.path.join(amass_dir, one[1].replace(' ./pose_data/', '').replace('npy', 'npz')) # remove the space
        start_f = one[2]
        end_f = one[3]
        if os.path.exists(npz_file):
            ones.append(one)
            
    ones = ones
    os.makedirs(f'{save_fitting_dir}/align_data_obj_v2_test/', exist_ok=True)
    os.makedirs(f'{save_fitting_dir}/align_data_obj_v2_test/vis/', exist_ok=True)
    for f, info, one in p_uimap(_foo, ones, num_cpus=8):
        if len(info) > 0:
            dump_path = '{}/align_data_obj_v2_test/{}'.format(save_fitting_dir, f)
            np.save(dump_path, info)
            if len(info) > 0:
                f, vis_img = _vis(one)
                if vis_img is not None:
                    dump_path = '{}/align_data_obj_v2_test/vis/{}'.format(save_fitting_dir, f).replace('.npy', '.png')
                    Image.fromarray(vis_img).save(dump_path)
    # for f, vis_img in p_uimap(_vis, ones, num_cpus=8):
    #     if vis_img is not None:
    #         dump_path = 'files/align_data_obj_v2_test/vis/{}'.format(f).replace('.npy', '.png')
    #         Image.fromarray(vis_img).save(dump_path)
