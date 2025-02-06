import numpy as np
import random
import os
import sys
import math
import heapq
import pickle, json
from PIL import Image
from tqdm import tqdm
import glob
import scipy
import cv2
from scipy.ndimage.morphology import distance_transform_edt

DISTANCE_THRES_PIXEL = 6
FLOOR_PLANE_GRID_SIZE = 256 # size of floor plane grids (square)
FLOOR_PLANE_GRID_SCALE = 6.2*2/FLOOR_PLANE_GRID_SIZE # scale of floor plane grid (metre)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")


class Env:
    def __init__(self, obs_map):
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        xs, ys = np.nonzero(obs_map)
        self.obs = set([(y, x) for x, y in zip(xs, ys)])
        self.x_min = 0
        self.x_max = obs_map.shape[1]
        self.y_min = 0
        self.y_max = obs_map.shape[0]
        
    

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, env_map, s_start, s_goal, heuristic_type='euclid'):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = Env(~env_map)  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        if self.s_goal not in PARENT:
            return None
        
        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return path

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])


class MotionDatabase(object):
    def __init__(self, clip_window=2, fps=20, speed_groups=[0.1, 0.35, 0.65], threshold=0.05):
        #traj_clips, clip_velos = self.clip_trajs(trajs)
        with open('files/motion_clips_db_2d_1s_filtered.pkl', 'rb') as f:
            db_data = pickle.load(f)
        self.speed_groups = speed_groups
        self.threshold = threshold
        self.grouped_trajs = self.group_by_velo(db_data['trans_clip_2d'], db_data['orients_clip_2d'], db_data['velos_clip_2d'], db_data['dis_clip_2d'])
        self.fps=fps
        self.clip_length=int(fps * clip_window)
    
    def get_clip_categories(self, trans, rots=None, init_rot=None):
        angle_diff_thres = np.pi / 180 * 15
        if rots is None:
            rots = trans[..., 1:, :] - trans[..., :1, :]
            rots = np.arctan2(rots[..., :, 1], rots[..., :, 0])
            angle_diff =(init_rot - rots[..., 0]) % (np.pi*2)
        else:
            angle_diff = (rots[..., 1:] - rots[..., 0]) % (np.pi*2)
        angle_diff[angle_diff > np.pi] = np.pi*2 - angle_diff[angle_diff > np.pi]
        angle_diff = angle_diff.max(axis=-1)
        return np.where(angle_diff > angle_diff_thres, 1, 0)
    
    def get_traj_categories(self, trans, dis):
        line = np.linalg.norm(trans[..., -1, :] - trans[..., 0, :], axis=-1)
        dist = dis[..., -1] - dis[..., 0]
        return np.abs(line - dist) / dist < 0.05
        
    def group_by_velo(self, trans, orients, velos, dists):
        self.groups_trans = [[] for _ in self.speed_groups]
        self.groups_rots = [[] for _ in self.speed_groups]
        self.groups_velos = [[] for _ in self.speed_groups]
        self.groups_dists = [[] for _ in self.speed_groups]
        self.groups_cls = [[] for _ in self.speed_groups]
        for i in range(len(trans)):
            for j in range(len(self.speed_groups)-1, -1, -1):
                if velos[i] > self.speed_groups[j]:
                    T = trans[i] - trans[i][0]
                    rots = np.arctan2(orients[i][:, 1], orients[i][:, 0])
                    rot_mat_inv = np.array([
                        [np.cos(-rots[0]), np.sin(-rots[0])],
                        [-np.sin(-rots[0]), np.cos(-rots[0])],
                    ])
                    T = (T[None, :] @ rot_mat_inv)[0]
                    rots = rots - rots[0]
                    self.groups_trans[j].append(T)
                    self.groups_rots[j].append(rots)
                    self.groups_velos[j].append(velos[i])
                    self.groups_dists[j].append(dists[i])
                    #self.groups_cls[j].append(self.get_clip_categories(T, rots))
                    self.groups_cls[j].append(self.get_traj_categories(T, dists[i]))
                    # symmetric
                    T_sym = T * np.array([1, -1])
                    rots_sym = -rots
                    self.groups_trans[j].append(T_sym)
                    self.groups_rots[j].append(rots_sym)
                    self.groups_velos[j].append(velos[i])
                    self.groups_dists[j].append(dists[i])
                    #self.groups_cls[j].append(self.get_clip_categories(T_sym, rots_sym))
                    self.groups_cls[j].append(self.get_traj_categories(T_sym, dists[i]))
                    break
                    
    def try_match_clip(self, target_traj, traj_dis, init_rot, speed, check_occ):
        def smooth(pred, target):
            pred = pred.copy()
            #pred[-3] = (pred[-3]*0.1 + target[-3]*0.9)
            #pred[-2] = (pred[-2]*0.3 + target[-2]*0.7)
            #pred[-1] = (pred[-1]*0.5 + target[-1]*0.5)
            return pred
            
        speed = speed / self.fps
        speed = min(speed, traj_dis[-1]/self.clip_length-1e-6)
        gid = 0
        for i, v in enumerate(self.speed_groups):
            if speed>v:
                gid = i
        try_idx = list(range(len(self.groups_rots[gid])))
        random.shuffle(try_idx)
        target_interp = scipy.interpolate.interp1d(traj_dis, target_traj, axis=0)
        #target_pts = target_interp(np.arange(self.clip_length+1)*speed)
        rot_mat = np.array([
                        [np.cos(init_rot), np.sin(init_rot)],
                        [-np.sin(init_rot), np.cos(init_rot)],
                    ])
        all_trans = self.groups_trans[gid] @ rot_mat + target_traj[0]
        all_dists = self.groups_dists[gid]
        all_velos = self.groups_velos[gid]
        all_cls = np.array(self.groups_cls[gid])
        #print((all_cls==0).sum(), (all_cls==1).sum())
        target_pts = target_interp(np.clip(all_dists, 0., traj_dis[-1]))
        match_cost = np.linalg.norm(all_trans - target_pts, axis=-1)
        exceed_mask = (all_dists >= traj_dis[-1]).astype(np.float32)
        exceed_cost = ((1.0 - exceed_mask)*1e6 + exceed_mask*match_cost).min(axis=-1)[:, None]
        match_cost = (1.0 - exceed_mask)*match_cost + exceed_mask * exceed_cost
        match_cost_mean = match_cost.mean(axis=-1)
        #candidate_idx = np.nonzero(match_cost_mean < self.threshold)[0]
        #candidate_idx = np.abs(speed - all_velos).argsort()
        candidate_idx = match_cost_mean.argsort()
        candidate_cls = np.array(all_cls[candidate_idx])
        #target_cls = self.get_clip_categories(target_pts[candidate_idx], init_rot=init_rot)
        target_cls = self.get_traj_categories(target_pts[candidate_idx], np.clip(all_dists, 0., traj_dis[-1])[candidate_idx])
        candidates_group_0 = candidate_idx[target_cls == candidate_cls]
        candidates_group_1 = candidate_idx[target_cls != candidate_cls]
        #print('min(match_cost_mean)', match_cost_mean.min())
        #print('min(match_cost[-1]', match_cost[:, -1].min())
        #np.random.shuffle(candidate_idx)
        for candidate_group in (candidates_group_0, candidates_group_1):
            for idx in candidate_group:
                if match_cost[idx, -1] < self.threshold:# and match_cost_mean[-1] < self.threshold:
                    occ_flag = check_occ(all_trans[idx])
                    #print(occ_flag)
                    if occ_flag:
                        pred_traj = smooth(all_trans[idx], target_pts[idx])
                        while len(traj_dis) > 0 and traj_dis[0] <= all_dists[idx][-1]:
                            traj_dis = traj_dis[1:]
                            target_traj = target_traj[1:]
                        if len(traj_dis) > 0:
                            traj_dis = traj_dis - traj_dis[0] + np.linalg.norm(pred_traj[-1] - target_traj[0])
                        traj_dis = np.concatenate([[0.], traj_dis])
                        target_traj = np.concatenate([pred_traj[-1:], target_traj], axis=0)
                        return pred_traj, init_rot + self.groups_rots[gid][idx], target_traj, traj_dis
        return None, None, target_traj, traj_dis
            
    def try_match(self, target_traj, traj_dis, init_rot, speed, check_occ):
        #print('speed', speed)
        pred_traj = [target_traj[:1]]
        pred_rots = [np.array([init_rot])]
        while traj_dis[-1] > 0.05:
            new_p, new_r, target_traj, traj_dis = self.try_match_clip(target_traj, traj_dis, init_rot, speed, check_occ)
            if new_p is None:
                return None, None
            pred_traj.append(new_p[1:])
            pred_rots.append(new_r[1:])
            init_rot = new_r[-1]
        return np.concatenate(pred_traj, axis=0), np.concatenate(pred_rots, axis=0)
    
    def match_traj(self, target_traj, check_occ, rots_bin=36):
        for init_rot in np.arange(0, np.pi * 2, rots_bin):
            #init_rot = target_traj[1] - target_traj[0]
            #init_rot = np.arctan2(init_rot[1], init_rot[0])
            traj_dis = np.linalg.norm(target_traj[1:]-target_traj[:-1], axis=-1)
            traj_dis = np.cumsum(traj_dis)
            traj_dis = np.concatenate([[0.], traj_dis])
            speed_candidates = np.arange(0.2, 1.5, 0.1)
            np.random.shuffle(speed_candidates)
            for speed in speed_candidates:
                res = self.try_match(target_traj, traj_dis, init_rot, speed, check_occ)
                if res[0] is None:
                    continue
                else:
                    print('find one')
                    return res
            return None, None
        

class TrajectoryGenerator:
    def __init__(self, traj_planner=AStar, motion_db=None, map_size=256, map_scale=6.2*2/256, check_interval=0.05, sample_interval=6.2*2/256, n_angle_bin=36, min_occ_ratio=0.2):
        self.traj_planner = traj_planner
        self.motion_db = motion_db
        self.map_size = map_size
        self.map_scale = map_scale
        self.check_interval = check_interval
        self.sample_interval = sample_interval
        self.n_angle_bin = n_angle_bin
        self.min_occ_ratio = min_occ_ratio
    
    def _check_map(self, ground_map, point):
        if ((point > self.map_size*self.map_scale) | (point < 0)).any() > 0:
            return False
        grid_x, grid_y = point // self.map_scale
        return ground_map[int(grid_y), int(grid_x)]
    
    def _check_path(self, ground_map, pts):
        grid_x, grid_y = pts[:, 0] // self.map_scale, pts[:, 1] // self.map_scale
        map_idx = (grid_y*self.map_size+grid_x).astype(np.int32)
        ground_map = ground_map.reshape(-1)
        return (~ground_map[map_idx]).astype(np.float32).sum() > len(map_idx) * self.min_occ_ratio
    
    def _check_valid_path(self, ground_map, pts):
        grid_x, grid_y = pts[:, 0] // self.map_scale, pts[:, 1] // self.map_scale
        map_idx = (grid_y*self.map_size+grid_x).astype(np.int32)
        ground_map = ground_map.reshape(-1)
        return (~ground_map[map_idx]).astype(np.float32).sum() == 0
    
    def get_random_start_end(self, ground_map, occ_mask, distance):
        start_pts = []
        for x in np.arange(0., self.map_size * self.map_scale, self.sample_interval):
            for y in np.arange(0., self.map_size * self.map_scale, self.sample_interval):
                p = np.array((x, y))
                if self._check_map(ground_map, p):
                    start_pts.append(p)
        random.shuffle(start_pts)
        for start_p in start_pts:
            angles = np.arange(0, np.pi * 2, np.pi * 2 / self.n_angle_bin)
            np.random.shuffle(angles)
            for angle in angles:
                end_p = start_p + np.array([np.cos(angle)*distance, np.sin(angle)*distance])
                if self._check_map(ground_map, end_p):
                    pts = np.arange(0, distance, self.check_interval)
                    pts = np.stack([pts * np.cos(angle), pts * np.sin(angle)], axis=-1) + start_p
                    if self._check_path(occ_mask, pts):
                        yield (start_p//self.map_scale).astype(np.int32), (end_p//self.map_scale).astype(np.int32)
    
    def get_path(self, ground_map, occ_mask, distance):
        for start_p, end_p in self.get_random_start_end(ground_map, occ_mask, distance):
            #print((start_p[0], start_p[1]), (end_p[0], end_p[1]))
            planner = self.traj_planner(ground_map, (start_p[0], start_p[1]), (end_p[0], end_p[1]))
            plan_traj, close_set = planner.searching()
            #print(len(close_set))
            if plan_traj is not None:
                plan_traj = np.stack([(np.array(p)+0.5)*self.map_scale for p in plan_traj])
                result_traj, result_rots = self.motion_db.match_traj(plan_traj, lambda pts: self._check_valid_path(ground_map, pts))
                if result_traj is not None:
                    return plan_traj, result_traj, result_rots
            # TODO: replace planned traj with amass trajectory
        return None, None, None
    
    def vis(self, ground_map, plan_traj, traj, rots, save_name, save_video=False):
        if traj is None:
            return None
        if save_video:
            video_writer = cv2.VideoWriter(save_name+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,1024))
        img = np.zeros((*ground_map.shape, 3), dtype=np.uint8)
        img += ground_map[:, :, None].astype(np.uint8) * 255
        img = img.reshape(-1, 3)
        plan_traj_grid = np.floor(plan_traj[:, 1]/self.map_scale) * self.map_size + np.floor(plan_traj[:, 0]/self.map_scale)
        img[plan_traj_grid.astype(np.int32)] = np.array([255, 0, 0])
        # traj_grid = np.floor(traj[:, 1]/self.map_scale) * self.map_size + np.floor(traj[:, 0]/self.map_scale)
        # img[traj_grid.astype(np.int32)] = np.array([0, 255, 0])
        img = img.reshape(*ground_map.shape, 3)
        img = np.array(Image.fromarray(img).resize((self.map_size*4, self.map_size*4)))
        #rots = rots[::20]
        #traj = traj[::20]
        for i, (r, p) in enumerate(zip(rots, traj)):
            st = p/self.map_scale*4
            en = st + np.array((np.cos(r), np.sin(r)))*40
            if i < len(traj)-1:
                next_p = traj[i+1] / self.map_scale * 4
                cv2.line(img, (int(st[0]),int(st[1])), (int(next_p[0]),int(next_p[1])), (128, 128, 255), 2)
            if save_video:
                img2 = img.copy()
                cv2.arrowedLine(img2, (int(st[0]),int(st[1])), (int(en[0]), int(en[1])), (0, 0, 255), 1, 0, 0, 0.2)
                video_writer.write(img2)
        if save_video:
            video_writer.release()
            #cv2.arrowedLine(img, (int(st[0]),int(st[1])), (int(en[0]), int(en[1])), (0, 0, 255), 1, 0, 0, 0.2)
            
        Image.fromarray(img).save(save_name+'.png')
    
    
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
    
    room_kind = 'livingroom'
    save_floor_dir = f'./mime_floor_plan/{room_kind}'
    distances = [1., 2., 3., 4., 5., 6.] # add noise. +- 0.5
    all_room_list = os.listdir(save_floor_dir)
    os.makedirs("files/astar_data_2s_1108/vis", exist_ok=True)
    save_dir = "files/astar_data_2s_1108"
    
    generator = TrajectoryGenerator(motion_db=MotionDatabase(speed_groups=[0.1, 0.65]))
    
    for i in tqdm(range(len(all_room_list))):
        room_dir = os.path.join(save_floor_dir, all_room_list[i])
        floor_plane_mask = load_mask(os.path.join(room_dir, "floor_plane_mask.npy"))
        obj_mask_path = glob.glob("./atiss_object_mask/*{}/render_obj_bbox.png".format(room_dir.split('_')[-1]))[0]
        obj_mask = load_mask(obj_mask_path)
        floor_plane_mask = floor_plane_mask & (~obj_mask)
        floor_plane_dis = distance_transform_edt(floor_plane_mask)
        floor_plane_mask_dis = (floor_plane_dis > DISTANCE_THRES_PIXEL)
        results = []
        results_vis = []
        for j, dis in enumerate(distances):
            dis += random.random() - 0.5
            plan, traj, rots = generator.get_path(floor_plane_mask_dis, floor_plane_mask, dis)
            vis = generator.vis(floor_plane_mask, plan, traj, rots, f"{save_dir}/vis/{i}_{all_room_list[i]}_{j+1}", save_video=True)
            results.append(dict(plan=plan, traj=traj, rots=rots))
        np.save(f"{save_dir}/{i}_{all_room_list[i]}.npy", results)
        