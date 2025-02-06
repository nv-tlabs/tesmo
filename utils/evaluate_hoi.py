import torch
import os
import sys
from utils.smpl_util import load_body_models, run_smplx_model
import numpy as np
import trimesh
import data_loaders.humanml.utils.paramUtil as paramUtil
# import paramUtil
from visualize.vis_utils import save_sample_poses

skeleton = paramUtil.t2m_kinematic_chain

code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sample_bone_kinematic_tree(sample_points, skeleton, num_samples=20):
    # sample_points: batch, frames, joints, 3
    # sample_points should be a numpy array

    sample_points_all = []
    sample_points_all.append(sample_points)

    for i in range(len(skeleton)):
        for j in range(len(skeleton[i])-1):
            start_idx = skeleton[i][j]
            end_idx = skeleton[i][j+1]
            start_point = sample_points[:, :, start_idx, :]
            end_point = sample_points[:, :, end_idx, :]

            # Perform linear interpolation
            interpolated_points = []
            for weight in np.linspace(0, 1, num_samples + 2)[1:-1]:
                interpolated_point = (1 - weight) * start_point + weight * end_point
                interpolated_points.append(interpolated_point)

            # Convert the result to a numpy array
            interpolated_array = np.stack(interpolated_points, axis=2)

            sample_points_all.append(interpolated_array)

    return np.concatenate(sample_points_all, axis=2)


def sample_bone_kinematic_tree_th(sample_points, skeleton, num_samples=20):
    """
    Sample points along the bones of a kinematic tree using PyTorch.

    :param sample_points: A PyTorch tensor of shape (batch, frames, joints, 3)
    :param skeleton: List of lists representing the kinematic chain.
    :param num_samples: Number of samples per bone.
    :return: A PyTorch tensor of sampled points.
    """
    sample_points_all = [sample_points]

    for i in range(len(skeleton)):
        for j in range(len(skeleton[i]) - 1):
            start_idx = skeleton[i][j]
            end_idx = skeleton[i][j + 1]
            start_point = sample_points[:, :, start_idx, :]
            end_point = sample_points[:, :, end_idx, :]

            # Perform linear interpolation
            interpolated_points = []
            for weight in torch.linspace(0, 1, num_samples + 2, device=sample_points.device)[1:-1]:
                interpolated_point = (1 - weight) * start_point + weight * end_point
                interpolated_points.append(interpolated_point)

            # Convert the result to a PyTorch tensor
            interpolated_tensor = torch.stack(interpolated_points, dim=2)

            sample_points_all.append(interpolated_tensor)

    return torch.cat(sample_points_all, dim=2)


def get_template_outsurface_joints(body_model='smpl', num_points=1000):

    roots = torch.zeros(1,1, 3).cuda()
    betas = torch.zeros(1, 16).cuda()
    thetas = torch.zeros(1, 1, 22, 3).cuda()

    if body_model == 'smpl':

        bm_dict = load_body_models()
        
        smpl_joints, smpl_verts, mesh_faces = run_smplx_model(roots, thetas, betas, None, bm_dict)

        # import pdb;pdb.set_trace()
        body_mesh = trimesh.Trimesh(vertices=smpl_verts[0, 0].cpu().numpy(), faces=mesh_faces.cpu().numpy())
        sample_path = os.path.join(code_dir, 'dataset', 'template_outsurface_points_no_back.ply')
        if os.path.exists(sample_path):
            sample_points = trimesh.load(sample_path, process=False).vertices
        else:
            sample_points = body_mesh.sample(num_points)
            trimesh.Trimesh(vertices=sample_points).export(sample_path)
        
        return  smpl_joints.squeeze().cpu().numpy(), sample_points

def create_bones_from_kinematic_tree(joint_positions, kinematic_chain):
    bones = []
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            start_joint = joint_positions[chain[i]]
            end_joint = joint_positions[chain[i + 1]]
            bones.append((start_joint, end_joint))
    return bones

def sample_points_around_bones(bones, points_per_bone):
    sampled_points = []
    for start, end in bones:
        for _ in range(points_per_bone):
            t = np.random.uniform(0, 1)
            point = start + t * (end - start)
            sampled_points.append(point)
    return np.array(sampled_points)

def calculate_skinning_weights(mesh_points, bones):
    skinning_weights = np.zeros((len(mesh_points), len(bones)))
    for i, point in enumerate(mesh_points):
        for j, (start, end) in enumerate(bones):
            nearest_point = nearest_point_on_line_segment(point, start, end)
            distance = np.linalg.norm(point - nearest_point)
            skinning_weights[i, j] = 1 / (distance + 1e-5)
    weight_sums = skinning_weights.sum(axis=1, keepdims=True)
    skinning_weights /= weight_sums
    return skinning_weights

def nearest_point_on_line_segment(point, start, end):
    line_vec = end - start
    point_vec = point - start
    length = np.linalg.norm(line_vec)
    line_unitvec = line_vec / length
    point_vec_scaled = point_vec / length
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = start + t * line_vec
    return nearest

def update_mesh_points(mesh_points, skinning_weights, new_bones): # make it into batch wise.
    updated_points = np.zeros_like(mesh_points)
    for i, point in enumerate(mesh_points):
        for j, (start, end) in enumerate(new_bones):
            updated_points[i] += skinning_weights[i, j] * nearest_point_on_line_segment(point, start, end)
    return updated_points

from trimesh import Trimesh
def save_single_pose(input_poses, save_dir, post_fix):
    poses_mesh = Trimesh(vertices=input_poses, process=False)
    poses_mesh.export(os.path.join(save_dir, f'poses_{post_fix}.obj'))


class KinematicTree:
    def __init__(self, joint_positions, kinematic_chain):
        self.bones = self.create_bones(joint_positions, kinematic_chain)

    @staticmethod
    def create_bones(joint_positions, kinematic_chain):
        bones = []
        for chain in kinematic_chain:
            for i in range(len(chain) - 1):
                start_joint = joint_positions[chain[i]]
                end_joint = joint_positions[chain[i + 1]]
                bones.append((start_joint, end_joint))
        return bones

def calculate_initial_distances(sampled_points, kinematic_tree):
    distances = []
    for point in sampled_points:
        bone_distances = [distance_to_bone(point, bone) for bone in kinematic_tree.bones]
        distances.append(bone_distances)
    return np.array(distances)

def distance_to_bone(point, bone):
    start, end = bone
    nearest_point = nearest_point_on_line_segment(point, start, end)
    return np.linalg.norm(point - nearest_point)

def nearest_point_on_line_segment(point, start, end):
    line_vec = end - start
    point_vec = point - start
    length = np.linalg.norm(line_vec)
    line_unitvec = line_vec / length
    point_vec_scaled = point_vec / length
    t = np.dot(line_unitvec, point_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = start + t * line_vec
    return nearest

def update_sampled_points(sampled_points, initial_distances, kinematic_tree):
    updated_points = []
    for point, distances in zip(sampled_points, initial_distances):
        displacement = np.zeros(3)
        for distance, bone in zip(distances, kinematic_tree.bones):
            nearest_point = nearest_point_on_line_segment(point, *bone)
            direction = (point - nearest_point) / (np.linalg.norm(point - nearest_point) + 1e-5)
            displacement += direction * distance
        updated_points.append(displacement)
    return np.array(updated_points)

class SampleBodySurface:
    def __init__(self, num_points_per_bone=20, device='cuda:0'):
        self.joints, self.sampled_points = get_template_outsurface_joints()
        # TODO: remove hands.

        self.kinematic_chain = skeleton
        # import pdb;pdb.set_trace()
        # self.bones = create_bones_from_kinematic_tree(self.joints, self.kinematic_chain)
        self.bones = sample_bone_kinematic_tree(self.joints[None, None], self.kinematic_chain, num_samples=num_points_per_bone)[0, 0]
        self.vectors_to_bones, self.nearest_bone_indices = self.calculate_vectors_to_bones()
        self.device = device
        self.vectors_to_bones = torch.from_numpy(self.vectors_to_bones).to(device)
        self.nearest_bone_indices = torch.from_numpy(np.array(self.nearest_bone_indices)).to(device)

    def calculate_vectors_to_bones(self):
        vectors = []
        nearest_bone_indices = []
        for point in self.sampled_points:
            nearest_bone_idx, nearest_point = self.find_nearest_bone_and_point(point)
            vector_to_bone = point - nearest_point
            vectors.append(vector_to_bone)
            nearest_bone_indices.append(nearest_bone_idx)
        return np.array(vectors), nearest_bone_indices

    def find_nearest_bone_and_point(self, point):
        min_distance = float('inf')
        nearest_bone_idx = -1
        nearest_point_on_bone = None
        for idx, bone in enumerate(self.bones):
            point_on_bone = bone
            distance = np.linalg.norm(point - point_on_bone)
            if distance < min_distance:
                min_distance = distance
                nearest_bone_idx = idx
                nearest_point_on_bone = point_on_bone
        return nearest_bone_idx, nearest_point_on_bone

    def update_sampled_points(self, new_joint_positions): # support batch wise.
        new_bones = sample_bone_kinematic_tree(new_joint_positions[None, None], self.kinematic_chain, num_samples=20)[0,0]
        updated_points = []
        frame = new_joint_positions.shape[-1]
        for vector, bone_idx in zip(self.vectors_to_bones, self.nearest_bone_indices):
            # nearest_point_on_new_bone = nearest_point_on_line_segment(self.sampled_points[bone_idx], *new_bones[bone_idx])
            nearest_point_on_new_bone = new_bones[bone_idx]
            updated_point = nearest_point_on_new_bone + vector.reshape(-1, 1).repeat(frame, 1)
            updated_points.append(updated_point)

        return np.array(updated_points)

    def update_sampled_points_th(self, new_joint_positions):
        """
        Update sampled points with new joint positions.
        
        :param new_joint_positions: A tensor of shape (22, 3, frames)
        :return: Updated points as a PyTorch tensor
        """

        if new_joint_positions.dim() == 4:
            # input: batch, frame, 22, 3
            new_bones = sample_bone_kinematic_tree_th(new_joint_positions, self.kinematic_chain, num_samples=20) # b, frames, n, 3

            updated_points = []
            batch, frame = new_joint_positions.shape[0], new_joint_positions.shape[1]
            for vector, bone_idx in zip(self.vectors_to_bones, self.nearest_bone_indices):
                nearest_point_on_new_bone = new_bones[:, :, bone_idx]
                # import pdb;pdb.set_trace()
                updated_point = nearest_point_on_new_bone + vector.reshape(1, 1, -1).repeat(batch, frame, 1)
                updated_points.append(updated_point)

            # Reshape updated points to match the frames dimension
            updated_points = torch.stack(updated_points).permute(1, 2, 0, 3)  # Shape: (frames, bones, 3

        else:
            # new_joint_positions is expected to be a PyTorch tensor
            # Reshape and sample new bone kinematic tree
            new_joint_positions = new_joint_positions.permute(2, 0, 1)  # Change to (frames, 22, 3)
            new_bones = sample_bone_kinematic_tree_th(new_joint_positions[None], self.kinematic_chain, num_samples=20)[0] # frames, n, 3
            
            # Flatten the frames and bones dimensions for vectorized computation
            # new_bones = new_bones.reshape(-1, 3)  # Shape: (frames * bones, 3)

            updated_points = []
            frame = new_joint_positions.shape[0]
            for vector, bone_idx in zip(self.vectors_to_bones, self.nearest_bone_indices):
                nearest_point_on_new_bone = new_bones[:, bone_idx]
                # import pdb;pdb.set_trace()
                updated_point = nearest_point_on_new_bone + vector.reshape(1, -1).repeat(frame, 1)
                updated_points.append(updated_point)

            # Reshape updated points to match the frames dimension
            updated_points = torch.stack(updated_points).permute(1, 0, 2)  # Shape: (frames, bones, 3
        return updated_points

