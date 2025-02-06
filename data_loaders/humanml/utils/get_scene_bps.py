
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from pytorch3d.loss import chamfer_distance
import os
import trimesh
### for sdf.
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_voxels, get_surface_point_cloud
import skimage
import pyrender
import pickle
from trimesh import sample
import open3d as o3d

SAMPLE_POINTS_NUM=3000 # the sampled number points would not be same.
CENTER_ALIGN=True
USE_TRIMESH_CONTAINS=True
# otherwises: zero_height_align.

ROOT_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

########################### bps encoding  ############################
# given the goal position: this is general enough for different class object, scans, large or small objects.
# Sample uniformly in the unit ball (random) | 
def bps_gen_ball_inside(n_bps=1000, random_seed=100): # with 1 radius around the center of the object.
    np.random.seed(random_seed)
    x = np.random.normal(size=[n_bps, 3])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms  # points on the unit ball surface
    r = np.random.uniform(size=[n_bps, 1])
    u = np.power(r, 1.0 / 3)
    basis_set = 1 * x_unit * u  # basic set coordinates, [n_bps, 3]
    return basis_set

def generate_random_points_in_cylinder(radius=1.0, height=2.0, n_bps=1000, random_seed=100):
    # Generate random cylindrical coordinates
    np.random.seed(random_seed)
    theta = 2 * np.pi * np.random.rand(n_bps)  # Angle
    r = radius * np.sqrt(np.random.rand(n_bps))  # Radius
    z = height * np.random.rand(n_bps)  # Height

    # Convert cylindrical coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.stack([x, y, z])

# Parameters of the cylinder
cylinder_radius = 5.0
cylinder_height = 10.0
num_points = 1000


def load_bps_set(save_dir=f'{ROOT_CODE_DIR}/dataset/SAMP_obj_bps_new'): # this is for v6 trained model.
    # import pdb;pdb.set_trace()
    os.makedirs(save_dir, exist_ok=True)
    
    if CENTER_ALIGN:
        if not os.path.exists(os.path.join(save_dir, 'bps_set.npy')):
            bps_set = bps_gen_ball_inside(n_bps=1000, random_seed=100)
            with open(os.path.join(save_dir, 'bps_set.npy'), 'wb') as f:
                np.save(f, bps_set)
        else:
            with open(os.path.join(save_dir, 'bps_set.npy'), 'rb') as f:
                bps_set = np.load(f)
    else:
        if not os.path.exists(os.path.join(save_dir, 'bps_set_cylinder.npy')):
            bps_set = generate_random_points_in_cylinder()
            with open(os.path.join(save_dir, 'bps_set_cylinder.npy'), 'wb') as f:
                np.save(f, bps_set)
        else:
            with open(os.path.join(save_dir, 'bps_set_cylinder.npy'), 'rb') as f:
                bps_set = np.load(f)
    return bps_set

# TODO: use torch.cdist to get the p-norm between two set of vectors.

### add pointnet pretrained encoder;

### TODO: bps encoders;

def get_one_directional_bps_feature(bps_set, point_set, return_bps_indices=False, device=None, return_distance_vector=False): # to different joints
    # bps_set: [batch, n_bps, 3]
    # point_set: [batch, n_points, 3]
    # return: [batch, n_bps, 1]
    # return_bps_indices: [batch, n_bps, 1]

    assert len(bps_set.shape) == 3 and len(point_set.shape) == 3

    if isinstance(bps_set, np.ndarray):
        if device is not None:
            bps_set = torch.from_numpy(bps_set).float().to(device)    
        else:
            bps_set = torch.from_numpy(bps_set).float().cuda()
    if isinstance(point_set, np.ndarray):
        if device is not None:
            point_set = torch.from_numpy(point_set).float().to(device)
        else:
            point_set = torch.from_numpy(point_set).float().cuda()
    
    
    distances_A_to_B = torch.cdist(bps_set, point_set) 
    nearest_distances_A_to_B, nearest_indices_A_to_B = torch.min(distances_A_to_B, dim=-1)

    # import pdb;pdb.set_trace()
    # Gather the nearest points
    # indices = indices.squeeze(-1)  # Shape becomes [batch_size, 100]
    batch_indices = torch.arange(nearest_indices_A_to_B.shape[0]).unsqueeze(-1).expand_as(nearest_indices_A_to_B)

    nearest_points = point_set[batch_indices, nearest_indices_A_to_B]

    # Compute distance vectors
    distance_vectors = nearest_points - bps_set

    if return_distance_vector:
        return_value = distance_vectors
    else:
        return_value = nearest_distances_A_to_B
    if return_bps_indices:
        # nearest_indices_A_to_B = torch.argmin(distances_A_to_B, dim=-1)
        return return_value, nearest_indices_A_to_B
    else:
        return return_value


# initailize the sampled basic points. 

# extract bps distance feature for each objects and save it down.

# get the bps distance feature for each body poses;

def get_original_mesh(obj_path, trans_dict, center_align=True):
    # coordinate system: xz is the plane, y is up. | mesh coordinates
    # self.name = '{}_{}_{}'.format(self.obj_folder, obj_category, self.obj_name)
    mesh = trimesh.load(obj_path, force='mesh', process=False)  # process=True will change vertices and cause error!
    
    # import pdb;pdb.set_trace()
    # if debug:
    #     print(f'cp {obj_path} {os.path.join(debug_dir, "old_" + obj_name + ".obj")}')
    #     os.system(f'cp {obj_path} {os.path.join(debug_dir, "old_" + obj_name + ".obj")}')
    new_transform = np.eye(4)

    # import pdb;pdb.set_trace()
    if not isinstance(trans_dict['scale'], float):
        new_transform[:3,:3] = trans_dict['scale'][:3,:3] # three dimension.
    else:
        new_transform[3,3] = new_transform[3,3] * trans_dict['scale'] # ! only works on real scale.

    mesh.apply_transform(new_transform)
    
    center_transl_max = mesh.vertices.max(0)
    center_transl_min = mesh.vertices.min(0)
    center_transl = (center_transl_max + center_transl_min) / 2.0

    # import pdb;pdb.set_trace()
    if not CENTER_ALIGN:
        center_transl[1] = mesh.vertices.min(0)[1] # height.
    
    # print('center_transl: ', center_transl)
    mesh.apply_translation(-center_transl)

    return mesh, center_transl

class ObjectSceneBPS:
    def __init__(self, 
        bps_set,
        obj_path, 
        trans_dict=None, 
        save_dir=None, 
        obj_category='chair', 
        build=False, \
        voxel_resolution = 128, 
        bps_metric=1.0,
        debug=False,
        debug_dir='debug_results/bps_dir',
        left_right_flip=False,
        return_distance_vector=False): 

        # input object is a real size;
        self.obj_path = obj_path
        self.return_distance_vector = return_distance_vector
        
        # import pdb;pdb.set_trace()
        if isinstance(self.obj_path, trimesh.Trimesh):
            self.mesh = self.obj_path
            self.obj_id = '0'
        else:
            # import pdb;pdb.set_trace()
            obj_name = self.obj_path.split('/')[-2]
            if debug:
                os.makedirs(debug_dir, exist_ok=True)

            self.obj_category = obj_category
            self.obj_folder = os.path.dirname(obj_path)
            self.obj_id = self.obj_folder.split('/')[-1]
            # self.obj_name = os.path.basename(obj_path)

            # import pdb;pdb.set_trace()
            self.mesh, self.center_transl = get_original_mesh(self.obj_path, trans_dict, center_align=CENTER_ALIGN)
            
            # import pdb;pdb.set_trace()
            # TODO: resample the mesh to have the same number of vertices.
            
        def trimesh2open3d(trimesh_mesh):
            vertices = np.asarray(trimesh_mesh.vertices)
            faces = np.asarray(trimesh_mesh.faces)

            # Create an Open3D mesh object
            o3d_mesh = o3d.geometry.TriangleMesh()

            # Set vertices and faces
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            return o3d_mesh

        mesh = trimesh2open3d(self.mesh)
        pcd = mesh.sample_points_uniformly(number_of_points=SAMPLE_POINTS_NUM)
        # import pdb;pdb.set_trace()
        # points = sample.sample_surface_even(self.mesh, SAMPLE_POINTS_NUM)[0]
        self.sample_points = np.asarray(pcd.points)

        if debug:
            print('save to', os.path.join(debug_dir, obj_name + ".obj"))
            self.mesh.export(os.path.join(debug_dir, obj_name + ".obj"))
            trimesh.Trimesh(self.sample_points).export(os.path.join(debug_dir, obj_name + "_sample.ply"))

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        self.bps_path = os.path.join(save_dir, '{}_{}_bps.npy'.format(obj_category, self.obj_id))
        
        if left_right_flip:
            # import pdb;pdb.set_trace()
            self.bps_path = self.bps_path.replace('.npy', '_flip.npy')
            self.mesh.vertices[:, 0] *= -1
            self.mesh.faces = self.mesh.faces[:, ::-1]

            # Alternatively, you can flip normals explicitly:
            # flipped_mesh.vertex_normals *= -1
            
        print('save bps into ', self.bps_path)
        os.makedirs(os.path.dirname(self.bps_path), exist_ok=True)
        
        # obj_verts = self.mesh.vertices
        obj_verts = self.sample_points
        # print(obj_verts.mean(0))
        
        bps_feat = get_one_directional_bps_feature(bps_set[None], obj_verts[None], return_distance_vector=self.return_distance_vector)
        with open(self.bps_path, 'wb') as f:
            np.save(f, bps_feat.cpu().numpy())
        self.bps_feat = bps_feat

        import pdb;pdb.set_trace()

class ObjectScene:
    def __init__(self, obj_path, trans_dict=None, save_dir=None, obj_category='chair', build=False, \
        voxel_resolution = 128, extent=1.5):

        # input object is a real size;
        
        # import pdb;pdb.set_trace()
        self.obj_path = obj_path
        if isinstance(obj_path, trimesh.Trimesh):
            self.mesh = obj_path
            self.obj_id = '0' 
        else:
            self.obj_category = obj_category
            self.obj_folder = os.path.dirname(obj_path)
            self.obj_id = self.obj_folder.split('/')[-1]
            self.obj_name = os.path.basename(obj_path)
            
            self.mesh, self.center_transl = get_original_mesh(self.obj_path, trans_dict, center_align=CENTER_ALIGN)

        # self.transl_xz = trans_dict['trans'][[0,2]]
        # self.rot_y_mat = trans_dict['rotation']
        # import pdb;pdb.set_trace()

        self.floor_height = 0

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        if extent != 1.5:
            self.sdf_path = os.path.join(save_dir, '{}_{}_{}_sdf_grad.pkl'.format(obj_category, self.obj_id, extent))
        else:
            self.sdf_path = os.path.join(save_dir, '{}_{}_sdf_grad.pkl'.format(obj_category, self.obj_id))

        print('save sdf into ', self.sdf_path)

        if build or not os.path.exists(self.sdf_path): # for evaluation.

            extents = self.mesh.bounding_box.extents
            
            # x,y,z: y is height
            # extents = np.array([extents[0] + extent, 0.5, extents[2] + extent]) # 

            # transform = np.array([[1.0, 0.0, 0.0, 0],
            #                       [0.0, 1.0, 0.0,  -0.25],
            #                       [0.0, 0.0, 1.0, 0],
            #                       [0.0, 0.0, 0.0, 1.0],
            #                       ])
            # transform[[0, 2], 3] += self.mesh.centroid[[0,2]]
            # floor_mesh = trimesh.creation.box(extents=extents,
            #                                   transform=transform,
            #                                   )
        # # scene_mesh = self.mesh + floor_mesh
            
            # import pdb;pdb.set_trace()
            # scene_mesh.show()
            scene_extents = extents + np.array([2*extent, 0.2, 2*extent]) # expand the 2meters along the xz plane;
            scene_scale = np.max(scene_extents) * 0.5
            
            # TODO: generate the newest sdf volume: 0.5, 1.0 extension | use kaolin to get the sdf value.
            # import pdb;pdb.set_trace()
            m_verts = self.mesh.vertices
            m_faces = self.mesh.faces
            scene_mesh = trimesh.Trimesh(vertices=m_verts, faces=m_faces, process=False)
            scene_centroid = self.mesh.bounding_box.centroid
            scene_mesh.vertices -= scene_centroid # center to origin
            scene_mesh.vertices /= scene_scale # make the maximum extents length to be 1 
            sign_method = 'normal'
            
            surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='scan', bounding_radius=1, scan_count=100,
                                                          scan_resolution=400, sample_point_count=10000000, calculate_normals=(sign_method == 'normal')) # 3 meters.

            # ! update the code to use new check_sign from kaolin. 
            sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', sample_count=11, pad=False, check_result=False, return_gradients=True)
            
            # sdf_dict for the translation.
            
            # import pdb;pdb.set_trace()
            if USE_TRIMESH_CONTAINS:
                # TODO: calculcate new sign of the sdf.
                voxel_size = 2 * scene_scale / voxel_resolution

                # Create a meshgrid of indices
                grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(voxel_resolution), 
                                                        torch.arange(voxel_resolution), 
                                                        torch.arange(voxel_resolution))

                # Calculate the center coordinates of each voxel
                voxel_centers = torch.stack((grid_x, grid_y, grid_z), dim=-1).float() * voxel_size + 0.5 * voxel_size + scene_centroid - scene_scale

                sign = scene_mesh.contains(voxel_centers.cpu().numpy().reshape(-1, 3)).reshape(voxel_resolution, voxel_resolution, voxel_resolution)
                print(f'inside: {sign.sum()}')
                sign = sign * -2 + 1
                sign = torch.tensor(sign)
                sdf_grid = np.abs(sdf_grid) * sign.numpy()
                
                if 0:
                    # 
                    # Define the path to save the PLY file
                    ply_file_path = "inside_signed_voxels.ply"

                    # Filter voxel centers based on sign (inside voxels)
                    inside_voxel_centers = voxel_centers[sign == -1].cpu().numpy().reshape(-1, 3)

                    # Create a Trimesh object from the inside voxel centers
                    inside_mesh = trimesh.Trimesh(vertices=inside_voxel_centers, faces=[])

                    # Translate mesh to the original scene centroid
                    inside_mesh.vertices += scene_centroid.cpu().numpy()

                    # Save the mesh to a PLY file
                    inside_mesh.export(ply_file_path)

                    print("PLY file saved successfully!")

            sdf_dict = {
                'grid': sdf_grid * scene_scale,
                'gradient_grid': gradient_grid,
                'dim': voxel_resolution,
                'centroid': scene_centroid,
                'scale': scene_scale,
            }
            print('save to ', self.sdf_path)
            with open(self.sdf_path, 'wb') as f:
                pickle.dump(sdf_dict, f)
        else:
            print('load ', self.sdf_path)
            with open(self.sdf_path, 'rb') as f:
                sdf_dict = pickle.load(f)

        self.sdf_dict = sdf_dict
    
    def calc_sdf(self, vertices):

        # TODO: add transl and rotation sampling, and coordinate system synchronization.

        if not hasattr(self, 'sdf_torch'):
            self.sdf_torch = torch.from_numpy(self.sdf_dict['grid']).to(dtype=torch.float32).squeeze().unsqueeze(0).unsqueeze(0) # 1x1xDxDxD
        sdf_grids = self.sdf_torch.to(vertices.device)
        sdf_centroid = torch.tensor(self.sdf_dict['centroid']).reshape(1, 1, 3).to(device=vertices.device, dtype=torch.float32)

        # vertices = torch.tensor(vertices).reshape(1, -1, 3)
        batch_size, num_vertices, _ = vertices.shape
        vertices = ((vertices - sdf_centroid) / self.sdf_dict['scale']) # scale to [-1,1]
        sdf_values = F.grid_sample(sdf_grids,
                                       vertices[:, :, [2, 1, 0]].view(-1, num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                       padding_mode='border',
                                       align_corners=True
                                       # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                    )

        '''
        # illustration of grid_sample dimension order, assume first dimension to be innermost
        # import torch
        # import torch.nn.functional as F
        # import numpy as np
        # sz = 5
        # input_arr = torch.from_numpy(np.arange(sz * sz).reshape(1, 1, sz, sz)).float()
        # indices = torch.from_numpy(np.array([-1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -1, 0.5, 0.5, -1]).reshape(1, 1, 7, 2)).float()
        # out = F.grid_sample(input_arr, indices, align_corners=True)
        # print(input_arr)
        # print(out)
        '''
        return sdf_values.reshape(batch_size, num_vertices)


if '__name__' == '__main__':

    save_dir = './SAMP_obj_bps'

    if not os.path.exists(os.path.join(save_dir, 'bps_set.npy')):
        bps_set = bps_gen_ball_inside(n_bps=1000, random_seed=100)
        with open(os.path.join(save_dir, 'bps_set.npy'), 'wb') as f:
            np.save(f, bps_set)
    else:
        with open(os.path.join(save_dir, 'bps_set.npy'), 'rb') as f:
            bps_set = np.load(f)
    
    # Example usage
    point_set_A = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    point_set_B = torch.tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])

    # Reshape the point sets to (1, num_points, 3) as pytorch3d's chamfer_distance expects batched inputs
    point_set_A = point_set_A.unsqueeze(0)
    point_set_B = point_set_B.unsqueeze(0)

    # Use pytorch3d's chamfer_distance function
    # chamfer_dist, _, nearest_indices_A_to_B, _, _ = chamfer_distance(point_set_A, point_set_B)

    # print("Chamfer Distance:", chamfer_dist.item())
    # print("Nearest Indices A to B:", nearest_indices_A_to_B[0])


    # Compute the nearest indices using torch.cdist and torch.argmin
    distances_A_to_B = torch.cdist(point_set_A[0], point_set_B[0])
    distances_B_to_A = torch.cdist(point_set_B[0], point_set_A[0])
    nearest_indices_A_to_B = torch.argmin(distances_A_to_B, dim=1)
    nearest_indices_B_to_A = torch.argmin(distances_B_to_A, dim=1)

    # print("Chamfer Distance:", chamfer_dist.item())
    print("Nearest Indices A to B:", nearest_indices_A_to_B)
    print("Nearest Indices B to A:", nearest_indices_B_to_A)