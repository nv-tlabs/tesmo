import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import utils.geometry_utils as GeoUtils
from scipy.ndimage.morphology import distance_transform_edt
from pytorch3d.loss.chamfer import chamfer_distance
from data_loaders.humanml.utils.utils_samp import canonicalize_poses_to_object_space_th
from data_loaders.humanml.utils.utils_samp import visualize_interactive_scene
from model.scene_condition import query_feature_grid_3D
from data_loaders.humanml.utils.utils_samp import save_sample_poses
from utils.evaluate_hoi import SampleBodySurface
# # Check if CUDA is available
# if torch.cuda.is_available():
#     # Get the currently running CUDA device
#     device = torch.cuda.current_device()
#     print(f"Currently running on CUDA device: {device}")
# else:
#     print("CUDA is not available.")

class StartGoalPosition:
    # the length of the trajectory is not fixed. 
    def __init__(self,  
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                #  classifiler_scale=0.1,
                #  classifiler_scale=2000, 
                 classifiler_scale=3000, 
                #  classifiler_scale=2000, 
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 motion_length_cut=None,
                 print_every=None,
                 only_xz=False,
                 with_y=False,
                 with_orient=False,
                 orient_cosine_distance=False,
                 classifier_scale_orient=0.01, # scale for orient only.
                 with_decay=False,
                 decay_step=50,
                 decay_scale=0.1,
                 only_end=False,
                 ):
        
        # import pdb;pdb.set_trace()
        self.target = target # input motion.
        # self.target_mask = target_mask
        # self.transform = transform
        # self.inv_transform = inv_transform
        # self.abs_3d = abs_3d
        self.classifiler_scale = classifiler_scale
        # self.reward_model = reward_model
        # self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        # self.use_rand_projection = use_rand_projection
        
        self.motion_length_cut = motion_length_cut # is a list;
        # self.cut_frame = int(self.motion_length_cut * 20)
        self.print_every = print_every
        
        self.n_joints = 22
        # NOTE: optimizing to the whole trajectory is not good.
        self.gt_style = 'target'  # 'inpainting_motion'
        
        self.only_xz = only_xz
        self.with_y = with_y
        self.with_orient = with_orient
        self.orient_cosine_distance = orient_cosine_distance

        self.classifier_scale_orient = classifier_scale_orient
        self.only_end = only_end

        self.with_decay = with_decay
        self.decay_step = decay_step
        self.decay_scale = decay_scale

    def __call__(self, x, t, p_mean_var, y=None,): # *args, **kwds):
        print('debug: ******* \n')
        print('self.use_mse_loss: ', self.use_mse_loss, 'self.only_xz: ', self.only_xz, 
            'self.with_y: ', self.with_y, 'self.only_end: ', self.only_end, 'self.with_orient: ', self.with_orient, 
            'orient_cosine_distance: ', self.orient_cosine_distance, 'classifier_scale_orient:', self.classifier_scale_orient)

        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        # Stop condition
        
        # import pdb;pdb.set_trace()
        
        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)
        assert y is not None
        # x shape [bs, 263, 1, 120]
        with torch.enable_grad():
            if self.gt_style == 'target':
                if self.guidance_style == 'xstart': # this is us.
                    xstart_in = p_mean_var['pred_xstart'] # cleaned x0
                elif self.guidance_style == 'eps':
                    # using epsilon style guidance
                    assert self.reward_model is None, "there is no need for the reward model in this case"
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
                
                trajec = xstart_in.requires_grad_(True) # bs, dimension, 1, num_frames
                
                # TODO: given any start position and output position: first thing need to normlize
                loss_sum = []
                loss_ori_sum = []
                for b_i in range(trajec.shape[0]):
                    
                    cut_frame = self.motion_length_cut[b_i]
                    
                    if self.use_mse_loss or True:
                        loss_fn = F.mse_loss
                        flag = True    
                    else:
                        loss_fn = F.l1_loss
                        flag = False
                        
                    if self.only_xz and not self.with_y:
                        loss_sum_start = loss_fn(trajec[b_i, :2, 0, 0] , 
                                                    self.target[b_i, :2, 0, 0].detach(), reduction='none')
                        loss_sum_end = loss_fn(trajec[b_i, :2, 0, cut_frame-1] , 
                                        self.target[b_i, :2, 0, cut_frame-1].detach(),reduction='none')
                    elif self.only_xz and self.with_y: # ! This is what we used.
                        idx = [0,1,4]
                        loss_sum_start = loss_fn(trajec[b_i, idx, 0, 0] , 
                                                    self.target[b_i, idx, 0, 0].detach(), reduction='none')
                        loss_sum_end = loss_fn(trajec[b_i, idx, 0, cut_frame-1] , 
                                        self.target[b_i, idx, 0, cut_frame-1].detach(),reduction='none')
                    else:
                        loss_sum_start = loss_fn(trajec[b_i, :, 0, 0] , 
                                                    self.target[b_i, :, 0, 0].detach(), reduction='none')
                        loss_sum_end = loss_fn(trajec[b_i, :, 0, cut_frame-1] , 
                                        self.target[b_i, :, 0, cut_frame-1].detach(),reduction='none')
                    
                    if self.with_orient: # this needs to be used for MSE, different from L1 loss.
                        idx_ori=[2,3]
                        
                        # import pdb;pdb.set_trace()
                        if self.orient_cosine_distance:
                            def cosine_distance(theta1, theta2):
                                # Reshape the input tensors if needed
                                theta1_tensor = theta1.view(1, -1)
                                theta2_tensor = theta2.view(1, -1)
                                
                                # Calculate the cosine distance
                                distance = 1 - F.cosine_similarity(theta1_tensor, theta2_tensor)
                                
                                return distance 

                            loss_ori_start = cosine_distance(trajec[b_i, idx_ori, 0, 0], self.target[b_i, idx_ori, 0, 0].detach())
                            loss_ori_end = cosine_distance(trajec[b_i, idx_ori, 0, cut_frame-1], self.target[b_i, idx_ori, 0, cut_frame-1].detach())
                            # loss_ori_start = loss_fn(trajec[b_i, idx_ori, 0, 0], self.target[b_i, idx_ori, 0, 0].detach(), reduction='none')
                            # loss_ori_end = loss_fn(trajec[b_i, idx_ori, 0, cut_frame-1], self.target[b_i, idx_ori, 0, cut_frame-1].detach(), reduction='none')
                        else:
                            loss_ori_start = loss_fn(trajec[b_i, idx_ori, 0, 0], self.target[b_i, idx_ori, 0, 0].detach(), reduction='none')    
                            loss_ori_end = loss_fn(trajec[b_i, idx_ori, 0, cut_frame-1], self.target[b_i, idx_ori, 0, cut_frame-1].detach(), reduction='none')    

                        loss_ori_sum.append((loss_ori_start.mean()+loss_ori_end.mean())/2)
                        # loss_ori_sum.append((loss_ori_start+loss_ori_end)/2) # this actually works for L1 loss, but the top is not for MSE loss.
                    
                    # TODO: check the value of the GT, and the prediction. whether they are normalized.
                    if self.only_end:
                        loss_sum.append(loss_sum_end.mean())
                    else:
                        loss_sum.append((loss_sum_start.mean()+loss_sum_end.mean())/2)
        
                # import pdb;pdb.set_trace()
                loss_sum = torch.stack(loss_sum)
                # Scale the loss up so that we get the same gradient as if each sample is computed individually
                loss_sum = loss_sum.mean()
                
                if self.with_orient:
                    loss_ori_sum = torch.stack(loss_ori_sum).mean()
                else:
                    loss_ori_sum = torch.zeros(1)

            else:   
                raise NotImplementedError()

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: %f, orient: %f" % (int(t[0]), float(loss_sum), float(loss_ori_sum)))

            # import pdb;pdb.set_trace()
            
            print(f'time step: {t[0]}, scale: {self.classifiler_scale}, inference guidance loss: {loss_sum.item()}, orient: {loss_ori_sum.item()}')
            # return -loss_sum * self.classifiler_scale
            
            # import pdb;pdb.set_trace()
            # retain_graph=True for Trace mode.
            # torch.autograd.set_detect_anomaly(True)

            grad = torch.autograd.grad(-loss_sum, x, retain_graph=True)[0] # x is noised input x.
            
            if self.with_orient:
                grad_ori = torch.autograd.grad(-loss_ori_sum, x, retain_graph=True)[0]
                return grad * self.classifiler_scale + grad_ori * self.classifier_scale_orient
            
            if self.with_decay and t[0] < self.decay_step: # work on this decay at first.
                print('decay scale: ', self.decay_scale, f'decay step: {t[0]}', self.decay_step)
                grad = grad * self.decay_scale #(1 - t / 1000)
                
            return grad * self.classifiler_scale


class MapCollisionLoss:
    # the length of the trajectory is not fixed. 
    '''
    Agents should not go offroad.
    NOTE: this currently depends on the map that's also passed into the network.
            if the network map viewport is small and the future horizon is long enough,
            it may go outside the range of the map and then this is really inaccurate.
    '''
    
    def __init__(self,  
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 classifiler_scale=3000, 
                 guidance_style='xstart',
                 gt_style='target',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 motion_length_cut=None,
                 print_every=None,
                 
                 # how to represent a person.
                #  num_points_lw=(10, 6),
                num_points_lw = (6, 4),
                 device='cuda',
                 z_positive_down=True,
                 return_loss=False,
                 norm_scale=None,
                 need_inv_transform=True,
                 save_vis=False,
                 save_dir='debug_results/'
                 ):



        self.z_positive_down = z_positive_down # positive z axis is facing down on the 2D floor map.
        if norm_scale is not None:
            self.norm_scale = norm_scale
        else:
            self.norm_scale = 256 / (6.4*2)
        self.need_inv_transform = need_inv_transform
        self.return_loss = return_loss
        self.save_vis = save_vis
        self.save_dir = save_dir 
        if self.save_vis and self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            print('******** debug collision loss in inference guidance: ', self.save_dir)


        if target.max() == 255.0:
            self.target = 1.0 - target * 1.0 / 255.0 # 0 is walkable space (free space), 1 is occupied.
        else:
            # self.target = 1.0 - target * 1.0 / 255.0
            self.target = 1.0 - target 

        # import pdb;pdb.set_trace()
        # # input mask. The free space is 0, while objects locates at 1.
        if torch.is_tensor(self.target):
            self.target = self.target.to(device).float() # B, 1, H, W;
        else:
            # ! only works for one image.
            H, W = self.target.shape[-2], self.target.shape[-1]
            self.target = torch.from_numpy(self.target).reshape(-1, 1, H, W).to(device).float() # 1, 256, 256

        kernel_size=7
        power=0.5
        def compute_edge(x):
            return F.max_pool2d(x, kernel_size, 1, kernel_size // 2) - x
        
        
        # gt_edge = compute_edge(self.target).cpu().numpy()
        # self.edt = torch.tensor(distance_transform_edt(1 - (gt_edge > 0)) ** (power * 2), dtype=torch.float).to(device)
        # self.edt = self.edt * self.target
        one_distance_map = []
        for i in range(self.target.shape[0]):
            one_distance_map.append(torch.tensor(distance_transform_edt(self.target[i:i+1].cpu().numpy()) ** (power * 2), dtype=torch.float).to(device))

        # TODO: add distance truncation to avoid the gradient on the partial region of the objects.
        self.edt = torch.cat(one_distance_map)
        
        # TODO: add visualization; This is some bugs there !!!
        if self.save_vis:
            for tmp_i in range(self.edt.shape[0]):
                # visualize distance map and silhouette maps
                plt.figure(figsize=(10, 10))
                plt.subplot(2, 1, 1) 
                plt.imshow(self.edt[tmp_i].squeeze().detach().cpu().numpy(), cmap='hot')
                plt.colorbar()

                plt.subplot(2, 1, 2)
                plt.imshow(self.target[tmp_i].squeeze().detach().cpu().numpy(), cmap='viridis')
                plt.colorbar()
                plt.savefig(f'{self.save_dir}/{tmp_i}_collision_gt_maps.png')
                plt.close('all')
        # import pdb;pdb.set_trace()
        
        self.gt_style = gt_style
        
        self.classifiler_scale = classifiler_scale
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.motion_length_cut = motion_length_cut # is a list;
        self.print_every = print_every    
        self.n_joints = 22
        self.inv_transform = inv_transform
        '''
        - num_points_lw : how many points will be sampled within each agent bounding box
                            to detect map collisions. e.g. (15, 10) will sample a 15 x 10 grid
                            of points where 15 is along the length and 10 along the width.
        '''
        self.num_points_lw = num_points_lw
        # lwise = torch.linspace(-0.3, 0.3, self.num_points_lw[0]) # x axis
        # wwise = torch.linspace(-0.15, 0.15, self.num_points_lw[1]) # z axis

        # almost the point.
        lwise = torch.linspace(-0.03, 0.03, self.num_points_lw[0]) # x axis
        wwise = torch.linspace(-0.015, 0.015, self.num_points_lw[1]) # z axis
        

        self.local_coords = torch.cartesian_prod(lwise, wwise)

    def gen_agt_coords(self, pos, yaw, lw, raster_from_agent=None):
        '''
        - pos : B x 2
        - yaw : B x 1
        - lw : B x 2 (length, width in real world coordinate system)
        '''
        B = pos.size(0)
        cur_loc_coords = self.local_coords.to(pos.device).unsqueeze(0).expand((B, -1, -1))
        # scale by the extents
        cur_loc_coords = cur_loc_coords.unsqueeze(-3) # added.

        # import pdb;pdb.set_trace()
        # transform initial coords to given pos, yaw
        s_yaw = torch.sin(yaw).unsqueeze(-1)
        c_yaw = torch.cos(yaw).unsqueeze(-1)
        rotM = torch.cat((torch.cat((c_yaw, s_yaw), dim=-1), torch.cat((-s_yaw, c_yaw), dim=-1)), dim=-2)
        agt_coords_agent_frame = cur_loc_coords.float() @ rotM + pos.unsqueeze(-2)
        
        # import pdb;pdb.set_trace()
        agt_coords_agent_frame *= lw.unsqueeze(-2)
        # then transform to raster frame
        # agt_coords_raster_frame = GeoUtils.transform_points_tensor(agt_coords_agent_frame, raster_from_agent)

        return agt_coords_agent_frame
    
    def gen_agt_bbox(self, pos, yaw): # ! not used.
        '''
        - pos : B x 2
        - yaw : B x 1
        - lw : B x 2 (length, width in real world coordinate system)
        '''
        B = pos.size(0)
        # scale by the extents
        wh = torch.Tensor([0.6, 0.3]).to(pos.device).unsqueeze(0).expand((B, -1)).requires_grad_(True) # body width, length
        
        bbox = torch.cat([torch.zeros((B, 1)).to(pos.device).requires_grad_(True), pos, wh, yaw], -1)
        
        return bbox
    
    def __call__(self, x, t, p_mean_var, y=None, idx=None): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        
        # norm_scale = 256 / 6.4
        # get the trajectory in absolute coordinates.
        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)

        # assert y is not None
        
        # x shape [bs, 263, 1, 120]
        with torch.enable_grad():
            if self.gt_style == 'target':
                if self.guidance_style == 'xstart': # this is us.
                    xstart_in = p_mean_var['pred_xstart'] # cleaned x0
                elif self.guidance_style == 'eps':
                    # using epsilon style guidance
                    assert self.reward_model is None, "there is no need for the reward model in this case"
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
                
                # import pdb;pdb.set_trace()
                xstart_in = xstart_in.requires_grad_(True)
                
                if self.need_inv_transform:
                    x_in_pose_space = self.inv_transform(
                        xstart_in.permute(0, 2, 3, 1),
                    ) # b, n, frame, 1
                else:
                    assert False, 'not implemented yet.'
                    x_in_pose_space = xstart_in # recover the pose space.

                B, N, T, _ = x_in_pose_space.shape
                
                # x_in_joints = recover_from_ric(x_in_pose_space, self.n_joints,
                #                         abs_3d=self.abs_3d)  
                
                x_in_pose_space = x_in_pose_space.requires_grad_(True) # bs, dimension, 1, num_frames
                pos_pred, yaw_pred_cos, yaw_pred_sin = x_in_pose_space[:, 0, :, :2], x_in_pose_space[:, 0, :, 2:3], x_in_pose_space[:, 0, :, 3:4]
                yaw_pred = torch.atan2(yaw_pred_sin, yaw_pred_cos)
                
            
                ### * calculate the distance transform loss.
                # import pdb;pdb.set_trace()
                # get points in each agent trajectory.
                # this is centered at the center of the image.
                agt_samp_pts = self.gen_agt_coords(pos_pred, yaw_pred, torch.from_numpy(np.array(( self.norm_scale, self.norm_scale))).to(pos_pred.device)) 
                # agt_samp_pts = agt_samp_pts.reshape((B, N, T, -1, 2)).long().detach() + torch.Tensor([25, 200]).cuda()
                
                norm_h, norm_w = self.edt.shape[-2], self.edt.shape[-1]
                agt_samp_pts[..., 0] /= norm_w / 2 # real size
                agt_samp_pts[..., 1] /= norm_h / 2 # real size
                
                if not self.z_positive_down: # in model it self, this is not needed. 
                    agt_samp_pts[..., 1] *= -1

                if idx == None:
                    distance_loss = F.grid_sample(self.edt, agt_samp_pts, mode='bilinear', align_corners=True) # sampling-> gradient
                    silhouette_loss = F.grid_sample(self.target, agt_samp_pts, mode='bilinear', align_corners=True)
                else: # the 
                    distance_loss = F.grid_sample(self.edt[idx:idx+1], agt_samp_pts, mode='bilinear', align_corners=True) # sampling-> gradient    
                    # get the silhouette loss.
                    silhouette_loss = F.grid_sample(self.target[idx:idx+1], agt_samp_pts, mode='bilinear', align_corners=True) # 

                
                if (t[0] == 999 or t[0] % 50 == 0) and self.save_vis:
                    if idx is not None:
                        nidx = idx
                    else:
                        nidx = 0
                    
                    # visualize the 2D points itself.
                    plt.figure(figsize=(10, 10))
                    plt.scatter(pos_pred[0, :, 0].detach().cpu().numpy(), pos_pred[0, :, 1].detach().cpu().numpy())
                    plt.savefig(f'{self.save_dir}/{nidx}_{t[0]}_points.png')
                    plt.close('all')
                    # visualize the 2D map

                    input_map = self.target[nidx:nidx+1, 0]
                    plt.figure(figsize=(10, 10))
                    plt.imshow(input_map.cpu().numpy().squeeze())
                    plt.show()
                    # plt.savefig(f'{self.save_dir}/{nidx}_{t[0]}_input_map.png')
                    # print(f'{self.save_dir}/{nidx}_{t[0]}_input_map.png')

                    x_coords = agt_samp_pts[0, :, :, 0].reshape(-1).detach().cpu().numpy() * norm_w / 2
                    y_coords = agt_samp_pts[0, :, :, 1].reshape(-1).detach().cpu().numpy() * norm_h / 2

                    # Create the scatter plot
                    plt.scatter(x_coords+128, y_coords+128)
                    plt.savefig(f'{self.save_dir}/{nidx}_{t[0]}_input_map_points.png')

                    print(f'{self.save_dir}/{nidx}_{t[0]}_input_map_points.png')
                    plt.close('all')

                if idx is not None:
                    cut_frame = self.motion_length_cut[idx] 
                    distance_loss_sum = distance_loss[:, :, :cut_frame].mean() / self.norm_scale # average distance in meters.
                    sil_loss_sum = silhouette_loss[:, :, :cut_frame].mean()
                    

                else: # ! batch-wise
                    distance_loss_sum = 0.0
                    sil_loss_sum = 0.0
                    for i in range(distance_loss.shape[0]):
                        cut_frame = self.motion_length_cut[i] 
                        distance_loss_sum += distance_loss[i, :, :, :cut_frame].mean() / self.norm_scale
                        sil_loss_sum += silhouette_loss[i, :, :, :cut_frame].mean()
                    
                    distance_loss_sum /= distance_loss.shape[0]
                    sil_loss_sum /= silhouette_loss.shape[0]

            else:   
                raise NotImplementedError()

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: dis: %f, sil: %f" % (int(t[0]), float(distance_loss_sum), float(sil_loss_sum)))

            
            print(f'time step: {t[0]}, scale: {self.classifiler_scale}, inference distance loss: {distance_loss_sum.item()}, sil: {sil_loss_sum.item()}')
            
            # loss_sum = sil_loss_sum # this is useful.
            loss_sum = distance_loss_sum

            if not self.return_loss:
                # grad = torch.autograd.grad(-loss_sum, x, allow_unused=True, retain_graph=True)[0] # x is noised input x.
                grad = torch.autograd.grad(-loss_sum, x)[0] # x is noised input x.
                # import pdb;pdb.set_trace()
                return grad * self.classifiler_scale
            else:
                return {
                    'distance_loss_sum': distance_loss_sum,
                    'sil_loss_sum': sil_loss_sum,
                }


def load_condition(model_kwargs, args, input_motions, can_maps=None, device=None, data=None, extra_sample_motion=None):

    ### add different guidance.
    
    # combination of different condition
    cond_fn = []
    if args.guidance_mode != '' and args.guidance_mode is not None:
        guidance_mode = args.guidance_mode.split(',')

        for one in guidance_mode:
            print(f'add {one} in guidance mode.')
            if one == 'start_goal':
                # import pdb;pdb.set_trace()
                print('add StartGoal')
                print('_______________\n')
                print('only_xz: ', args.only_xz)
                print('with_y: ', args.with_y)
                print('with_orient: ', args.with_orient)
                print('orient_cosine_distance: ', args.orient_cosine_distance)
                print('classifier_scale_orient: ', args.classifier_scale_orient)
                print('use mse loss: ', args.gen_mse_loss)
                print('only end: ', args.only_end)
                print('_______________\n')
                cond_fn.append(
                    StartGoalPosition(target=input_motions, 
                            motion_length_cut=model_kwargs['y']['lengths'],
                            only_xz=args.only_xz,
                            classifiler_scale=args.classifier_scale,
                            with_y=args.with_y,
                            with_orient=args.with_orient,
                            orient_cosine_distance=args.orient_cosine_distance,
                            classifier_scale_orient=args.classifier_scale_orient,
                            with_decay=args.with_decay,
                            decay_step=args.decay_step,
                            decay_scale=args.decay_scale,
                            use_mse_loss=args.gen_mse_loss,
                            only_end=args.only_end, 
                            )
                )
            if one == 'floor_map_2d':
                print('add MapCollisionLoss')
                cond_fn.append(
                    MapCollisionLoss(
                        target=can_maps, 
                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                        motion_length_cut=model_kwargs['y']['lengths'],
                        classifiler_scale=args.classifier_scale_2d_maps,
                        device=device,
                        z_positive_down=False, # only for inference guidance loss. the positive z axis is facing down for the feature maps;
                        # return_loss=False,
                        norm_scale=256 / (6.4*2), 
                        need_inv_transform=True, # do not need to inverse transform the input motion.
                        save_vis=False,
                        save_dir='debug_results/collision_guidance',
                    )
                )
            if one == 'dense_trajectory':
                cond_fn.append(
                    CDTrajectoryLoss(
                    target=extra_sample_motion, 
                    inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                    motion_length_cut=model_kwargs['y']['lengths'],
                    classifiler_scale=args.classifier_scale_CD_trajectory,
                    device=device,
                    )
            )
            
            if one == 'obj_collision':
                cond_fn.append(
                    CollisionObjectLoss(
                    target_dict=model_kwargs['y'],
                    root_representation=args.root_representation,
                    inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                    motion_length_cut=model_kwargs['y']['lengths'],
                    classifiler_scale=args.classifier_scale_object_collision,
                    device=device,
                    need_inv_transform=True, # do not need to inverse transform the input motion.
                    save_vis=True,
                    save_dir='debug_results/collision_guidance_object',
                    )
                )
            if one == 'kps':
                pass
                raise NotImplementedError()
    
    if len(cond_fn) == 0:
        return None
    else:
        return cond_fn