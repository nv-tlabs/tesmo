import sys
sys.path.append('/home/hyi/data/workspace/motion_generation/priorMDM')


import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from data_loaders import humanml_utils
# import humanml_utils
import copy

MAX_LINE_LENGTH = 20
PLOT_END_ROOT_ORIENTATION=False # this is used for debugging the no pose in the groundtruth.

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

    
def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, ori_motion_param=None, not_show_video=False, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], handshake_size=0, blend_size=0, step_sizes=[], lengths = [], joints2=None, painting_features=[],
                   input_motion=None, floor_maps=None):
    
    matplotlib.use('Agg')
    """
    A wrapper around explicit_plot_3d_motion that 
    uses gt_frames to determine the colors of the frames
    """
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = ['blue' if index in gt_frames else 'orange' for index in range(frames_number)]

    if vis_mode == 'unfold':
        frame_colors = ['purple'] *handshake_size + ['blue']*blend_size + ['orange'] *(120-handshake_size*2-blend_size*2) +['orange']*blend_size
        frame_colors = ['orange'] *(120-handshake_size-blend_size) + ['orange']*blend_size + frame_colors*1024
    elif vis_mode == 'unfold_arb_len':
        for ii, step_size in enumerate(step_sizes):
            if ii == 0:
                frame_colors = ['orange']*(step_size - handshake_size - blend_size) + ['orange']*blend_size + ['purple'] * (handshake_size//2)
                continue
            if ii == len(step_sizes)-1:
                frame_colors += ['purple'] * (handshake_size//2) + ['orange'] * blend_size + ['orange'] * (lengths[ii] - handshake_size - blend_size)
                continue
            frame_colors += ['purple'] * (handshake_size // 2) + ['orange'] * blend_size + ['orange'] * (
                            lengths[ii] - 2 * handshake_size - 2 * blend_size) + ['orange'] * blend_size + \
                            ['purple'] * (handshake_size // 2)
    elif vis_mode == 'gt':
        frame_colors = ['blue'] * frames_number
        
    ## plot the overall trajectory and start pose and end pose
    
    # import pdb;pdb.set_trace()
    
    # explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=figsize, 
    #                         fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors, 
    #                         joints2=joints2, painting_features=painting_features, input_motion=input_motion, floor_image=floor_maps)
    
    if not not_show_video:
        ### need to open after 1106.
        explicit_plot_3d_motion_static_camera(save_path, kinematic_tree, joints, title, dataset, 
                                ori_motion_param=ori_motion_param, figsize=figsize, 
                                fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors, 
                                joints2=joints2, painting_features=painting_features, input_motion=input_motion, floor_image=floor_maps)
        
    if floor_maps is not None: # ! hack the results;
        explicit_plot_3d_image(save_path.replace(".mp4", "_noscene.png"), kinematic_tree, joints, title, dataset, \
                        ori_motion_param=ori_motion_param, figsize=figsize, 
                        fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors, 
                        joints2=joints2, painting_features=painting_features, input_motion=input_motion, floor_image=None)

        # this is used to match the maps and trajectory.
        joints[:, :, 2] = -1 * joints[:, :, 2]
        input_motion[:, :, 2] = -1 * input_motion[:, :, 2]
        
    explicit_plot_3d_image(save_path.replace(".mp4", ".png"), kinematic_tree, joints, title, dataset, \
                        ori_motion_param=ori_motion_param, figsize=figsize, 
                        fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors, 
                        joints2=joints2, painting_features=painting_features, input_motion=input_motion, floor_image=floor_maps)

# joint2 is for second person.
def explicit_plot_3d_image(save_path, kinematic_tree, joints, title, dataset, 
                        ori_motion_param=None, figsize=(3, 3), 
                        fps=120, radius=3, vis_mode="default", frame_colors=[], 
                        joints2=None, painting_features=[], input_motion=None, floor_image=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)


    # 2D image no need for scale.

    # preparation related to specific datasets
    # if dataset == "kit":
    #     data *= 0.003  # scale for visualization
    # elif dataset == "humanml":
    #     data *= 1.3  # scale for visualization
    #     if data2 is not None:
    #         data2 *= 1.3
    # elif dataset in ["humanact12", "uestc"]:
    #     data *= -1.5  # reverse axes, scale for visualization
    # elif dataset in ['humanact12', 'uestc', 'amass']:
    #     data *= -1.5 # reverse axes, scale for visualization
    # elif dataset =='babel':
    #     data *= -1.3


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
        
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]
    colors_upper_body = colors_blue[:2] + colors_orange[2:]
    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    # import pdb;pdb.set_trace()
    trajec = copy.deepcopy(data[:, 0])
    
    # import pdb;pdb.set_trace()
    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    ## scale the trajectory into image space.
    if floor_image is not None:
        # import pdb;pdb.set_trace()
        from matplotlib import image
        # floor = image.imread(floor_image)
        floor = floor_image.astype(np.uint8) * 255
        ax.imshow(floor, origin='lower',  cmap='gray')
        scale = 256 / (2 * 6.4)
        # scale = 256 / 6.4
        left_right_scale = 0.4
        trajec[..., [0,2]] = trajec[..., [0,2]] * scale + floor_image.shape[0] / 2
        data[..., [0,2]] = data[..., [0,2]] * scale + floor_image.shape[0] / 2
    else:
        left_right_scale = 1

    # Plot the trajectory
    # plot_xzPlane(MINS[0], MAXS[0], 0.0, MINS[2], MAXS[2])
    
    # Plot the start point and end point;
    def plot_start_end():
        # ax.scatter(trajec[[0], 0], np.zeros_like(trajec[[0], 1]), trajec[[0], 2], marker='o', linewidths=1.0) # (x,y,z)
        # ax.scatter(trajec[[-1], 0], np.zeros_like(trajec[[-1], 1]), trajec[[-1], 2], marker='x', linewidths=1.0) # (x,y,z)
        ax.scatter(trajec[[0], 0], trajec[[0], 2], marker='o', linewidths=1.0) # (x,y,z)
        ax.scatter(trajec[[-1], 0], trajec[[-1], 2], marker='x', linewidths=1.0) # (x,y,z)
        
        # import pdb;pdb.set_trace()
        # with orientation
        left_right_hip = (data[0, 1]-data[0, 2]) * 0.8 * left_right_scale
        orientation = np.array([[trajec[0,0], trajec[0,2]], [trajec[0, 0]-left_right_hip[2], trajec[0, 2]+left_right_hip[0]]])
        ax.plot(orientation[:, 0], orientation[:, 1], color='blue', linewidth=2.0)
        # ax.plot(orientation[:, 0], orientation[:, 1], 'bo') # (x,y,z)
        
        left_right_hip = (data[-1, 1]-data[-1, 2]) * 0.8 * left_right_scale
        orientation = np.array([[trajec[-1,0], trajec[-1,2]], [trajec[-1, 0]-left_right_hip[2], trajec[-1, 2]+left_right_hip[0]]])
        ax.plot(orientation[:, 0], orientation[:, 1], color='blue', linewidth=2.0)
        
    plot_start_end()
    
    if input_motion is not None: # plot start frame and end frame.
        # import pdb;pdb.set_trace()
        input_data = input_motion.copy().reshape(len(joints), -1, 3)
        # input_data *= 1.3 # no need scale.
        input_data[:, :, 1] -= height_offset
        input_traject = copy.deepcopy(input_data[:, 0])
        
        if floor_image is not None:
            scale = 256 / (2 * 6.4)
            input_traject[..., [0,2]] = input_traject[..., [0,2]] * scale + floor_image.shape[0] / 2
            input_data[..., [0,2]] = input_data[..., [0,2]] * scale + floor_image.shape[0] / 2
        
        # import pdb;pdb.set_trace()
        # ax.scatter(input_traject[[0], 0], input_traject[[0], 2], marker='o', linewidths=2) # (x,y,z)
        # ax.scatter(input_traject[[-1], 0], input_traject[[-1], 2], marker='x', linewidths=2) # (x,y,z)
        ax.plot(input_traject[[0], 0], input_traject[[0], 2], 'ro', linewidth=1.5) # (x,y,z)
        ax.plot(input_traject[[-1], 0], input_traject[[-1], 2], 'r*', linewidth=1.5) # (x,y,z)
        
        # with orientation
        left_right_hip = (input_data[0, 1]-input_data[0, 2]) * 1.0 * left_right_scale
        orientation = np.array([[input_traject[0,0], input_traject[0,2]], [input_traject[0, 0]-left_right_hip[2], input_traject[0, 2]+left_right_hip[0]]])
        ax.plot(orientation[:, 0], orientation[:, 1], color='red', linewidth=1.0)
        
        if PLOT_END_ROOT_ORIENTATION and ori_motion_param is not None:
            # import pdb;pdb.set_trace()
            ori_cos = ori_motion_param[-1, 2] # frames, dim;
            ori_sin = ori_motion_param[-1, 3]
            R = np.array([[ori_cos, -ori_sin], [ori_sin, ori_cos]]) 

            # print('new_orientation')
            left_right_hip = np.array([0, 1]) # x, y
            left_right_hip = np.matmul(R, left_right_hip) * 1.0 * 0.2 # this is clockwise rotation.
            orientation = np.array([[input_traject[-1,0], input_traject[-1,2]], [input_traject[-1, 0]+left_right_hip[0], input_traject[-1, 2]+left_right_hip[1]]])
            ax.plot(orientation[:, 0], orientation[:, 1], color='red', linewidth=1.0)

        else:
            left_right_hip = (input_data[-1, 1]-input_data[-1, 2]) * 1.0 * left_right_scale
            orientation = np.array([[input_traject[-1,0], input_traject[-1,2]], [input_traject[-1, 0]-left_right_hip[2], input_traject[-1, 2]+left_right_hip[0]]])
            ax.plot(orientation[:, 0], orientation[:, 1], color='red', linewidth=1.0)
        
        
    def plot_horizontal_trajectory():
        used_colors = colors_dict['orange']
        ax.plot(trajec[1:-1, 0], trajec[1:-1, 2], linewidth=1.0, color=used_colors[0])
        ax.plot(trajec[:2, 0], trajec[:2, 2], linewidth=1.0, color=colors_dict['purple'][0])
        ax.plot(trajec[-2:, 0], trajec[-2:, 2], linewidth=1.0, color=colors_dict['purple'][-1])
    
    plot_horizontal_trajectory() 

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def explicit_plot_3d_motion_static_camera(save_path, kinematic_tree, joints, title, dataset, 
                            ori_motion_param=None, figsize=(6, 6), 
                            fps=120, radius=3, vis_mode="default", frame_colors=[],
                            joints2=None, painting_features=[], input_motion=None, floor_image=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]
    
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.3, 0.3, 0.3, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == "humanml":
        data *= 1.3  # scale for visualization
        if data2 is not None:
            data2 *= 1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
    
    def init():
        ax.set_xlim3d([MINS[0]-0.2, MAXS[0]+0.2])
        # ax.set_xlim3d([-7, 7])
        ax.set_ylim3d([0, 2])
        ax.set_zlim3d([MINS[2]-0.2, MAXS[2]+0.2])
        
        # ax.set_zlim3d([-7, 7])
        print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)
    
    init()

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = copy.deepcopy(data[:, 0])
    
    
    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    if input_motion is not None:
        input_data = input_motion.copy().reshape(len(joints), -1, 3)
        input_data *= 1.3 
        input_data[:, :, 1] -= height_offset
        input_traject = copy.deepcopy(input_data[:, 0])

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=130, azim=-90)
        ax.dist = 10
        
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
            
            
        # ## scale the trajectory into image space.
        # if floor_image is not None:
        #     # import pdb;pdb.set_trace()
        #     from matplotlib import image
        #     # floor = image.imread(floor_image)
        #     floor = floor_image.astype(np.uint8) * 255
        #     ax.imshow(floor, origin='lower',  cmap='gray')
        #     scale = 256 / 6.4
        #     left_right_scale = 0.4
        #     trajec[..., [0,2]] = trajec[..., [0,2]] * scale + 256
        #     data[..., [0,2]] = data[..., [0,2]] * scale + 256
        # else:
        #     left_right_scale = 1
        # # Plot the trajectory
        # # plot_xzPlane(MINS[0], MAXS[0], 0.0, MINS[2], MAXS[2])
                    
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        # this is to draw the local pose
        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = used_colors  # colors_purple
        for i, (chain, color, other_color) in enumerate(zip(kinematic_tree, used_colors, other_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0]+trajec[[index], 0], data[index, chain, 1], data[index, chain, 2]+trajec[[index], 2], linewidth=linewidth, color=color)
            
        def plot_start_end():
            ax.plot3D(trajec[[0], 0], trajec[[0], 1], trajec[[0], 2] , marker='o', color='green') # (x,y,z)
            ax.plot3D(trajec[[-1], 0], trajec[[-1], 1], trajec[[-1], 2] , marker='x', color='green') # (x,y,z)
            
            ax.plot3D(trajec[[0], 0], 0, trajec[[0], 2] , marker='o', color='green') # (x,y,z)
            ax.plot3D(trajec[[-1], 0], 0, trajec[[-1], 2] , marker='x', color='green') # (x,y,z)
            
            # print orientation
            left_right_hip = (data[0, 1]-data[0, 2]) * 1.2
            orientation = np.array([[trajec[0,0], trajec[0, 1], trajec[0,2]], \
                [trajec[0, 0]-left_right_hip[2], trajec[0, 1], trajec[0, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            
            left_right_hip = (data[-1, 1]-data[-1, 2]) * 1.2
            orientation = np.array([[trajec[-1,0], trajec[-1, 1], trajec[-1,2]], 
                                    [trajec[-1, 0]-left_right_hip[2], trajec[-1, 1], trajec[-1, 2]+left_right_hip[0]]])
            ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            
        def plot_root_orientation():
            left_right_hip = (data[index, 1]-data[index, 2]) * 1.2
            orientation = np.array([[trajec[index,0], trajec[index, 1], trajec[index,2]], 
                                    [trajec[index, 0]-left_right_hip[2], trajec[index, 1], trajec[index, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='purple', linewidth=2.0)
            
            ax.plot3D(orientation[:, 0], np.zeros_like(orientation[:, 1]), orientation[:, 2], color='purple', linewidth=2.0)
            
            
        def plot_root_horizontal():
            ax.plot3D(trajec[:index+1, 0], np.zeros_like(trajec[:index+1, 1]), trajec[:index+1, 2] , linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index+1, 0], trajec[:index+1, 1], trajec[:index+1, 2] , linewidth=2.0,
                      color=used_colors[0])
        
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] ), linewidth=2.0,
                        color=used_colors[0])
        
        def plot_start_end_rec(): 
            # ax.scatter(input_traject[[0], 0], input_traject[[0], 1], input_traject[[0], 2], marker='o', linewidths=3) # (x,y,z)
            # ax.scatter(input_traject[[-1], 0], input_traject[[-1], 1], input_traject[[-1], 2], marker='x', linewidths=3) # (x,y,z)

            ax.plot3D(input_traject[[-1], 0], input_traject[[-1], 1], input_traject[[-1], 2], color='red', marker='x')
            ax.plot3D(input_traject[[0], 0], input_traject[[0], 1], input_traject[[0], 2], color='red', marker='o')#
            
            # print orientation
            left_right_hip = (input_data[0, 1]-input_data[0, 2]) * 1.2
            orientation = np.array([[input_traject[0,0], input_traject[0, 1], input_traject[0,2]], \
                [input_traject[0, 0]-left_right_hip[2], input_traject[0, 1], input_traject[0, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
            if PLOT_END_ROOT_ORIENTATION and ori_motion_param is not None:
                # import pdb;pdb.set_trace()
                ori_cos = ori_motion_param[-1, 2] # frames, dim;
                ori_sin = ori_motion_param[-1, 3]
                R = np.array([[ori_cos, -ori_sin], [ori_sin, ori_cos]]) 

                # print('new_orientation')
                left_right_hip = np.array([0, 1]) # x, y
                left_right_hip = np.matmul(R, left_right_hip) * 1.0 * 0.2 # this is clockwise rotation.
                orientation = np.array([[input_traject[-1,0], input_traject[-1, 1], input_traject[-1,2]], \
                    [input_traject[-1, 0]+left_right_hip[0], input_traject[-1, 1], input_traject[-1, 2]+left_right_hip[1]]])
                ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            else:    
                left_right_hip = (input_data[-1, 1]-input_data[-1, 2]) * 1.2
                orientation = np.array([[input_traject[-1,0], input_traject[-1, 1], input_traject[-1,2]], 
                                        [input_traject[-1, 0]-left_right_hip[2], input_traject[-1, 1], input_traject[-1, 2]+left_right_hip[0]]])
                ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
        plot_root_orientation()
        if 'root_horizontal' in painting_features or True: # TODO： default
            plot_root_horizontal()
            
        if 'end_pose' in painting_features or True:
            plot_start_end()
            if input_motion is not None:
                plot_start_end_rec()    
            
        if (vis_mode == 'gt' or 'root' in painting_features) and False:
            plot_root()
            
            
        for feat in painting_features:
            plot_feature(feat)
            
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close('all')



##### below is useless ######




def explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), 
                            fps=120, radius=3, vis_mode="default", frame_colors=[],
                            joints2=None, painting_features=[], input_motion=None, floor_image=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset == "humanml":
        data *= 1.3  # scale for visualization
        if data2 is not None:
            data2 *= 1.3
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization
    elif dataset in ['humanact12', 'uestc', 'amass']:
        data *= -1.5 # reverse axes, scale for visualization
    elif dataset =='babel':
        data *= -1.3

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    trajec = copy.deepcopy(data[:, 0])
    
    
    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    if input_motion is not None:
        input_data = input_motion.copy().reshape(len(joints), -1, 3)
        input_data *= 1.3 
        input_data[:, :, 1] -= height_offset
        input_traject = copy.deepcopy(input_data[:, 0])

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])


        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = used_colors  # colors_purple
        for i, (chain, color, other_color) in enumerate(zip(kinematic_tree, used_colors, other_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # print('kinematic tree')
            # print(data.shape)
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            if data2 is not None: # TODO print second person.
                ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=other_color)
        
        def plot_start_end():
            ax.plot3D(trajec[[0], 0] - trajec[index, 0], trajec[[0], 1], trajec[[0], 2] - trajec[index, 2], marker='o', color='green') # (x,y,z)
            ax.plot3D(trajec[[-1], 0] - trajec[index, 0], trajec[[-1], 1], trajec[[-1], 2] - trajec[index, 2], marker='x', color='green') # (x,y,z)
            
            # import pdb;pdb.set_trace()
            # print orientation
            left_right_hip = (data[0, 1]-data[0, 2]) * 1.2
            orientation = np.array([[trajec[0,0]- trajec[index, 0], trajec[0, 1], trajec[0,2]- trajec[index, 2]], \
                [trajec[0, 0]-left_right_hip[2]- trajec[index, 0], trajec[0, 1], trajec[0, 2]+left_right_hip[0]- trajec[index, 2]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
            left_right_hip = (data[-1, 1]-data[-1, 2]) * 1.2
            orientation = np.array([[trajec[-1,0]- trajec[index, 0], trajec[-1, 1], trajec[-1,2]- trajec[index, 2]], 
                                    [trajec[-1, 0]-left_right_hip[2]- trajec[index, 0], trajec[[-1], 1], trajec[-1, 2]+left_right_hip[0]- trajec[index, 2]]])
            ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
            # import pdb;pdb.set_trace()
            # left_right_hip = trajec[1]-trajec[2]
            # orientation = np.array([-left_right_hip[2], 0, left_right_hip[0]])
            # ax.plot(orientation[0]+0.2, orientation[1]+0.2, color='red', linewidth=2.0)
        
        def plot_root_orientation():
            left_right_hip = (data[index, 1]-data[index, 2]) * 1.2
            orientation = np.array([[trajec[index,0], trajec[index, 1], trajec[index,2]], [trajec[index, 0]-left_right_hip[2], trajec[index, 1], trajec[index, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0]- trajec[index, 0], orientation[:, 1], orientation[:, 2]- trajec[index, 2], color='purple', linewidth=2.0)
            
            ax.plot3D(orientation[:, 0]- trajec[index, 0], np.zeros_like(orientation[:, 1]), orientation[:, 2]- trajec[index, 2], color='purple', linewidth=2.0)
            
        def plot_root_horizontal():
            ax.plot3D(trajec[:index+1, 0] - trajec[index, 0], np.zeros_like(trajec[:index+1, 1]), trajec[:index+1, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index+1, 0] - trajec[index, 0], trajec[:index+1, 1], trajec[:index+1, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0] - trajec[index, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] - trajec[index, 2]), linewidth=2.0,
                        color=used_colors[0])
        
        def plot_start_end_rec():
            ax.scatter(input_traject[[0], 0]- trajec[index, 0], input_traject[[0], 1], input_traject[[0], 2]- trajec[index, 2], marker='o', linewidths=3) # (x,y,z)
            ax.scatter(input_traject[[-1], 0]- trajec[index, 0], input_traject[[-1], 1], input_traject[[-1], 2]- trajec[index, 2], marker='x', linewidths=3) # (x,y,z)

            # print orientation
            left_right_hip = (input_data[0, 1]-input_data[0, 2]) * 1.2
            orientation = np.array([[input_traject[0,0]- trajec[index, 0], input_traject[0, 1], input_traject[0,2]- trajec[index, 2]], \
                [input_traject[0, 0]-left_right_hip[2]- trajec[index, 0], input_traject[0, 1], input_traject[0, 2]+left_right_hip[0]- trajec[index, 2]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            
            left_right_hip = (input_data[-1, 1]-input_data[-1, 2]) * 1.2
            orientation = np.array([[input_traject[-1,0]- trajec[index, 0], input_traject[-1, 1], input_traject[-1,2]- trajec[index, 2]], 
                                    [input_traject[-1, 0]-left_right_hip[2]- trajec[index, 0], input_traject[[-1], 1], input_traject[-1, 2]+left_right_hip[0]- trajec[index, 2]]])
            ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            

        if 'root_horizontal' in painting_features or True: # TODO： default
            plot_root_horizontal()
            plot_root_orientation()
            
        if 'end_pose' in painting_features or True:
            plot_start_end()
            if input_motion is not None:
                plot_start_end_rec()    
            
        if vis_mode == 'gt' or 'root' in painting_features:
            plot_root()
            plot_root_orientation()
            
        for feat in painting_features:
            plot_feature(feat)
            
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close('all')

def explicit_plot_3d_image_SDF(joints, sdf_feat, save_path, kinematic_tree=None, figsize=(3, 3), 
                        fps=120, radius=3, vis_mode="default", frame_colors=[], 
                        painting_features=[], input_motion=None, floor_image=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    # import pdb;pdb.set_trace()
    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    # fig, ax0 = plt.subplots(1, 1)
    # get cmap for the color.
    # import pdb;pdb.set_trace()
    colors = np.linspace(0.05, 0.95, data.shape[0]) 
    # input dimension: bs, njoints, nfeats
    root_dist = sdf_feat[:, 0, 0] # sdf distance of the root joint; bs, nframe, nfeats, njoints;
    sorted_idx = np.argsort(root_dist)
    sorted_colors = colors[sorted_idx]

    # scatter = ax0.scatter(joints[:, 0, 0], joints[:, 0, 2], c=colors, cmap='plasma')
    # fig.colorbar(scatter, ax=ax0)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_path, 'root_distance.png'))



    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    trajec = copy.deepcopy(data[:, 0])

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    left_right_scale = 1

    def plot_start_end():
        # ax.scatter(trajec[[0], 0], np.zeros_like(trajec[[0], 1]), trajec[[0], 2], marker='o', linewidths=1.0) # (x,y,z)
        # ax.scatter(trajec[[-1], 0], np.zeros_like(trajec[[-1], 1]), trajec[[-1], 2], marker='x', linewidths=1.0) # (x,y,z)
        ax.scatter(trajec[[0], 0], trajec[[0], 2], marker='o', linewidths=1.0) # (x,y,z)
        ax.scatter(trajec[[-1], 0], trajec[[-1], 2], marker='x', linewidths=1.0) # (x,y,z)
        
        # import pdb;pdb.set_trace()
        # with orientation
        left_right_hip = (data[0, 1]-data[0, 2]) * 0.8 * left_right_scale
        orientation = np.array([[trajec[0,0], trajec[0,2]], [trajec[0, 0]-left_right_hip[2], trajec[0, 2]+left_right_hip[0]]])
        ax.plot(orientation[:, 0], orientation[:, 1], color='blue', linewidth=2.0)
        # ax.plot(orientation[:, 0], orientation[:, 1], 'bo') # (x,y,z)
        
        left_right_hip = (data[-1, 1]-data[-1, 2]) * 0.8 * left_right_scale
        orientation = np.array([[trajec[-1,0], trajec[-1,2]], [trajec[-1, 0]-left_right_hip[2], trajec[-1, 2]+left_right_hip[0]]])
        ax.plot(orientation[:, 0], orientation[:, 1], color='blue', linewidth=2.0)
        
    plot_start_end()
        
    def plot_horizontal_trajectory():
        sc = ax.scatter(trajec[:, 0], trajec[:, 2], linewidth=1.0, c=plt.cm.viridis(sorted_colors))
        return sc
    sc = plot_horizontal_trajectory() 

    cbar = plt.colorbar(sc, label='Sorted Values')
    ax.set_aspect('equal')
    plt.tight_layout()
    print('save to : ', save_path)
    plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()

def explicit_plot_3d_motion_static_camera_withSDF(joints, sdf_feat, save_path, kinematic_tree, dataset, figsize=(6, 6), 
                            fps=120, radius=3, vis_mode="default", frame_colors=[],
                            painting_features=[], input_motion=None, floor_image=None):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        # plot_colortable(mcolors.CSS4_COLORS)
        # plt.show()
        xz_plane.set_facecolor((0.3, 0.3, 0.3, 0.5))
        # import matplotlib.colors as mcolors
        # xz_plane.set_facecolor((mcolors.CSS4_COLORS['dimgrey'], 0.5)
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    def init():
        ax.set_xlim3d([MINS[0]-0.2, MAXS[0]+0.2])
        # ax.set_xlim3d([-7, 7])
        ax.set_ylim3d([0, 2])
        ax.set_zlim3d([MINS[2]-0.2, MAXS[2]+0.2])
        
        # ax.set_zlim3d([-7, 7])
        print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)
    
    init()

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    trajec = copy.deepcopy(data[:, 0])
    
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    if input_motion is not None:
        input_data = input_motion.copy().reshape(len(joints), -1, 3)
        input_data *= 1.3 
        input_data[:, :, 1] -= height_offset
        input_traject = copy.deepcopy(input_data[:, 0])

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=130, azim=-90)
        ax.dist = 10
        
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
            
            
        # ## scale the trajectory into image space.
        # if floor_image is not None:
        #     # import pdb;pdb.set_trace()
        #     from matplotlib import image
        #     # floor = image.imread(floor_image)
        #     floor = floor_image.astype(np.uint8) * 255
        #     ax.imshow(floor, origin='lower',  cmap='gray')
        #     scale = 256 / 6.4
        #     left_right_scale = 0.4
        #     trajec[..., [0,2]] = trajec[..., [0,2]] * scale + 256
        #     data[..., [0,2]] = data[..., [0,2]] * scale + 256
        # else:
        #     left_right_scale = 1
        # # Plot the trajectory
        # # plot_xzPlane(MINS[0], MAXS[0], 0.0, MINS[2], MAXS[2])
                    
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])


        # this is to draw the local pose
        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = used_colors  # colors_purple
        for i, (chain, color, other_color) in enumerate(zip(kinematic_tree, used_colors, other_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # print('kinematic tree')
            # print(data.shape)
            ax.plot3D(data[index, chain, 0]+trajec[[index], 0], data[index, chain, 1], data[index, chain, 2]+trajec[[index], 2], linewidth=linewidth, color=color)
            # if data2 is not None: # TODO print second person.
            #     ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=other_color)
        
        def plot_start_end():
            ax.plot3D(trajec[[0], 0], trajec[[0], 1], trajec[[0], 2] , marker='o', color='green') # (x,y,z)
            ax.plot3D(trajec[[-1], 0], trajec[[-1], 1], trajec[[-1], 2] , marker='x', color='green') # (x,y,z)
            
            ax.plot3D(trajec[[0], 0], 0, trajec[[0], 2] , marker='o', color='green') # (x,y,z)
            ax.plot3D(trajec[[-1], 0], 0, trajec[[-1], 2] , marker='x', color='green') # (x,y,z)
            
            
            # import pdb;pdb.set_trace()
            # print orientation
            left_right_hip = (data[0, 1]-data[0, 2]) * 1.2
            orientation = np.array([[trajec[0,0], trajec[0, 1], trajec[0,2]], \
                [trajec[0, 0]-left_right_hip[2], trajec[0, 1], trajec[0, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            
            left_right_hip = (data[-1, 1]-data[-1, 2]) * 1.2
            orientation = np.array([[trajec[-1,0], trajec[-1, 1], trajec[-1,2]], 
                                    [trajec[-1, 0]-left_right_hip[2], trajec[-1, 1], trajec[-1, 2]+left_right_hip[0]]])
            ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='green', linewidth=2.0)
            
            # import pdb;pdb.set_trace()
            # left_right_hip = trajec[1]-trajec[2]
            # orientation = np.array([-left_right_hip[2], 0, left_right_hip[0]])
            # ax.plot(orientation[0]+0.2, orientation[1]+0.2, color='red', linewidth=2.0)
        
        def plot_root_orientation():
            left_right_hip = (data[index, 1]-data[index, 2]) * 1.2
            orientation = np.array([[trajec[index,0], trajec[index, 1], trajec[index,2]], 
                                    [trajec[index, 0]-left_right_hip[2], trajec[index, 1], trajec[index, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='purple', linewidth=2.0)
            
            ax.plot3D(orientation[:, 0], np.zeros_like(orientation[:, 1]), orientation[:, 2], color='purple', linewidth=2.0)
            
            
        def plot_root_horizontal():
            ax.plot3D(trajec[:index+1, 0], np.zeros_like(trajec[:index+1, 1]), trajec[:index+1, 2] , linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index+1, 0], trajec[:index+1, 1], trajec[:index+1, 2] , linewidth=2.0,
                      color=used_colors[0])
        
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] ), linewidth=2.0,
                        color=used_colors[0])
        
        def plot_start_end_rec(): 
            # ax.scatter(input_traject[[0], 0], input_traject[[0], 1], input_traject[[0], 2], marker='o', linewidths=3) # (x,y,z)
            # ax.scatter(input_traject[[-1], 0], input_traject[[-1], 1], input_traject[[-1], 2], marker='x', linewidths=3) # (x,y,z)

            ax.plot3D(input_traject[[-1], 0], input_traject[[-1], 1], input_traject[[-1], 2], color='red', marker='x')
            ax.plot3D(input_traject[[0], 0], input_traject[[0], 1], input_traject[[0], 2], color='red', marker='o')#
            
            # print orientation
            left_right_hip = (input_data[0, 1]-input_data[0, 2]) * 1.2
            orientation = np.array([[input_traject[0,0], input_traject[0, 1], input_traject[0,2]], \
                [input_traject[0, 0]-left_right_hip[2], input_traject[0, 1], input_traject[0, 2]+left_right_hip[0]]])
            ax.plot3D(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
            left_right_hip = (input_data[-1, 1]-input_data[-1, 2]) * 1.2
            orientation = np.array([[input_traject[-1,0], input_traject[-1, 1], input_traject[-1,2]], 
                                    [input_traject[-1, 0]-left_right_hip[2], input_traject[-1, 1], input_traject[-1, 2]+left_right_hip[0]]])
            ax.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2], color='red', linewidth=2.0)
            
        plot_root_orientation()
        if 'root_horizontal' in painting_features or True: # TODO： default
            plot_root_horizontal()
            
            
        if 'end_pose' in painting_features or True:
            plot_start_end()
            
            if input_motion is not None:
                plot_start_end_rec()    
            
        if (vis_mode == 'gt' or 'root' in painting_features) and False:
            plot_root()
            
            
        for feat in painting_features:
            plot_feature(feat)
            
        plt.axis("off")
        ax.set_xticklabels(['x'])
        ax.set_yticklabels([])
        ax.set_zticklabels(['z'])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close('all')



if __name__ == '__main__':
    import data_loaders.humanml.utils.paramUtil as paramUtil
    # test the visualization.
    animation_save_path = 'debug_results/3d_motion.mp4'
    skeleton = paramUtil.t2m_kinematic_chain
    npy_path = 'debug_results/results.npy'
    motion = np.load(npy_path, allow_pickle=True).tolist()
    
    motion = motion['motion'][0].transpose(2, 0, 1)
    caption = ''
    
    plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                            dataset='humanml', fps=20, vis_mode='gt',
                            gt_frames=[], input_motion = motion)