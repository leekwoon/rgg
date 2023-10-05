import gym
import d4rl
import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sympy import ShapeError


def plot_maze2d(env_name, ax=None, fix_xy_lim=False):
    offset = 0.1

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    env = gym.make(env_name)
    background = env.maze_arr == 10
    ax.imshow(background, cmap='Greys')
    if fix_xy_lim:
        if env_name == 'maze2d-umaze-v1':
            ax.set_xlim(0.5 - offset, 3.5 + offset)
            ax.set_ylim(3.5 + offset, 0.5 - offset)
        elif env_name == 'maze2d-medium-v1':
            ax.set_xlim(0.5 - offset, 6.5 + offset)
            ax.set_ylim(6.5 + offset, 0.5 - offset)
        elif env_name == 'maze2d-large-v1':
            ax.set_xlim(0.5 - offset, 10.5 + offset)
            ax.set_ylim(7.5 + offset, 0.5 - offset)
        else:
            raise NotImplementedError


def plot_maze2d_observations(observations, ax=None, plot_start=True, goal=None, cmap=None):
    """
        observations [plan_hor x observation_dim]
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    if cmap is None:
        cmap = cm.Reds
    colors = cmap(np.linspace(0, 1, observations.shape[0]))
    colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
    cmap = ListedColormap(colors)  
    ax.scatter(
        observations[:, 1], observations[:, 0], 
        c=np.arange(observations.shape[0]), cmap=cmap, s=60
    )        
    if plot_start:
        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=100, edgecolors='black', color='black', alpha=1, linewidth=3
        )
    if goal is not None:
        ax.scatter(
            goal[1], goal[0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        ax.scatter(
            goal[1], goal[0], 
            marker='*', s=300, edgecolors='black', color='black', alpha=1
        )


def plot_maze2d_observations_with_attribution(observations, attribution, ax=None, plot_start=True, goal=None, cmap=None):
    """
        observations [plan_hor x observation_dim]
        attribution [plan_hor] or [plan_hor x transition_dim]
    """
    if len(attribution.shape) < 0 or len(attribution.shape) > 2:
        raise ShapeError
    if len(attribution.shape) == 2:
        attribution = attribution.sum(axis=1)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    if cmap is None:
        cmap = cm.bwr

    # edge
    ax.scatter(
        observations[:, 1], observations[:, 0], 
        c='white', s=60, edgecolor='k', linewidth=4
    )
    ax.scatter(
        observations[:, 1], observations[:, 0], 
        c=attribution, cmap=cmap, s=60, alpha=0.5
    )     

    if plot_start:
        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        ax.scatter(
            observations[0, 1], observations[0, 0], 
            marker='o', s=100, edgecolors='black', color='black', alpha=1, linewidth=3
        )
    if goal is not None:
        ax.scatter(
            goal[1], goal[0], 
            marker='o', s=500, edgecolors='black', color='white', alpha=1, linewidth=3
        )
        ax.scatter(
            goal[1], goal[0], 
            marker='*', s=300, edgecolors='black', color='black', alpha=1
        )


def plot_locomotion_observations(env_name, observations, ax=None, img_width=1024, img_height=512, skip_frame=1, edge_only=True):
    # references
    # 1. https://github.com/jannerm/diffuser/blob/main/diffuser/utils/rendering.py
    # 2. https://github.com/yusukeurakami/mujoco_2d_projection
    import cv2
    import mujoco_py as mjc
    from gym.envs.robotics.rotations import mat2euler
    import diffuser

    def env_map(env_name):
        '''
            map D4RL dataset names to custom fully-observed
            variants for rendering
        '''
        if 'halfcheetah' in env_name:
            return 'HalfCheetahFullObs-v2'
        elif 'hopper' in env_name:
            return 'HopperFullObs-v2'
        elif 'walker2d' in env_name:
            return 'Walker2dFullObs-v2'
        else:
            return env_name

    def get_2d_from_3d(
        obj_pos, cam_pos, cam_ori, width, height, fov=90
    ):
        """
        :param obj_pos: 3D coordinates of the joint from MuJoCo in nparray [m]
        :param cam_pos: 3D coordinates of the camera from MuJoCo in nparray [m]
        :param cam_ori: camera 3D rotation (Rotation order of x->y->z) from MuJoCo in nparray [rad]
        :param fov: field of view in integer [degree]
        :return: 2D coordinates of the obj [pixel].
        """

        e = np.array([height/2, width/2, 1])
        fov = np.array([fov])

        # Converting the MuJoCo coordinate into typical computer vision coordinate.
        cam_ori_cv = np.array([cam_ori[1], cam_ori[0], -cam_ori[2]])
        obj_pos_cv = np.array([obj_pos[1], obj_pos[0], -obj_pos[2]])
        cam_pos_cv = np.array([cam_pos[1], cam_pos[0], -cam_pos[2]])

        # obj_pos_in_2D, obj_pos_from_cam = get_2D_from_3D(obj_pos_cv, cam_pos_cv, cam_ori_cv, fov, e)
        # Get the vector from camera to object in global coordinate.
        ac_diff = obj_pos_cv - cam_pos_cv
        # Rotate the vector in to camera coordinate
        x_rot = np.array([[1 ,0, 0],
                        [0, np.cos(cam_ori_cv[0]), np.sin(cam_ori_cv[0])],
                        [0, -np.sin(cam_ori_cv[0]), np.cos(cam_ori_cv[0])]])

        y_rot = np.array([[np.cos(cam_ori_cv[1]) ,0, -np.sin(cam_ori_cv[1])],
                    [0, 1, 0],
                    [np.sin(cam_ori_cv[1]), 0, np.cos(cam_ori_cv[1])]])

        z_rot = np.array([[np.cos(cam_ori_cv[2]) ,np.sin(cam_ori_cv[2]), 0],
                    [-np.sin(cam_ori_cv[2]), np.cos(cam_ori_cv[2]), 0],
                    [0, 0, 1]])

        transform = z_rot.dot(y_rot.dot(x_rot))
        d = transform.dot(ac_diff)    

        # scaling of projection plane using fov
        fov_rad = np.deg2rad(fov)    
        e[2] *= e[0]*1/np.tan(fov_rad/2.0)

        # Projection from d to 2D
        bx = e[2]*d[0]/(d[2]) + e[0]
        by = e[2]*d[1]/(d[2]) + e[1]
        return bx, by

    def pad_observations(env, observations):
        qpos_dim = env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def set_state(env, state):
        qpos_dim = env.sim.data.qpos.size
        qvel_dim = env.sim.data.qvel.size
        assert(state.size == qpos_dim + qvel_dim)
        env.set_state(state[:qpos_dim], state[qpos_dim:])

    def get_image_mask(img):
        background = (img == 255).all(axis=-1, keepdims=True)
        mask = ~background.repeat(3, axis=-1)
        return mask

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    # === fixed parameters ===
    render_kwargs = {
        'trackbodyid': 2,
        'distance': 3,
        'lookat': [2, 0, 1],
        'elevation': 0
    }
    width = img_width
    height = img_height

    env = gym.make(env_map(env_name))
    viewer = mjc.MjRenderContextOffscreen(env.sim)

    for key, val in render_kwargs.items():
        if key == 'lookat':
            viewer.cam.lookat[:] = val[:]
        else:
            setattr(viewer.cam, key, val)

    cam_pos = render_kwargs['lookat'].copy()
    cam_pos[1] = -render_kwargs['distance']
    cam_ori = mat2euler(env.sim.data.get_camera_xmat('track'))
    fov = env.sim.model.cam_fovy[0]

    if 'hopper' in env_name:
        joints = ['torso', 'thigh', 'foot']
        joint_colors = ['Reds', 'Greens', 'Blues']
    elif 'walker2d' in env_name:
        joints = ['torso', 'thigh', 'foot']
        joint_colors = ['Reds', 'Greens', 'Blues']
    else:
        raise NotImplementedError()

    observations = pad_observations(env, observations) # compute x coordinate
    imgs = []
    joint_poses = {k: [] for k in joints}
    for observation in observations:
        set_state(env, observation)

        dim = (width, height)
        viewer.render(*dim)
        img = viewer.render(*dim)
        img = viewer.read_pixels(*dim, depth=False)
        img = img[::-1, :, :]
        img = np.ascontiguousarray(img, dtype=np.uint8)
        imgs.append(img)

        for k in joints:
            joint_poses[k].append(env.sim.data.get_body_xipos(k).copy())

    composite_img = np.ones_like(imgs[0]) * 255

    for k, jc in zip(joints, joint_colors):
        colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))
        colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
        for t, pos in enumerate(joint_poses[k]):
            x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
            composite_img = cv2.circle(
                composite_img,
                (int(y), height - int(x)),
                width // 150, # radius
                colors[t][:3] * 255,
                -1 # fill
            )       

    for t in range(0, len(imgs), skip_frame):
        img = imgs[t]
        if edge_only:
            mask = get_image_mask(img)
            img[mask[:, :, 0], 0] = 255
            img[mask[:, :, 1], 1] = 196
            img[mask[:, :, 2], 2] = 131

            edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
            # edges = cv2.Canny(image=img, threshold1=180, threshold2=280)
            edge_mask = (edges == 255)[:, :, None].repeat(3, axis=-1)
            img[edge_mask] = 0

        mask = get_image_mask(img)
        composite_img[mask] = img[mask]

        for k, jc in zip(joints, joint_colors):
            colors = getattr(cm, jc)(np.linspace(0, 1, observations.shape[0]))
            colors[:observations.shape[0]//5] = colors[observations.shape[0]//5]
            pos = joint_poses[k][t]
            x, y = get_2d_from_3d(pos, cam_pos, cam_ori, width, height, fov)
            composite_img = cv2.circle(
                composite_img,
                (int(y), height - int(x)),
                width // 100, # radius
                colors[t][:3] * 255,
                -1 # fill
            )   
            composite_img = cv2.circle(
                composite_img,
                (int(y), height - int(x)),
                width // 100, # radius
                (0, 0, 0),
                1 # not fill
            )   
    ax.imshow(composite_img)

