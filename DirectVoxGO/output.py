import os, sys, copy, glob, json, time, random, argparse
from tqdm import tqdm, trange
import sys
import mmcv
import imageio
import numpy as np
import torch

from lib import utils, dvgo, dcvgo, dmpigo


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
def load_blender_data(basedir):
    K, depths = None, None
    near_clip = None
    with open(os.path.join(basedir), 'r') as fp:
        meta = json.load(fp)
    all_poses = []
    poses = []
    img_name = []
    for frame in meta['frames'][::1]:

        img_name.append(frame['file_path'].split('/')[-1])
        poses.append(np.array(frame['transform_matrix']))
    poses = np.array(poses).astype(np.float32)

    all_poses.append(poses)

    poses = np.concatenate(all_poses, 0)

    H, W =800,800
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    near, far = 2., 6.

    H, W = int(H), int(W)
    hwf = [H, W, focal]


    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[..., :4]

    data_dict = dict(
        img_name=img_name,
        hwf=hwf, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        poses=poses, render_poses=render_poses,
        depths=depths,
    )
    return data_dict


@torch.no_grad()
def render_viewpoints(model, render_poses, Ks, ndc, render_kwargs,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      ):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    # assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        # HW = np.copy(HW)
        Ks = np.copy(Ks)
        # HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []


    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = 800,800
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)


def load_everything(json_path):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_blender_data(json_path)

    # remove useless field
    kept_keys = {
            'hwf', 'Ks', 'near', 'far', 'near_clip',
            'poses', 'render_poses','img_name'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


if __name__=='__main__':

    # load setup
    json_file, output = sys.argv[1], sys.argv[2]

    cfg = mmcv.Config.fromfile('configs/nerf/hotdog.py')

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(json_file)

    ckpt_path = './fine_last.tar'

    model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, ckpt_path).to(device)
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        },
    }

    os.makedirs(output, exist_ok=True)
    print('All results are dumped into', output)
    rgbs, depths, bgmaps = render_viewpoints(
            render_poses=data_dict['poses'],
            Ks=data_dict['Ks'],
            **render_viewpoints_kwargs)
    for i in range(rgbs.shape[0]):
        imageio.imwrite(os.path.join(output, '%s.png' %data_dict['img_name'][i]),utils.to8b(rgbs[i]))



