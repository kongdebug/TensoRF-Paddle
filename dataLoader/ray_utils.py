import paddle, re
import numpy as np
from paddle import searchsorted


# from utils import index_point_feature

def depth2dist(z_vals, cos_angle):
    # z_vals: [N_ray N_sample]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = paddle.concat([dists, paddle.to_tensor([1e10]).expand(dists[..., :1].shape)], axis=-1)  # [N_rays, N_samples]
    dists = dists * cos_angle.unsqueeze(-1)
    return dists


def ndc2dist(ndc_pts, cos_angle):
    dists = paddle.norm(ndc_pts[:, 1:] - ndc_pts[:, :-1], axis=-1)
    dists = paddle.concat([dists, 1e10 * cos_angle.unsqueeze(-1)], axis=-1)  # [N_rays, N_samples]
    return dists


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = paddle.meshgrid(paddle.linspace(0, W-1, W), paddle.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t() + 0.5
    j = j.t() + 0.5
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = paddle.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], paddle.ones_like(i)], axis=-1)  # (H, W, 3)

    return directions


def get_ray_directions_blender(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = paddle.meshgrid(paddle.linspace(0, W-1, W), paddle.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t() + 0.5
    j = j.t() + 0.5
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = paddle.stack([(i - cent[0]) / focal[0], -(j - cent[1]) / focal[1], -paddle.ones_like(i)],
                             axis=-1)  # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.reshape([-1, 3])
    rays_o = rays_o.reshape([-1, 3])

    return rays_o, rays_d


def ndc_rays_blender(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = paddle.stack([o0, o1, o2], axis=-1)
    rays_d = paddle.stack([d0, d1, d2], axis=-1)

    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = (near - rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = 1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = 1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. - 2. * near / rays_o[..., 2]

    d0 = 1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = 1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = 2. * near / rays_o[..., 2]

    rays_o = paddle.stack([o0, o1, o2], axis=-1)
    rays_d = paddle.stack([d0, d1, d2], axis=-1)

    return rays_o, rays_d

def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / paddle.sum(weights, axis=-1, keepdim=True)
    cdf = paddle.cumsum(pdf, axis=-1)
    cdf = paddle.concat([paddle.zeros_like(cdf[..., :1]), cdf], axis=-1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = paddle.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = paddle.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = paddle.to_tensor(u)

    # Invert CDF
    inds = searchsorted(cdf.detach(), u, right=True)
    below = paddle.max(paddle.zeros_like(inds - 1), inds - 1)
    above = paddle.min((cdf.shape[-1] - 1) * paddle.ones_like(inds), inds)
    inds_g = paddle.stack([below, above], axis=-1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = paddle_gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = paddle_gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = paddle.where(denom < 1e-5, paddle.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def dda(rays_o, rays_d, bbox_3D):
    inv_ray_d = 1.0 / (rays_d + 1e-6)
    t_min = (bbox_3D[:1] - rays_o) * inv_ray_d  # N_rays 3
    t_max = (bbox_3D[1:] - rays_o) * inv_ray_d
    t = paddle.stack((t_min, t_max))  # 2 N_rays 3
    t_min = paddle.max(paddle.min(t, axis=0)[0], axis=-1, keepdim=True)[0]
    t_max = paddle.min(paddle.max(t, axis=0)[0], axis=-1, keepdim=True)[0]
    return t_min, t_max


def ray_marcher(rays,
                N_samples=64,
                lindisp=False,
                perturb=0,
                bbox_3D=None):
    """
    sample points along the rays
    Inputs:
        rays: ()

    Returns:

    """

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    if bbox_3D is not None:
        # cal aabb boundles
        near, far = dda(rays_o, rays_d, bbox_3D)

    # Sample depth points
    z_steps = paddle.linspace(0, 1, N_samples)  # (N_samples)
    if not lindisp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = paddle.concat([z_vals_mid, z_vals[:, -1:]], axis=-1)
        lower = paddle.concat([z_vals[:, :1], z_vals_mid], axis=-1)

        perturb_rand = perturb * paddle.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    return xyz_coarse_sampled, rays_o, rays_d, z_vals


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def ndc_bbox(all_rays):
    near_min = paddle.min(all_rays[...,:3].reshape([-1, 3]), axis=0)[0]
    near_max = paddle.max(all_rays[..., :3].reshape([-1, 3]), axis=0)[0]
    far_min = paddle.min((all_rays[...,:3]+all_rays[...,3:6]).reshape([-1, 3]), axis=0)[0]
    far_max = paddle.max((all_rays[...,:3]+all_rays[...,3:6]).reshape([-1, 3]), axis=0)[0]
    print(f'===> ndc bbox near_min:{near_min} near_max:{near_max} far_min:{far_min} far_max:{far_max}')
    return paddle.stack((paddle.minimum(near_min,far_min),paddle.maximum(near_max,far_max)))