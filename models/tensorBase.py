import numpy as np
import time

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .sh import eval_sh_bases
from .initializer import *


def positional_encoding(positions, freqs):
    freq_bands = (2**paddle.arange(freqs).astype('float32'))  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + [freqs * positions.shape[-1], ])  # (..., DF)
    pts = paddle.concat([paddle.sin(pts), paddle.cos(pts)], axis=-1)
    return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - paddle.exp(-sigma*dist)

    T = paddle.cumprod(paddle.concat([paddle.ones([alpha.shape[0], 1]), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.reshape([-1, 3, sh_mult.shape[-1]])
    rgb = F.relu(paddle.sum(sh_mult * rgb_sh, axis=-1) + 0.5)
    return rgb

def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
  
def paddle_trans(tensor, i, j):
    """
    return:tensor
    """
    shape = list(range(tensor.ndim))
    shape[i], shape[j] = shape[j], shape[i]
    y = paddle.transpose(tensor, perm=shape)
    return y


class AlphaGridMask(nn.Layer):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.aabb= aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.reshape([1,1,*alpha_volume.shape[-3:]])
        self.gridSize = paddle.to_tensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).astype('int64')

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.reshape([1,-1,1,1,3]), align_corners=True).reshape([-1])

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(nn.Layer):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)
        constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = paddle.concat(indata, axis=-1)
        rgb = self.mlp(mlp_in)
        rgb = F.sigmoid(rgb)

        return rgb


class MLPRender_PE(nn.Layer):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp =nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)
        constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = paddle.concat(indata, axis=-1)
        rgb = self.mlp(mlp_in)
        rgb = F.sigmoid(rgb)

        return rgb

class MLPRender(nn.Layer):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)
        constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = paddle.concat(indata, axis=-1)
        rgb = self.mlp(mlp_in)
        rgb = F.sigmoid(rgb)

        return rgb


class TensorBase(nn.Layer):
    def __init__(self, aabb, gridSize, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.mat_index = paddle.to_tensor(self.matMode)
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.init_svd_volume(gridSize[0])

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.reshape([-1]))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = paddle.to_tensor(gridSize).astype('int64')
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = paddle.mean(self.units)*self.step_ratio
        self.aabbDiag = paddle.sqrt(paddle.sum(paddle.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.astype('bool')
            alpha_volume = alpha_volume.cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape([-1]))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        paddle.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = paddle.to_tensor(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(ckpt['alphaMask.aabb'], alpha_volume.astype('float32'))
        self.set_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = paddle.linspace(near, far, N_samples).unsqueeze(0).astype(rays_o.dtype)

        if is_train:
            interpx += paddle.rand(shape=interpx.shape).astype(rays_o.dtype) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(axis=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = paddle.where(rays_d==0, paddle.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = paddle.clip(paddle.minimum(rate_a, rate_b).amax(-1), min=near, max=far)

        rng = paddle.arange(N_samples)[None].astype('float32')
        if is_train:
            rng = paddle.tile(rng, repeat_times=[rays_d.shape[-2],1])
            rng += paddle.rand(shape=rng[:,0:1].shape)
        step = stepsize * rng.astype(rays_o.dtype)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(axis=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @paddle.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = paddle.stack(paddle.meshgrid(
            paddle.linspace(0, 1, gridSize[0]),
            paddle.linspace(0, 1, gridSize[1]),
            paddle.linspace(0, 1, gridSize[2]),
        ), axis=-1)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        alpha = paddle.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].reshape([-1,3]), self.stepSize).reshape([gridSize[1], gridSize[2]])
        return alpha, dense_xyz

    @paddle.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = paddle_trans(dense_xyz, 0, 2)
        alpha = paddle_trans(alpha.clip(0,1), 0, 2)
        alpha = alpha[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).reshape(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = paddle.stack((xyz_min, xyz_max))

        total = paddle.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @paddle.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = paddle.to_tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        total_len = int(N)
        if total_len % chunk == 0:
            chunk_list = [chunk] * (total_len//chunk)
        else:
            chunk_list = [chunk] * (total_len//chunk) + [-1]
        idx_chunks = paddle.split(paddle.arange(N), chunk_list)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk]

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = paddle.where(rays_d == 0, paddle.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = paddle.minimum(rate_a, rate_b).amax(-1)
                t_max = paddle.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).reshape(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = paddle.concat(mask_filtered).reshape(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {paddle.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = paddle.ones_like(xyz_locs[:,0]).astype('bool')
            

        sigma = paddle.zeros(xyz_locs.shape[:-1])

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - paddle.exp(-sigma*length).reshape(xyz_locs.shape[:-1])

        return alpha


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = paddle.concat((z_vals[:, 1:] - z_vals[:, :-1], paddle.zeros_like(z_vals[:, :1])), axis=-1)
            rays_norm = paddle.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = paddle.concat((z_vals[:, 1:] - z_vals[:, :-1], paddle.zeros_like(z_vals[:, :1])), axis=-1)
        viewdirs = viewdirs.reshape([-1, 1, 3]).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None and ray_valid.any():
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = (alphas > 0).astype('int32')
            ray_invalid = (~ray_valid).astype('int32')
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_invalid =paddle.where(ray_invalid>0, 1, 0).astype('bool')
            ray_valid = ~ray_invalid


        sigma = paddle.zeros(xyz_sampled.shape[:-1])
        rgb = paddle.zeros((*xyz_sampled.shape[:2], 3))

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = paddle.sum(weight, axis=-1)
        rgb_map = paddle.sum(weight[..., None] * rgb, axis=-2)

        if white_bg or (is_train and paddle.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clip(0,1)

        with paddle.no_grad():
            depth_map = paddle.sum(weight * z_vals, axis=-1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

