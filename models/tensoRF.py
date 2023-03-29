import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .tensorBase import *


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, **kargs)

    def init_svd_volume(self, res):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = nn.Linear(sum(self.app_n_comp), self.app_dim, bias_attr=False)

    def init_one_svd(self, n_component, gridSize, scale):
        gridSize_list = gridSize.tolist()
        plane_coef, line_coef = [], []
        weight_attr = paddle.ParamAttr(learning_rate=20)
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_init = nn.initializer.Assign(scale * paddle.randn([1, n_component[i], gridSize_list[mat_id_1], gridSize_list[mat_id_0]]))
            
            plane_coef.append(paddle.create_parameter(shape=[1, n_component[i], gridSize_list[mat_id_1], gridSize_list[mat_id_0]], dtype='float32',  \
                                                 default_initializer=plane_init, attr=weight_attr))
            
            line_init = nn.initializer.Assign(scale * paddle.randn([1, n_component[i], gridSize_list[vec_id], 1]))
            
            line_coef.append(paddle.create_parameter(shape=[1, n_component[i], gridSize_list[vec_id], 1], dtype='float32', \
                                                     default_initializer=line_init, attr=weight_attr))

        return nn.ParameterList(plane_coef), nn.ParameterList(line_coef)
    
    # torch的优化器和paddle优化器构造时有区别，参数字典中设置的lr是初始学习率比例  
    # lr_init_spatialxyz 为初始的学习率0.02，比例为1
    # lr_init_network 为初始学习率0.02的1/20，即0.001，比例为0.05
    def get_optparam_groups(self):
        grad_vars = [{'params': self.density_line}, {'params': self.density_plane},
                     {'params': self.app_line}, {'params': self.app_plane},
                         {'params': self.basis_mat.parameters()}]
        if isinstance(self.renderModule, nn.Layer):
            grad_vars += [{'params':self.renderModule.parameters()}]
        return grad_vars
    
    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[:-1]
            dotp = paddle.matmul(vector_comps[idx].reshape([n_comp,n_size]), paddle_trans(vector_comps[idx].reshape([n_comp,n_size]), -1,-2))
            non_diagonal = dotp.reshape([-1])[1:].reshape([n_comp-1, n_comp+1])[...,:-1]
            total = total + paddle.mean(paddle.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + paddle.mean(paddle.abs(self.density_plane[idx])) + paddle.mean(paddle.abs(self.density_line[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 
        return total

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = paddle.stack((xyz_sampled.index_select(self.mat_index[0], axis=-1), \
                                         xyz_sampled.index_select(self.mat_index[1], axis=-1), \
                                         xyz_sampled.index_select(self.mat_index[2], axis=-1))).detach().reshape([3, -1, 1, 2])    
        coordinate_line = paddle.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = paddle.stack((paddle.zeros_like(coordinate_line), coordinate_line), axis=-1).detach().reshape([3, -1, 1, 2])

        sigma_feature = paddle.zeros((xyz_sampled.shape[0],))
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
            sigma_feature = sigma_feature + paddle.sum(plane_coef_point * line_coef_point, axis=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = paddle.stack((xyz_sampled.index_select(self.mat_index[0], axis=-1), \
                                         xyz_sampled.index_select(self.mat_index[1], axis=-1), \
                                         xyz_sampled.index_select(self.mat_index[2], axis=-1))).detach().reshape([3, -1, 1, 2])
        coordinate_line = paddle.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = paddle.stack((paddle.zeros_like(coordinate_line), coordinate_line), axis=-1).detach().reshape([3, -1, 1, 2])

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).reshape([-1, *xyz_sampled.shape[:1]]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).reshape([-1, *xyz_sampled.shape[:1]]))
        plane_coef_point, line_coef_point = paddle.concat(plane_coef_point), paddle.concat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @paddle.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            paddle.assign(F.interpolate(
            plane_coef[i], size=(res_target[mat_id_1], res_target[mat_id_0]), mode="bilinear", align_corners=True), plane_coef[i]
            )
            paddle.assign(F.interpolate(
            line_coef[i], size=(res_target[vec_id], 1), mode='bilinear', align_corners=True), line_coef[i]
            )

        return plane_coef, line_coef

    @paddle.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @paddle.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = paddle.round(paddle.round(t_l)).astype('int64'), paddle.round(b_r).astype('int64') + 1
        b_r = paddle.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            paddle.assign(self.density_line[i][..., t_l[mode0]:b_r[mode0], :], self.density_line[i])
            paddle.assign(self.app_line[i][...,t_l[mode0]:b_r[mode0],:], self.app_line[i])
             
            mode0, mode1 = self.matMode[i]
            paddle.assign(self.density_plane[i][...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]], self.density_plane[i])
            paddle.assign(self.app_plane[i][...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]], self.app_plane[i])


        if not paddle.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = paddle.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, **kargs)


    def init_svd_volume(self, res):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2)
        self.basis_mat = nn.Linear(self.app_n_comp[0], self.app_dim, bias_attr=False)


    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        gridSize_list = gridSize.tolist()
        weight_attr = paddle.ParamAttr(learning_rate=20)
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_init = nn.initializer.Assign(scale * paddle.randn([1, n_component, gridSize_list[vec_id], 1]))            
            line_coef.append(paddle.create_parameter(shape=[1, n_component, gridSize_list[vec_id], 1], dtype='float32', \
                                                     default_initializer=line_init, attr=weight_attr))

        return nn.ParameterList(line_coef)

    # torch的优化器和paddle优化器构造时有区别，参数字典中设置的lr是初始学习率比例  
    # lr_init_spatialxyz 为初始的学习率0.02，比例为1
    # lr_init_network 为初始学习率0.02的1/20，即0.001，比例为0.05
    def get_optparam_groups(self):
        grad_vars = [{'params': self.density_line},
                     {'params': self.app_line},
                     {'params': self.basis_mat.parameters()}]
        if isinstance(self.renderModule, nn.Layer):
            grad_vars += [{'params':self.renderModule.parameters()}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        
        coordinate_line = paddle.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = paddle.stack((paddle.zeros_like(coordinate_line), coordinate_line), axis=-1).detach().reshape([3, -1, 1, 2])

        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                        align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
        sigma_feature = paddle.sum(line_coef_point, axis=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        
        coordinate_line = paddle.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = paddle.stack((paddle.zeros_like(coordinate_line), coordinate_line), axis=-1).detach().reshape([3, -1, 1, 2])


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).reshape([-1, *xyz_sampled.shape[:1]])

        return self.basis_mat(line_coef_point.T)
    

    @paddle.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            paddle.assign(
                F.interpolate(density_line_coef[i], size=(res_target[vec_id], 1), mode='bilinear', align_corners=True), density_line_coef[i])
            paddle.assign(
                F.interpolate(app_line_coef[i], size=(res_target[vec_id], 1), mode='bilinear', align_corners=True), app_line_coef[i]
            )

        return density_line_coef, app_line_coef

    @paddle.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @paddle.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = paddle.round(paddle.round(t_l)).astype('int64'), paddle.round(b_r).astype('int64') + 1
        b_r = paddle.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]

            paddle.assign(self.density_line[i][..., t_l[mode0]:b_r[mode0], :], self.density_line[i])
            paddle.assign(self.app_line[i][...,t_l[mode0]:b_r[mode0],:], self.app_line[i])


        if not paddle.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = paddle.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + paddle.mean(paddle.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total