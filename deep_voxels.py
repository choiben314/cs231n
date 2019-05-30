import torch
import torch.nn as nn
import numpy as np

from projection import *
from custom_layers import *
import functools
import util

from pytorch_prototyping.pytorch_prototyping import *

class DeepVoxels(nn.Module):
    def __init__(self,
                 img_sidelength,
                 lifting_img_dims,
                 frustrum_img_dims,
                 grid_dims,
                 num_grid_feats=8,
                 nf0=8,
                 use_occlusion_net=True):
        ''' Initializes the DeepVoxels model.

        :param img_sidelength: The sidelength of the input images (for instance 512)
        :param lifting_img_dims: The dimensions of the feature map to be lifted.
        :param frustrum_img_dims: The dimensions of the canonical view volume that DeepVoxels are resampled to.
        :param grid_dims: The dimensions of the deepvoxels grid.
        :param grid_dims: The number of featres in the outermost layer of U-Nets.
        :param use_occlusion_net: Whether to use the OcclusionNet or not.
        '''
        super().__init__()

        self.use_occlusion_net = use_occlusion_net
        self.grid_dims = grid_dims

        self.norm = nn.BatchNorm2d

        self.lifting_img_dims = lifting_img_dims
        self.frustrum_img_dims = frustrum_img_dims
        self.grid_dims = grid_dims

        # The frustrum depth is the number of voxels in the depth dimension of the canonical viewing volume.
        # It's calculated as the length of the diagonal of the DeepVoxels grid.
        self.frustrum_depth = int(np.ceil(1.5 * grid_dims[-1]))

        self.nf0 = nf0 # Number of features to use in the outermost layer of all U-Nets
        self.n_grid_feats = num_grid_feats  # Number of features in the DeepVoxels grid.
        self.occnet_nf = 4  # Number of features to use in the 3D unet of the occlusion subnetwork

        num_downs = util.num_divisible_by_2(img_sidelength) - 1

        # Feature extractor is an asymmetric UNet: Straight downsampling to 64x64, then a UNet with skip connections
        self.feature_extractor = nn.Sequential(
            # DownsamplingNet([self.nf0 * (2 ** i) for i in range(num_downs - 5)],
            #                 in_channels=3,
            #                 use_dropout=False,
            #                 norm=self.norm),
            # Unet(in_channels=self.nf0 * (2 ** (num_downs - 6)),
            #      out_channels=self.n_grid_feats,
            #      nf0=self.nf0 * (2 ** (num_downs - 6)),
            #      use_dropout=False,
            #      max_channels=8*self.nf0,
            #      num_down=5,
            #      norm=self.norm)
            Unet(in_channels=3,
                 out_channels=self.n_grid_feats,
                 nf0=self.nf0 * (2 ** (num_downs - 6)),
                 use_dropout=False,
                 max_channels=8*self.nf0,
                 num_down=5,
                 norm=self.norm)
        )

        # Rendering net is an asymmetric UNet: UNet with skip connections and then straight upsampling
        # self.rendering_net = nn.Sequential(
        #     Unet(in_channels=self.n_grid_feats,
        #          # out_channels=4 * self.nf0,
        #          out_channels=self.nf0,
        #          use_dropout=True,
        #          dropout_prob=0.1,
        #          nf0=self.nf0 * (2 ** (num_downs - 6)),
        #          max_channels=8 * self.nf0,
        #          num_down=5,
        #          norm=self.norm),  # from 64 to 2 and back
        #     # UpsamplingNet([4 * self.nf0, self.nf0] + max(0, num_downs - 7) * [self.nf0],
        #     #               in_channels=4 * self.nf0,  # 4*self.nf0
        #     #               use_dropout=True,
        #     #               dropout_prob=0.1,
        #     #               norm=self.norm,
        #     #               upsampling_mode='transpose'),
        #     Conv2dSame(self.nf0, out_channels=self.nf0 // 2, kernel_size=3, bias=False),
        #     nn.BatchNorm2d(self.nf0 // 2),
        #     nn.ReLU(True),
        #     Conv2dSame(self.nf0 // 2, out_channels=3, kernel_size=3),
        #     nn.Tanh()
        # )

        # NEW RENDERING NET
        self.rendering_net = nn.Sequential(
        		Conv2dSame(self.n_grid_feats, out_channels=128, kernel_size=1, bias=False),
	            nn.BatchNorm2d(128),
	            nn.ReLU(True),
	            Conv2dSame(128, out_channels=128, kernel_size=1, bias=False),
	            nn.BatchNorm2d(128),
	            nn.ReLU(True),
	            Conv2dSame(128, out_channels=128, kernel_size=1, bias=False),
	            nn.BatchNorm2d(128),
	            nn.ReLU(True),
	            Conv2dSame(128, out_channels=128, kernel_size=1, bias=False),
	            nn.BatchNorm2d(128),
	            nn.ReLU(True),
	            Conv2dSame(128, out_channels=3, kernel_size=1),
	            nn.Tanh(),

        	)

        if self.use_occlusion_net:
            self.occlusion_net = OcclusionNet(nf0=self.n_grid_feats,
                                              occnet_nf=self.occnet_nf,
                                              frustrum_dims=[self.frustrum_img_dims[0], self.frustrum_img_dims[1],
                                                             self.frustrum_depth])
            print(self.occlusion_net)
        else:
            self.depth_collapse_net = nn.Sequential(
                Conv2dSame(self.n_grid_feats * self.frustrum_depth,
                           out_channels=self.nf0 * self.grid_dims[-1] // 2,
                           kernel_size=3,
                           bias=False),
                self.norm(self.nf0 * self.grid_dims[-1] // 2),
                nn.ReLU(True),
                Conv2dSame(self.nf0 * self.grid_dims[-1] // 2,
                           out_channels=self.nf0 * self.grid_dims[-1] // 8,
                           kernel_size=3,
                           bias=False),
                self.norm(self.nf0 * self.grid_dims[-1] // 8),
                nn.ReLU(True),
                Conv2dSame(self.nf0 * self.grid_dims[-1] // 8,
                           out_channels=self.nf0,
                           kernel_size=3,
                           bias=False),
                self.norm(self.nf0),
                nn.ReLU(True),
            )
            print(self.frustrum_collapse_net)

        # The deepvoxels grid is registered as a buffer - meaning, it is safed together with model parameters, but is
        # not trainable.
        self.register_buffer("deepvoxels",
                             torch.zeros(
                                 (1, self.n_grid_feats, self.grid_dims[0], self.grid_dims[1], self.grid_dims[2])))

        self.integration_net = IntegrationNet(self.n_grid_feats,
                                              use_dropout=True,
                                              coord_conv=True,
                                              per_feature=False,
                                              grid_dim=grid_dims[-1])

        self.inpainting_net = Unet3d(in_channels=self.n_grid_feats + 3,
                                     out_channels=self.n_grid_feats,
                                     num_down=2,
                                     nf0=self.n_grid_feats,
                                     max_channels=4 * self.n_grid_feats)

        print(100 * "*")
        print("inpainting_net")
        util.print_network(self.inpainting_net)
        print(self.inpainting_net)
        print("rendering net")
        util.print_network(self.rendering_net)
        print(self.rendering_net)
        print("feature extraction net")
        util.print_network(self.feature_extractor)
        print(self.feature_extractor)
        print(100 * "*")

        # Coordconv volumes
        coord_conv_volume = np.mgrid[-self.grid_dims[0] // 2:self.grid_dims[0] // 2,
                                     -self.grid_dims[1] // 2:self.grid_dims[1] // 2,
                                     -self.grid_dims[2] // 2:self.grid_dims[2] // 2]

        coord_conv_volume = np.stack(coord_conv_volume, axis=0).astype(np.float32)
        coord_conv_volume = coord_conv_volume / self.grid_dims[0]
        self.coord_conv_volume = torch.Tensor(coord_conv_volume).float().cuda()[None, :, :, :, :]

    def forward(self,
                input_img,
                proj_frustrum_idcs_list,
                proj_grid_coords_list,
                lift_volume_idcs,
                lift_img_coords,
                writer):
        if input_img is not None:
            # Training mode: Extract features from input img, lift them, and update the deepvoxels volume.
            img_feats = self.feature_extractor(input_img)
            temp_feat_vol = interpolate_lifting(img_feats, lift_volume_idcs, lift_img_coords, self.grid_dims)

            dv_new = self.integration_net(temp_feat_vol, self.deepvoxels.detach(), writer)
            self.deepvoxels.data = dv_new
        else:
            # Testing mode: Use the pre-trained deepvoxels volume.
            dv_new = self.deepvoxels

        inpainting_input = torch.cat([dv_new, self.coord_conv_volume], dim=1)
        dv_inpainted = self.inpainting_net(inpainting_input)

        novel_views, depth_maps = list(), list()

        for i, (proj_frustrum_idcs, proj_grid_coords) in enumerate(zip(proj_frustrum_idcs_list, proj_grid_coords_list)):
            can_view_vol = interpolate_trilinear(dv_inpainted,
                                                 proj_frustrum_idcs,
                                                 proj_grid_coords,
                                                 self.frustrum_img_dims,
                                                 self.frustrum_depth)
            if self.use_occlusion_net:
                visibility_weights, depth_map = self.occlusion_net(can_view_vol)
                depth_maps.append(depth_map)

                collapsed_frustrum = torch.mean(visibility_weights * can_view_vol, dim=2)
                novel_image_features = collapsed_frustrum.contiguous().view(
                    [1, -1, self.frustrum_img_dims[0], self.frustrum_img_dims[1]])
            else:
                frustrum_collapse_input = can_view_vol.view(1, -1, self.frustrum_img_dims[0], self.frustrum_img_dims[1])
                novel_image_features = self.depth_collapse_net(frustrum_collapse_input)
                depth_maps.append(torch.zeros((1, 1, 64, 64)))

            rendered_img = 0.5 * self.rendering_net(novel_image_features)
            novel_views.append(rendered_img)

        return novel_views, depth_maps
