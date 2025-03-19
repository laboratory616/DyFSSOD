from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mmengine.visualization import Visualizer

from mmdet.models.backbones.resnet import ResNet
from mmdet.registry import MODELS
import torch.nn as nn
import torch.nn.functional as F
from .fe_twostagedetector import DyFSSOD_twostagedetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor
from mmdet.structures import SampleList, DetDataSample
import copy
import math
from matplotlib.colors import Normalize


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


import matplotlib.pyplot as plt
import numpy as np
import torch



class ResConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Dy_Sample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp',
                 groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2,
                         (self.scale - 1) / 2 + 1) / self.scale
        meshgrid = torch.meshgrid(h, h, indexing='ij')
        return torch.stack(meshgrid).transpose(1, 2).repeat(
            1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        meshgrid = torch.meshgrid(coords_w, coords_h, indexing='ij')
        coords = torch.stack(meshgrid).transpose(1, 2).unsqueeze(
            1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype,
                                  device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W),
                                 self.scale).view(B, 2, -1,
                                                  self.scale * H,
                                                  self.scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        x = x.reshape(B * self.groups, -1, H, W)
        return F.grid_sample(x, coords, mode='bilinear',
                             align_corners=False,
                             padding_mode="border").view(
            B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5
            offset += self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) *
                                       self.scope(x_).sigmoid(),
                                       self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale
                                       ) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class BlockDCT2D(nn.Module):
    def __init__(self, block_size=8):
        super(BlockDCT2D, self).__init__()
        self.block_size = block_size
        self.dct_mat = self.create_dct_matrix(block_size)
        self.register_buffer('dct_mat_buffer', self.dct_mat)

    def create_dct_matrix(self, N):
        dct_mat = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    alpha = math.sqrt(1 / N)
                else:
                    alpha = math.sqrt(2 / N)
                dct_mat[k, n] = alpha * math.cos((math.pi * (2 * n + 1) * k) / (2 * N))
        return dct_mat

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block_size

        pad_h = (B - H % B) % B
        pad_w = (B - W % B) % B
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_unfold = x_padded.unfold(2, B, B).unfold(3, B, B)  # (N, C, n_blocks_h, n_blocks_w, B, B)
        n_blocks_h = x_unfold.size(2)
        n_blocks_w = x_unfold.size(3)

        x_blocks = x_unfold.contiguous().view(-1, B, B)  # (total_blocks, B, B)

        dct_mat = self.dct_mat_buffer.to(x.device)
        x_dct = torch.matmul(dct_mat, x_blocks)
        x_dct = torch.matmul(x_dct, dct_mat.t())

        x_dct = x_dct.view(N, C, n_blocks_h, n_blocks_w, B, B)

        return x_dct, (H, W)


class BlockIDCT2D(nn.Module):
    def __init__(self, block_size=8):
        super(BlockIDCT2D, self).__init__()
        self.block_size = block_size
        self.idct_mat = self.create_idct_matrix(block_size)
        self.register_buffer('idct_mat_buffer', self.idct_mat)

    def create_idct_matrix(self, N):
        idct_mat = torch.zeros(N, N)
        for n in range(N):
            for k in range(N):
                if k == 0:
                    alpha = math.sqrt(1 / N)
                else:
                    alpha = math.sqrt(2 / N)
                idct_mat[n, k] = alpha * math.cos((math.pi * (2 * n + 1) * k) / (2 * N))
        return idct_mat

    def forward(self, x_dct, output_size):
        N, C, n_blocks_h, n_blocks_w, B, _ = x_dct.shape
        H, W = output_size

        x_dct_blocks = x_dct.view(-1, B, B)

        idct_mat = self.idct_mat_buffer.to(x_dct.device)

        x_idct = torch.matmul(idct_mat, x_dct_blocks)
        x_idct = torch.matmul(x_idct, idct_mat.t())

        x_idct = x_idct.view(N, C, n_blocks_h, n_blocks_w, B, B)

        x_idct = x_idct.permute(0, 1, 2, 4, 3, 5).contiguous()  # 将 B, B 块的维度与 h, w 对齐
        x_idct = x_idct.view(N, C, n_blocks_h * B, n_blocks_w * B)

        x_reconstructed = x_idct[:, :, :H, :W]

        return x_reconstructed


class DCTSCE(nn.Module):
    def __init__(self, block_size=8):
        super(DCTSCE, self).__init__()
        self.block_size = block_size
        self.dct = BlockDCT2D(block_size=block_size)
        self.idct = BlockIDCT2D(block_size=block_size)
        self.zigzag_indices = self.get_zigzag_indices(block_size)
        # self.idct_norm = nn.BatchNorm2d(1)

    def get_zigzag_indices(self, N):
        """Generate zig-zag order indices for an N x N block."""
        indices = []
        for i in range(2 * N - 1):
            if i % 2 == 0:
                for y in range(max(0, i - N + 1), min(N, i + 1)):
                    x = i - y
                    indices.append((y, x))
            else:
                for x in range(max(0, i - N + 1), min(N, i + 1)):
                    y = i - x
                    indices.append((y, x))
        return indices  # List of (k1, k2) positions

    def forward(self, x):
        # x shape: (batch_size, channels, H, W)
        batch_size, channels, H, W = x.shape
        B = self.block_size

        # Perform block DCT
        x_dct, output_size = self.dct(x)  # x_dct shape: (batch_size, channels, n_blocks_h, n_blocks_w, B, B)
        n_blocks_h, n_blocks_w = x_dct.shape[2], x_dct.shape[3]

        # Reorder coefficients in ZIG-ZAG order
        # Initialize a list to store spatial subbands
        dct_spatial_cube = []

        # Create a mapping from index j to (k1, k2)
        index_to_coord = {}
        for idx, (k1, k2) in enumerate(self.zigzag_indices):
            index_to_coord[idx] = (k1, k2)

        # For each coefficient index j (from 0 to 63)
        for j in range(B * B):
            k1, k2 = index_to_coord[j]

            # Initialize a zeros tensor for DCT coefficients
            zeros = torch.zeros_like(x_dct)

            # Extract the j-th coefficient from all blocks
            coef = x_dct[..., k1, k2]  # Shape: (batch_size, channels, n_blocks_h, n_blocks_w)

            # Place the coefficient into zeros tensor at position (k1, k2)
            zeros[..., k1, k2] = coef

            # Perform inverse DCT to get spatial representation
            x_reconstructed = self.idct(zeros, output_size)  # Shape: (batch_size, channels, H, W)

            # Rescale each channel of x_reconstructed to range [-255, 255]
            min_val = x_reconstructed.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Min value per channel
            max_val = x_reconstructed.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Max value per channel
            # Scale to [-255, 255]
            x_reconstructed = 255 * (2 * (x_reconstructed - min_val) / (max_val - min_val + 1e-5) - 1)
            x_reconstructed = x_reconstructed.abs()

            dct_spatial_cube.append(x_reconstructed)

        # Stack along a new dimension to form the DCT spatial cube
        dct_spatial_cube = torch.cat(dct_spatial_cube, dim=1)  # Shape: (batch_size, channels, 64, H, W)

        return dct_spatial_cube


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()
        # Learnable parameters for temperature scaling and weight amplification
        self.temperature_param = nn.Parameter(torch.tensor(1.0))
        self.scale_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Compute the mean and standard deviation along the spatial dimensions (H, W)
        mean = x.mean(dim=(-2, -1), keepdim=True)  # (batch_size, channels, 1, 1)
        std = x.std(dim=(-2, -1), keepdim=True)  # (batch_size, channels, 1, 1)
        descriptor = std / (mean + 1e-8)  # Avoid division by zero, shape (batch_size, channels, 1, 1)

        # Squeeze spatial dimensions and transpose for convolution
        y = self.conv(descriptor.squeeze(-1).transpose(-1, -2))  # Shape: (batch_size, 1, channels)
        y = y.transpose(-1, -2).squeeze(-1)  # Shape: (batch_size, channels)

        # Apply temperature scaling (ensure temperature is positive)
        temperature = F.softplus(self.temperature_param)
        y = y / temperature

        # Apply Softmax over the channel dimension
        y = F.softmax(y, dim=1)  # Shape: (batch_size, channels)

        # Apply weight amplification (ensure scale is positive)
        scale = F.softplus(self.scale_param)
        y = y * scale

        # Reshape y to match the dimensions of x
        y = y.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, channels, 1, 1)

        # # Multi-scale information fusion
        # y = self.sigmoid(y)

        return x * y.expand_as(x)


# class WCMF(nn.Module):
#     def __init__(self, channel=256, dct_channels=64):
#         super(WCMF, self).__init__()
#         self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
#         self.conv_d1 = nn.Sequential(nn.Conv2d(dct_channels, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
#
#         self.conv_c1 = nn.Sequential(nn.Conv2d(2 * channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
#         self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#     def fusion(self, f1, f2, f_vec):
#         w1 = f_vec[:, 0, :, :].unsqueeze(1)
#         w2 = f_vec[:, 1, :, :].unsqueeze(1)
#         out1 = (w1 * f1) + (w2 * f2)
#         out2 = (w1 * f1) * (w2 * f2)
#         return out1 + out2
#
#     def forward(self, c1, dct):
#         Fr = self.conv_r1(c1)
#         Fd = self.conv_d1(dct)
#         f = torch.cat([Fr, Fd], dim=1)
#         f = self.conv_c1(f)
#         f = self.conv_c2(f)
#         # f = self.avgpool(f)
#         Fo = self.fusion(Fr, Fd, f)
#         return Fo


class DCTAttention(nn.Module):
    def __init__(self, c1_channels=256, dct_channels=64, reduction=4):
        super(DCTAttention, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(c1_channels * 2, c1_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(c1_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = ECA(c1_channels, k_size=5)
        # 定义嵌入层
        self.local_embedding = nn.Sequential(
            nn.Conv2d(dct_channels, c1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1_channels),
            nn.Conv2d(c1_channels, c1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1_channels),
        )

        self.global_embedding = nn.Sequential(
            nn.Conv2d(c1_channels, c1_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1_channels),
            nn.Conv2d(c1_channels, c1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1_channels)
        )

        self.act = nn.Hardsigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.pool_types = ['avg', 'max']
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(c1_channels, c1_channels // 16),
            nn.ReLU(),
            nn.Linear(c1_channels // 16, c1_channels)
        )

    def forward(self, c1, dct_spatial_cube):
        dct_att = self.attention(dct_spatial_cube)
        x_l = dct_att
        x_g = c1

        # 应用嵌入层
        local_feat = self.local_embedding(x_l)
        global_feat = self.global_embedding(x_g)
        sig_act = self.act(local_feat)

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x_g, (x_g.size(2), x_g.size(3)), stride=(x_g.size(2), x_g.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x_g, (x_g.size(2), x_g.size(3)), stride=(x_g.size(2), x_g.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x_g)
        x_out = x_g * scale
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        out = alpha * sig_act * global_feat + beta * x_out

        return out


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


@MODELS.register_module()
class DyFSSOD_CascadeRCNN_Att(DyFSSOD_twostagedetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 loss_res: OptConfigType = None,
                 dy_inchannels: int = 256,
                 outchannels: int = 256,
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.in_channels = dy_inchannels
        self.dysample1x = Dy_Sample(
            in_channels=self.in_channels, scale=2, style='lp', groups=4, dyscope=True)
        self.dysample2x = Dy_Sample(in_channels=self.in_channels, scale=2, style='lp', groups=4, dyscope=True)
        self.resconv = ResConv(self.in_channels, 3)
        self.channels = outchannels
        self.loss_res = MODELS.build(loss_res)
        self.block_dct = BlockDCT2D(block_size=8)
        self.block_idct = BlockIDCT2D(block_size=8)
        self.global_step = 0
        self.activation = nn.Sigmoid()
        self.learnable_thresh = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.initialized_thresh = False
        self.dctsce = DCTSCE()
        self.dctatt = DCTAttention(c1_channels=256, dct_channels=8 * 8)
        self.pool_types = ['avg', 'max']
        # self.wcmf = WCMF(self.in_channels, dct_channels=8 * 8)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             img_inputs: Tensor) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        # x = self.extract_feat(batch_inputs)

        x = self.backbone(batch_inputs)

        # DyFSSOD
        c1 = x[0].clone()
        reconstructed_c1 = self.dysample1x(c1)
        reconstructed_c1 = self.dysample2x(reconstructed_c1)
        if reconstructed_c1.shape[1] != img_inputs.shape[1]:
            reconstructed_c1 = self.resconv(reconstructed_c1)

        if reconstructed_c1.shape[2:] != img_inputs.shape[2:]:
            print('size error')
            reconstructed_c1 = F.interpolate(
                reconstructed_c1, size=img_inputs.shape[2:],
                mode='bilinear', align_corners=False)

        difference_map = torch.sum(torch.abs(reconstructed_c1 - img_inputs), dim=1, keepdim=True) / 3
        dct_spatial_cube = self.dctsce(difference_map)
        dct_spatial_cube = torch.nn.functional.interpolate(dct_spatial_cube,
                                                           size=(c1.shape[2], c1.shape[3]))  # nearest
        freq_att_c1 = self.dctatt(c1, dct_spatial_cube)
        x = list(x)
        x[0] = freq_att_c1
        if self.with_neck:
            x = self.neck(x)
        x = tuple(x)

        losses = dict()
        reconstruction_loss = self.loss_res(reconstructed_c1, img_inputs)
        custom_losses = {
            'reconstruction_loss': 1 * reconstruction_loss,
        }
        losses.update(custom_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                img_inputs: Tensor,
                rescale: bool = True) -> List[DetDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        # x = self.extract_feat(batch_inputs)
        x = self.backbone(batch_inputs)
        # DyFSSOD
        c1 = x[0].clone()
        reconstructed_c1 = self.dysample1x(c1)
        reconstructed_c1 = self.dysample2x(reconstructed_c1)
        if reconstructed_c1.shape[1] != img_inputs.shape[1]:
            reconstructed_c1 = self.resconv(reconstructed_c1)
        if reconstructed_c1.shape[2:] != img_inputs.shape[2:]:
            print('size error')
            reconstructed_c1 = F.interpolate(
                reconstructed_c1, size=img_inputs.shape[2:],
                mode='bilinear', align_corners=False)

        difference_map = torch.sum(torch.abs(reconstructed_c1 - img_inputs), dim=1, keepdim=True) / 3
        dct_spatial_cube = self.dctsce(difference_map)
        dct_spatial_cube = torch.nn.functional.interpolate(dct_spatial_cube,
                                                           size=(c1.shape[2], c1.shape[3]))  # nearest
        freq_att_c1 = self.dctatt(c1, dct_spatial_cube)
        x = list(x)
        x[0] = freq_att_c1

        if self.with_neck:
            x = self.neck(x)
        # x = tuple(x)
        # return x
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
