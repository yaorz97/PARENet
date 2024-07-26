import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.layers import VNLinear, VNLinearLeakyReLU, VNLeakyReLU, VNStdFeature
from pareconv.modules.ops import index_select

class CorrelationNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1):
        super(CorrelationNet, self).__init__()
        self.vn_layer = VNLinearLeakyReLU(in_channel, out_channel * 2, dim=4, share_nonlinearity=False, negative_slope=0.2)
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.insert(0, out_channel * 2)
        hidden_unit.append(out_channel)
        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            if i < len(hidden_unit) - 1 or last_bn:
                self.mlp_bns_hidden.append(nn.BatchNorm1d(hidden_unit[i]))

    def forward(self, xyz):
        # xyz : N * D * 3 * k
        N, _, _, K = xyz.size()
        scores = self.vn_layer(xyz)
        scores = torch.norm(scores, p=2, dim=2)  # transform rotation equivairant feats into rotation invariant feats
        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        scores = F.softmax(scores/self.temp, dim=1)
        return scores

class PARE_Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, share_nonlinearity=False):
        super(PARE_Conv_Block, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        in_dim = in_dim + 2   # 1 + 2: [xyz, mean, cross]
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, in_dim, out_dim // 2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(in_dim, self.kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        """
        q_pts N1 * 3
        s_pts N2 * 3
        q_feats N1 * D * 3
        s_feats N2 * D * 3
        neighbor_indices   N1 * k
        """
        N, K = neighbor_indices.shape

        # compute relative coordinates
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)  # [N, 1, 3, K]
        centers = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], 1) # [N, 3, 3, K] rotation equivariant spatial features

        # predict correlation scores
        scores = self.score_net(local_feats) # [N, kernel_size,  K]

        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', local_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N,  self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1) # [N, D/2, 3, K]

        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)
        # mean pooling
        new_feats = normed_feats.mean(-1)
        # applying VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # mapping D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]

        return new_feats

class PARE_Conv_Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, shortcut_linear=False, share_nonlinearity=False, conv_info=None):
        super(PARE_Conv_Resblock, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        self.conv_way = conv_info["conv_way"]
        self.use_xyz = conv_info["use_xyz"]
        conv_dim = in_dim * 2 if self.conv_way == 'edge_conv' else in_dim
        if self.use_xyz: conv_dim += 1
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, conv_dim, out_dim//2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(conv_dim, self.kernel_size * out_dim//2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.shortcut_proj = VNLinear(in_dim, out_dim) if shortcut_linear else nn.Identity()
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)
    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        """
        q_pts N1 * 3
        s_pts N2 * 3
        q_feats N1 * D * 3
        s_feats N2 * D * 3
        neighbor_indices   N1 * k
        """

        N, K = neighbor_indices.shape
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)    # N1 *1 * 3 * k
        # compute relative coordinates
        center = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, center, dim=2)
        local_feats = torch.cat([pts, center, cross], 1)# [N, 3, 3, K] rotation equivariant spatial features
        # predict correlation scores
        scores = self.score_net(local_feats)
        # gather neighbors features
        neighbor_feats = s_feats[neighbor_indices, :].permute(0, 2, 3, 1)                            # N1  D * 3 k
        # shortcut
        identify = neighbor_feats[..., 0]
        identify = self.shortcut_proj(identify)
        # get edge features
        if self.conv_way == 'edge_conv':
            q_feats = neighbor_feats[..., 0:1]
            neighbor_feats = torch.cat([neighbor_feats - q_feats, neighbor_feats], 1)
        # use relative coordinates
        if self.use_xyz:
            neighbor_feats = torch.cat([neighbor_feats, pts], 1)
        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', neighbor_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N, self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1)

        # use L2 Norm instead of VNBatchNorm to reduce computation cost and accelerate convergence
        normed_feats = F.normalize(pro_feats, p=2, dim=2)
        # mean pooling
        new_feats = normed_feats.mean(-1)
        # apply VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # map D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]
        # add shortcut
        new_feats = new_feats + identify

        return new_feats

class PAREConvFPN(nn.Module):
    def __init__(self, init_dim, output_dim, kernel_size, share_nonlinearity=False, conv_way='edge_conv', use_xyz=True, use_encoder_re_feats=True):
        super(PAREConvFPN, self).__init__()
        conv_info = {'conv_way': conv_way, 'use_xyz': use_xyz}
        self.use_encoder_re_feats = use_encoder_re_feats
        self.encoder2_1 = PARE_Conv_Block(1, init_dim // 3, kernel_size, share_nonlinearity=share_nonlinearity)
        self.encoder2_2 = PARE_Conv_Resblock(init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder2_3 = PARE_Conv_Resblock(2 * init_dim // 3, 2 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.encoder3_1 = PARE_Conv_Resblock(2 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_2 = PARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder3_3 = PARE_Conv_Resblock(4 * init_dim // 3, 4 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.encoder4_1 = PARE_Conv_Resblock(4 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=True, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_2 = PARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)
        self.encoder4_3 = PARE_Conv_Resblock(8 * init_dim // 3, 8 * init_dim // 3, kernel_size, shortcut_linear=False, share_nonlinearity=share_nonlinearity, conv_info=conv_info)

        self.coarse_RI_head = VNLinear(8 * init_dim // 3, 8 * init_dim // 3)
        self.coarse_std_feature = VNStdFeature(8 * init_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.decoder3 = VNLinearLeakyReLU(12 * init_dim // 3, 4 * init_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.decoder2 = VNLinearLeakyReLU(6 * init_dim // 3, output_dim // 3, dim=3, share_nonlinearity=share_nonlinearity)
        self.RI_head = VNLinear(output_dim // 3, output_dim // 3)
        self.RE_head = VNLinear(output_dim // 3, output_dim // 3)

        self.fine_std_feature = VNStdFeature(output_dim // 3, dim=3, normalize_frame=True, share_nonlinearity=share_nonlinearity)

        self.matching_score_proj = nn.Linear(output_dim // 3 * 3, 1)

    def forward(self, data_dict):
        # feats_list = []
        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
        feats_s1 = points_list[0][:, None]
        # feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        coarse_feats = self.coarse_RI_head(feats_s4)
        ri_feats_c, _ = self.coarse_std_feature(coarse_feats)

        ri_feats_c = ri_feats_c.reshape(ri_feats_c.shape[0], -1)

        up1 = upsampling_list[1]
        latent_s3 = index_select(feats_s4, up1[:, 0], dim=0)
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        up2 = upsampling_list[0]
        latent_s2 = index_select(latent_s3, up2[:, 0], dim=0)
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        ri_feats = self.RI_head(latent_s2)
        re_feats = self.RE_head(latent_s2)

        ri_feats_f, local_rot = self.fine_std_feature(ri_feats)
        ri_feats_f = ri_feats_f.reshape(ri_feats_f.shape[0], -1)
        m_scores = self.matching_score_proj(ri_feats_f).sigmoid().squeeze()
        if not self.training and self.use_encoder_re_feats:
            # using rotation equivariant features from encoder to solve transformation may generate better hypotheses,
            # probably because a larger receptive field would contaminate rotation equivariant features
            re_feats_f = feats_s2
        else:
            re_feats_f = re_feats
        return re_feats_f, ri_feats_f, feats_s4, ri_feats_c, m_scores

