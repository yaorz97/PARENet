import torch
import torch.nn as nn
import torch.nn.functional as F
from pareconv.modules.ops import point_to_node_partition, index_select
from pareconv.modules.registration import get_node_correspondences

from pareconv.modules.dual_matching import PointDualMatching

from pareconv.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)

from pareconv.modules.registration import HypothesisProposer

from backbone import PAREConvFPN


class PARE_Net(nn.Module):
    def __init__(self, cfg):
        super(PARE_Net, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = PAREConvFPN(
            cfg.backbone.init_dim,
            cfg.backbone.output_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.share_nonlinearity,
            cfg.backbone.conv_way,
            cfg.backbone.use_xyz,
            cfg.fine_matching.use_encoder_re_feats
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = HypothesisProposer(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            num_hypotheses=cfg.fine_matching.num_hypotheses,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.point_matching = PointDualMatching(dim=cfg.backbone.output_dim // 3 * 3)

    def forward(self, data_dict):
        output_dict = {}
        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )  # ref_N_c,  [ref_N_c, 64],  [ref_N_c, 64],
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )
        output_dict['ref_node_knn_indices'] = ref_node_knn_indices
        output_dict['src_node_knn_indices'] = src_node_knn_indices

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)# [ref_N_f + 1, 3]
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0) #[ref_N_c, 64, 3]
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )  # coarse correspondences  gt_node_corr_indices: [N, 2]  gt_node_corr_overlaps : N


        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. PARE-Conv Encoder
        re_feats_f, feats_f, re_feats_c, feats_c, m_scores = self.backbone(data_dict)

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]

        ref_feats_c_re = re_feats_c[:ref_length_c]
        src_feats_c_re = re_feats_c[ref_length_c:]
        output_dict['ref_feats_c_re'] = ref_feats_c_re
        output_dict['src_feats_c_re'] = src_feats_c_re

        ref_feats_c, src_feats_c, scores_list = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )


        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 4. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        m_ref_scores = m_scores[:ref_length_f]
        m_src_scores = m_scores[ref_length_f:]
        re_ref_feats_f = re_feats_f[:ref_length_f]
        re_src_feats_f = re_feats_f[ref_length_f:]

        output_dict['m_ref_scores'] = m_ref_scores
        output_dict['m_src_scores'] = m_src_scores
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        output_dict['re_ref_feats_f'] = re_ref_feats_f
        output_dict['re_src_feats_f'] = re_src_feats_f

        # 5. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices
            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 6 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        m_ref_padded_scores = torch.cat([m_ref_scores, torch.zeros_like(m_ref_scores[:1])], dim=0)
        m_src_padded_scores = torch.cat([m_src_scores, torch.zeros_like(m_src_scores[:1])], dim=0)
        ref_node_corr_knn_scores = index_select(m_ref_padded_scores, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_scores = index_select(m_src_padded_scores, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points   # 256 64 3
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        re_ref_padded_feats_f = torch.cat([re_ref_feats_f, torch.zeros_like(re_ref_feats_f[:1])], dim=0)
        re_src_padded_feats_f = torch.cat([re_src_feats_f, torch.zeros_like(re_src_feats_f[:1])], dim=0)
        re_ref_node_corr_knn_feats = index_select(re_ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        re_src_node_corr_knn_feats = index_select(re_src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['re_ref_node_corr_knn_feats'] = re_ref_node_corr_knn_feats   # 256 64 21 3
        output_dict['re_src_node_corr_knn_feats'] = re_src_node_corr_knn_feats

        # 7 Match batched points
        matching_scores = self.point_matching(ref_node_corr_knn_feats, src_node_corr_knn_feats, ref_node_corr_knn_scores, src_node_corr_knn_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores   # 256 64 64
        output_dict['ref_node_corr_knn_scores'] = ref_node_corr_knn_scores
        output_dict['src_node_corr_knn_scores'] = src_node_corr_knn_scores

        # 8 Generate hypotheses and select the best one
        with torch.no_grad():
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, hypotheses, re_ref_corr_feats, re_src_corr_feats, = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                re_ref_node_corr_knn_feats,
                re_src_node_corr_knn_feats,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
            )

        output_dict['re_ref_corr_feats'] = re_ref_corr_feats
        output_dict['re_src_corr_feats'] = re_src_corr_feats
        output_dict['hypotheses'] = hypotheses
        output_dict['ref_corr_points'] = ref_corr_points
        output_dict['src_corr_points'] = src_corr_points
        output_dict['corr_scores'] = corr_scores
        output_dict['estimated_transform'] = estimated_transform
        output_dict['transform'] = transform

        return output_dict


def create_model(config):
    model = PARE_Net(config)
    return model

def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
