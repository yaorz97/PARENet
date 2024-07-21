import pdb

import torch
import torch.nn as nn

from pareconv.modules.loss import WeightedCircleLoss
from pareconv.modules.ops.transformation import apply_transform
from pareconv.modules.registration.metrics import isotropic_transform_error, relative_rotation_error
from pareconv.modules.ops.pairwise_distance import pairwise_distance


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss

class FineMatchingLoss(nn.Module): # for fine dual matching
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius
        self.negative_radius = cfg.fine_loss.negative_radius
        self.positive_margin = cfg.fine_loss.positive_margin
        self.negative_margin = cfg.fine_loss.negative_margin

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        ref_node_corr_knn_scores = output_dict['ref_node_corr_knn_scores']
        src_node_corr_knn_scores = output_dict['src_node_corr_knn_scores']

        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']
        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)
        # compute matching loss of rotation invariant features
        fine_ri_loss = - (matching_scores[gt_corr_map].log().mean()
                  + 0.5 * (1 - ref_node_corr_knn_scores)[slack_row_labels].log().mean()
                  + 0.5 * (1 - src_node_corr_knn_scores)[slack_col_labels].log().mean())

        # compute loss of rotation equivariant features
        neg_map = torch.gt(dists, self.negative_radius ** 2)
        neg_map = torch.logical_and(neg_map, gt_masks)
        fine_re_loss = self.fine_re_loss(output_dict, gt_corr_map, neg_map, transform)

        return fine_ri_loss, fine_re_loss

    def fine_re_loss(self, out_dict, gt_corr_map, neg_map, gt_trans):
        ref_feats = out_dict['re_ref_node_corr_knn_feats']
        src_feats = out_dict['re_src_node_corr_knn_feats']
        batch_indices, ref_indices, src_indices = torch.nonzero(gt_corr_map, as_tuple=True)
        if batch_indices.shape[0] == 0:
            return torch.tensor(0.0, device=ref_feats.device)
        ref_feats_rot = ref_feats[batch_indices, ref_indices]
        src_feats_rot = src_feats[batch_indices, src_indices]
        src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_trans[:3, :3])
        pos_loss = torch.relu(torch.norm(src_feats_rot - ref_feats_rot, 2, -1) - self.positive_margin).mean()

        batch_indices, ref_indices, src_indices = torch.nonzero(neg_map, as_tuple=True)
        ref_feats_rot = ref_feats[batch_indices, ref_indices]
        src_feats_rot = src_feats[batch_indices, src_indices]
        src_feats_rot = torch.einsum('bck, lk -> bcl', src_feats_rot, gt_trans[:3, :3])
        neg_loss = torch.relu(self.negative_margin - torch.norm(src_feats_rot - ref_feats_rot, 2, -1)).mean()
        re_loss = pos_loss + neg_loss
        return re_loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_ri_loss = cfg.loss.weight_fine_ri_loss
        self.weight_fine_re_loss = cfg.loss.weight_fine_re_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_ri_loss, fine_re_loss = self.fine_loss(output_dict, data_dict)
        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_ri_loss * fine_ri_loss + self.weight_fine_re_loss * fine_re_loss
        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_ri_loss': fine_ri_loss,
            'f_re_loss': fine_re_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.rre_threshold = cfg.eval.rre_threshold
        self.rte_threshold = cfg.eval.rte_threshold
        self.feat_rre_threshold = cfg.eval.feat_rre_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']

        if src_corr_points.shape[0] == 0:
            return torch.tensor(0.0, device=transform.device)
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        mask = torch.lt(corr_distances, self.acceptance_radius)
        precision = mask.float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, est_transform)
        recall = torch.logical_and(torch.lt(rre, self.rre_threshold), torch.lt(rte, self.rte_threshold)).float()
        return rre, rte, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte,  recall = self.evaluate_registration(output_dict, data_dict)
        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RR': recall,
        }
