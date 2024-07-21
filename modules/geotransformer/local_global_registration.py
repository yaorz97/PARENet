import pdb
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pareconv.modules.ops import apply_transform
from pareconv.modules.registration import WeightedProcrustes

class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        acceptance_radius: float,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        use_global_score: bool = False,
        correspondence_threshold: int = 3,
        correspondence_limit: Optional[int] = None,
        num_refinement_steps: int = 5,
    ):
        r"""Point Matching with Local-to-Global Registration.

        Args:
            k (int): top-k selection for matching.
            acceptance_radius (float): acceptance radius for LGR.
            mutual (bool=True): mutual or non-mutual matching.
            confidence_threshold (float=0.05): ignore matches whose scores are below this threshold.
            use_dustbin (bool=False): whether dustbin row/column is used in the score matrix.
            use_global_score (bool=False): whether use patch correspondence scores.
            correspondence_threshold (int=3): minimal number of correspondences for each patch correspondence.
            correspondence_limit (optional[int]=None): maximal number of verification correspondences.
            num_refinement_steps (int=5): number of refinement steps.
        """
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.acceptance_radius = acceptance_radius
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.use_global_score = use_global_score
        self.correspondence_threshold = correspondence_threshold
        self.correspondence_limit = correspondence_limit
        self.num_refinement_steps = num_refinement_steps
        self.procrustes = WeightedProcrustes(return_transform=True)

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()

        # delta = (ref_knn_masks.sum(-1) * src_knn_masks.sum(-1) + 1).float().sqrt()

        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)
        # ref_corr_mat = torch.gt(ref_score_mat, 0.2 / delta[:, None, None])



        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        num_correspondences = min(2000, mask_mat.sum())
        corr_scores, corr_indices = score_mat.view(-1).topk(k=num_correspondences, largest=True)
        batch_sel_indices = corr_indices // (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices0 = corr_indices % (score_mat.shape[1] * score_mat.shape[2])
        ref_sel_indices = ref_sel_indices0 // (score_mat.shape[2])
        src_sel_indices = ref_sel_indices0 % score_mat.shape[1]
        corr_mat = torch.zeros_like(mask_mat, device=mask_mat.device)
        corr_mat[batch_sel_indices, ref_sel_indices, src_sel_indices] = True

        # src_corr_mat = torch.gt(src_score_mat, 0.2 / delta[:, None, None])
        # merge results from two sides


        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]
        corr_mat = torch.logical_and(corr_mat, mask_mat)
        return corr_mat

    @staticmethod
    def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
        r"""Convert stacked correspondences to batched points.

        The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
        transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
        into a batch.

        Args:
            ref_corr_points (Tensor): (C, 3)
            src_corr_points (Tensor): (C, 3)
            corr_scores (Tensor): (C,)
            chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

        Returns:
            batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
            batch_corr_scores (Tensor): (B, K), padded with zeros.
        """
        batch_size = len(chunks)
        indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
        ref_corr_points = ref_corr_points[indices]  # (total, 3)
        src_corr_points = src_corr_points[indices]  # (total, 3)
        corr_scores = corr_scores[indices]  # (total,)

        max_corr = np.max([y - x for x, y in chunks])
        target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
        indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
        indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
        indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

        batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
        batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

        batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
        batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
        batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

        batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
        batch_corr_scores.index_put_([indices], corr_scores)
        batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

        return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

    def recompute_correspondence_scores(self, ref_corr_points, src_corr_points, corr_scores, estimated_transform):
        aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
        corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
        inlier_masks = torch.lt(corr_residuals, self.acceptance_radius)
        new_corr_scores = corr_scores * inlier_masks.float()
        return new_corr_scores
    def iter_gen_hypos(self, L2_dis,  src_keypts, tgt_keypts):
        bs = L2_dis.shape[0]
        m, k = 20, 6
        all_trans = []
        all_fitness = []
        all_inliers_mask = []
        inliers_mask = L2_dis < 0.1
        # all_inliers_mask.append(inliers_mask)
        for it in range(10):
            bias, knn_idx = torch.topk(L2_dis, m, -1, largest=False)
            knn_idx1 = knn_idx[:, :, m-k:]

            src_knn = src_keypts.gather(dim=1, index=knn_idx1.reshape([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
                [bs, -1, k, 3])  # [bs, num_seeds, k, 3]
            tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx1.reshape([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
                [bs, -1, k, 3])  # [bs, num_seeds, k, 3]


            src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
            #
            seedwise_trans1 = self.procrustes(src_knn, tgt_knn)
            seedwise_trans1 = seedwise_trans1.view([bs, -1, 4, 4])
            pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans1[:, :, :3, :3],
                                         src_keypts.permute(0, 2, 1)) + seedwise_trans1[:, :, :3,
                                                                     3:4]  # [bs, num_seeds, num_corr, 3]
            pred_position = pred_position.permute(0, 1, 3, 2)
            L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
            inliers_mask = L2_dis < 0.1
            inliers_fitness = torch.mean(inliers_mask.float(), dim=-1)
            all_inliers_mask.append(inliers_mask)
            all_trans.append(seedwise_trans1)
            all_fitness.append(inliers_fitness)

        all_trans = torch.cat(all_trans, 1)
        all_fitness = torch.cat(all_fitness, 1)
        all_inliers_mask = torch.cat(all_inliers_mask, 1)
        return all_fitness, all_trans, all_inliers_mask
    def hypothesis_nms(self,local_transform, scores, d_thresh, n):
        dist = torch.norm(local_transform[:, None, :3, 3] - local_transform[None, :, :3, 3], 2, -1)
        scores = scores + torch.rand_like(scores).to(scores) * 1e-4
        score_relation = scores[:, None] >= scores[None, :]  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dist >= d_thresh).bool()
        is_local_max = score_relation.min(-1)[0].float()
        if is_local_max.sum() > n:
            cluster_value, clusters_idx = torch.topk(scores + is_local_max * 1, n, -1, largest=True)
        else:
            clusters_idx = torch.where(is_local_max)[0]
        return clusters_idx
    def local_to_global_registration(self, ref_knn_points, src_knn_points, score_mat, corr_mat):
        # extract dense correspondences
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)
        global_ref_corr_points = ref_knn_points[batch_indices, ref_indices]
        global_src_corr_points = src_knn_points[batch_indices, src_indices]
        global_corr_scores = score_mat[batch_indices, ref_indices, src_indices]
        # build verification set
        if self.correspondence_limit is not None and global_corr_scores.shape[0] > self.correspondence_limit:
            corr_scores, sel_indices = global_corr_scores.topk(k=self.correspondence_limit, largest=True)
            ref_corr_points = global_ref_corr_points[sel_indices]
            src_corr_points = global_src_corr_points[sel_indices]
        else:
            ref_corr_points = global_ref_corr_points
            src_corr_points = global_src_corr_points
            corr_scores = global_corr_scores

        # compute starting and ending index of each patch correspondence.
        # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
        # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
        unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
        unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
        unique_indices = unique_indices.detach().cpu().numpy().tolist()
        unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
        chunks = [
            (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= self.correspondence_threshold
        ]

        batch_size = len(chunks)
        if batch_size > 0:
            # local registration
            batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = self.convert_to_batch(
                global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
            )
            batch_transforms = self.procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
            batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
            batch_corr_residuals = torch.linalg.norm(
                ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
            )
            batch_inlier_masks = torch.lt(batch_corr_residuals, self.acceptance_radius)  # (P, N)
            # n_seeds = min(batch_inlier_masks.shape[0], 50)
            # # # _, seeds = batch_inlier_masks.sum(-1).topk(n_seeds, -1, largest=True)
            # #
            # seeds = self.hypothesis_nms(batch_transforms, batch_inlier_masks.float().mean(-1), 0.2, n_seeds),
            # batch_corr_residuals = batch_corr_residuals[seeds]
            # batch_transforms = batch_transforms[seeds]
            # all_fitness, all_trans, all_inliers_mask = self.iter_gen_hypos(batch_corr_residuals.unsqueeze(0),
            #                                                                src_corr_points.unsqueeze(0),
            #                                                                ref_corr_points.unsqueeze(0))
            # best_index = all_fitness[0].argmax()
            # cur_corr_scores = corr_scores * all_inliers_mask[0, best_index].float()
            # print(all_inliers_mask[0, best_index].sum())
            # print(batch_inlier_masks[seeds[0]].sum())
            # pdb.set_trace()


            ir = batch_inlier_masks.sum(dim=1)
            best_index = ir.argmax()
            cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
        else:
            # degenerate: initialize transformation with all correspondences
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, corr_scores)
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )

        # global refinement
        estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
        for _ in range(self.num_refinement_steps - 1):
            cur_corr_scores = self.recompute_correspondence_scores(
                ref_corr_points, src_corr_points, corr_scores, estimated_transform
            )
            estimated_transform = self.procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

        return global_ref_corr_points, global_src_corr_points, global_corr_scores, estimated_transform

    def forward(
        self,
        ref_knn_points,
        src_knn_points,
        ref_knn_masks,
        src_knn_masks,
        score_mat,
        global_scores,
    ):
        r"""Point Matching Module forward propagation with Local-to-Global registration.

        Args:
            ref_knn_points (Tensor): (B, K, 3)
            src_knn_points (Tensor): (B, K, 3)
            ref_knn_masks (BoolTensor): (B, K)
            src_knn_masks (BoolTensor): (B, K)
            score_mat (Tensor): (B, K, K) or (B, K + 1, K + 1), log likelihood
            global_scores (Tensor): (B,)

        Returns:
            ref_corr_points: torch.LongTensor (C, 3)
            src_corr_points: torch.LongTensor (C, 3)
            corr_scores: torch.Tensor (C,)
            estimated_transform: torch.Tensor (4, 4)
        """
        # score_mat = torch.exp(score_mat)

        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks)  # (B, K, K)
        if self.use_dustbin:
            score_mat = score_mat[:, :-1, :-1]
        if self.use_global_score:
            score_mat = score_mat * global_scores.view(-1, 1, 1)
        score_mat = score_mat * corr_mat.float()

        ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.local_to_global_registration(
            ref_knn_points, src_knn_points, score_mat, corr_mat
        )

        return ref_corr_points, src_corr_points, corr_scores, estimated_transform, None
