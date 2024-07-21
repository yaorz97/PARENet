import os
import os.path as osp
import pdb
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm

from pareconv.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
    uniform_2_sphere
)
from pareconv.utils.registration import get_correspondences, compute_overlap, compute_overlap_mask

class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        metadata_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        augmentation_crop=False,
        point_keep_ratio=1.,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,

    ):
        super(ThreeDMatchPairDataset, self).__init__()
        self.metadata_root = metadata_root
        self.data_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] < self.overlap_threshold]
        self.augmentation_crop = augmentation_crop
        self.point_keep_ratio = point_keep_ratio

    def __len__(self):
        return len(self.metadata_list)

    def _compute_overlap_mask(self):
        # compute overlapped points for RandomCrop augmentation
        for index in tqdm(range(self.__len__())):
            metadata: Dict = self.metadata_list[index]
            scene_name = metadata['scene_name']
            ref_frame = metadata['frag_id0']
            src_frame = metadata['frag_id1']
            overlap = metadata['overlap']
            if overlap < 0.3:
                continue
            # get transformation
            rotation = metadata['rotation']
            translation = metadata['translation']
            transform = get_transform_from_rotation_translation(rotation, translation)
            # get point cloud
            ref_points = torch.load(osp.join(self.data_root, metadata['pcd0']))
            src_points = torch.load(osp.join(self.data_root, metadata['pcd1']))
            ref_overlap_mask, src_overlap_mask = compute_overlap_mask(ref_points, src_points, transform)
            os.makedirs(osp.join(self.data_root, f'train_pair_overlap_masks/{scene_name}'), exist_ok=True)
            np.savez_compressed(osp.join(self.data_root, f'train_pair_overlap_masks/{scene_name}/masks_{ref_frame}_{src_frame}.npz'),
                                ref_masks=ref_overlap_mask,
                                src_masks=src_overlap_mask)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        indices = None
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points, indices

    def _load_overlap_masks(self, scene_name, ref_frame, src_frame):
        data = np.load(osp.join(self.data_root, f'train_pair_overlap_masks/{scene_name}/masks_{ref_frame}_{src_frame}.npz'))
        ref_masks = data['ref_masks']
        src_masks = data['src_masks']
        return ref_masks, src_masks

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        """Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def _random_crop(self, ref_points, src_points, ref_masks, src_masks, p_keep):

        rand_xyz = uniform_2_sphere()
        centroid = np.mean(ref_points, axis=0)
        points_centered = ref_points - centroid
        dist_from_plane = np.dot(points_centered, rand_xyz)
        mask1 = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
        mask2 = dist_from_plane < np.percentile(dist_from_plane, p_keep * 100)
        mask = mask1 if ref_masks[mask1].sum() < ref_masks[mask2].sum() else mask2
        ref_points = ref_points[mask]

        rand_xyz = uniform_2_sphere()
        centroid = np.mean(src_points, axis=0)
        points_centered = src_points - centroid
        dist_from_plane = np.dot(points_centered, rand_xyz)
        mask1 = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
        mask2 = dist_from_plane < np.percentile(dist_from_plane, p_keep * 100)
        mask = mask1 if src_masks[mask1].sum() < src_masks[mask2].sum() else mask2
        src_points = src_points[mask]

        return ref_points, src_points

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points, ref_indices = self._load_point_cloud(metadata['pcd0'])
        src_points, src_indices = self._load_point_cloud(metadata['pcd1'])

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)


        transform = get_transform_from_rotation_translation(rotation, translation)
        # cropping point cloud pairs whose overlap is greater than 0.3
        if self.augmentation_crop and metadata['overlap'] > 0.3:
            ref_masks, src_masks = self._load_overlap_masks(metadata['scene_name'], metadata['frag_id0'], metadata['frag_id1'])
            ref_masks = ref_masks[ref_indices] if not ref_indices is None else ref_masks
            src_masks = src_masks[src_indices] if not src_indices is None else src_masks
            ref_points, src_points = self._random_crop(ref_points, src_points, ref_masks, src_masks, self.point_keep_ratio)

            overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.1)
            # ensuring overlap greater than 0.1
            if overlap < 0.1:
                return self.__getitem__(np.random.randint(0, len(self.metadata_list)))

        overlap = compute_overlap(ref_points, src_points, transform)
        data_dict['overlap'] = overlap

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict
