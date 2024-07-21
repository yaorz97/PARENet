import os.path as osp
import pdb
import random

import numpy as np
import torch.utils.data

from pareconv.utils.common import load_pickle
from pareconv.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
    uniform_2_sphere
)
from pareconv.utils.registration import get_correspondences, compute_overlap, compute_overlap_mask
import os

class OdometryKittiPairDataset(torch.utils.data.Dataset):
    ODOMETRY_KITTI_DATA_SPLIT = {
        'train': ['00', '01', '02', '03', '04', '05'],
        'val': ['06', '07'],
        'test': ['08', '09', '10'],
    }

    def __init__(
        self,
        dataset_root,
        metadata_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        augmentation_crop=False,
        point_keep_ratio=1.,
        return_corr_indices=False,
        matching_radius=0.6,
    ):
        super(OdometryKittiPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius

        self.metadata = load_pickle(osp.join(metadata_root, f'{subset}.pkl'))
        self.augmentation_crop = augmentation_crop
        self.point_keep_ratio = point_keep_ratio

    def _load_overlap_masks(self, seq_id, ref_frame, src_frame):
        data = np.load(osp.join(self.dataset_root, f'train_pair_overlap_masks/{seq_id}/masks_{ref_frame}_{src_frame}.npz'))
        ref_masks = data['ref_masks']
        src_masks = data['src_masks']
        return ref_masks, src_masks
    def _compute_overalap_mask(self):
        from tqdm import tqdm
        for index in tqdm(range(self.__len__())):
            metadata = self.metadata[index]
            seq_id = metadata['seq_id']
            ref_frame = metadata['frame0']
            src_frame = metadata['frame1']
            # get transformation

            transform = metadata['transform']
            # get point cloud
            ref_points = np.load(osp.join(self.dataset_root, metadata['pcd0']))
            src_points = np.load(osp.join(self.dataset_root, metadata['pcd1']))
            ref_overlap_mask, src_overlap_mask = compute_overlap_mask(ref_points, src_points, transform, positive_radius=self.matching_radius)
            os.makedirs(osp.join(self.dataset_root, f'train_pair_overlap_masks/{seq_id}'), exist_ok=True)
            np.savez_compressed(
                osp.join(self.dataset_root, f'train_pair_overlap_masks/{seq_id}/masks_{ref_frame}_{src_frame}.npz'),
                ref_masks=ref_overlap_mask,
                src_masks=src_overlap_mask)
    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale
        src_points = src_points * scale
        translation = translation * scale
        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_points = ref_points + ref_shift
        src_points = src_points + src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform

    def _load_point_cloud(self, file_name):
        points = np.load(file_name)
        indices = None
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points, indices

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

        metadata = self.metadata[index]
        data_dict['seq_id'] = metadata['seq_id']
        data_dict['ref_frame'] = metadata['frame0']
        data_dict['src_frame'] = metadata['frame1']
        ref_points, ref_indices = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd0']))
        src_points, src_indices = self._load_point_cloud(osp.join(self.dataset_root, metadata['pcd1']))
        transform = metadata['transform']

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        if self.subset == 'train' and self.augmentation_crop:
            ref_masks, src_masks = self._load_overlap_masks(metadata['seq_id'], metadata['frame0'], metadata['frame1'])
            ref_masks = ref_masks[ref_indices] if not ref_indices is None else ref_masks
            src_masks = src_masks[src_indices] if not src_indices is None else src_masks
            ref_points, src_points = self._random_crop(ref_points, src_points, ref_masks, src_masks, self.point_keep_ratio)

            overlap = compute_overlap(ref_points, src_points, transform, positive_radius=self.matching_radius)
            if overlap < 0.1:
                return self.__getitem__(np.random.randint(0, len(self.metadata)))

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        return data_dict

    def __len__(self):
        return len(self.metadata)
