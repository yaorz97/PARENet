import pdb
from functools import partial

import numpy as np
import torch

from pareconv.modules.ops import grid_subsample, radius_search
from pareconv.utils.torch import build_dataloader


# Stack mode utilities

def precompute_subsample(points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio):
    assert num_stages == len(num_neighbors)

    points_list = []
    lengths_list = []
    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= subsample_ratio  # 2 for 3DMatch, 2.5 for KITTI
    return {
        'points': points_list,
        'lengths': lengths_list,
    }
def precompute_neibors(points_list, lengths_list, num_stages, num_neighbors):

    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # knn search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]
        if i < num_stages:    # without adaptive sampling  i < num_stages, with: i < num_stages
            neighbors = radius_search(
                cur_points,
                cur_points,
                cur_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                num_neighbors[i],
            )
            subsampling_list.append(subsampling)

            if i > 0:
                upsampling = radius_search(
                    cur_points,
                    sub_points,
                    cur_lengths,
                    sub_lengths,
                    1,
                )
                upsampling_list.append(upsampling)
    return {
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }

def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, num_neighbors, subsample_ratio, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        num_neighbors (List[int])
        precompute_data (bool)
    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_subsample(points, lengths, num_stages, voxel_size, num_neighbors, subsample_ratio)

        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 2))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    num_neighbors,
    subsample_ratio,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            num_neighbors=num_neighbors,
            subsample_ratio=subsample_ratio,
            precompute_data=precompute_data,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader
