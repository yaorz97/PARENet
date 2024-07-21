from pareconv.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from pareconv.utils.data import (
    registration_collate_fn_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=cfg.train.augmentation_crop,
        point_keep_ratio=cfg.train.point_keep_ratio,
        matching_radius=cfg.train.matching_radius
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
        precompute_data=True
    )

    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        augmentation_crop=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
        precompute_data=True
    )

    return train_loader, valid_loader, cfg.backbone.num_neighbors


def test_data_loader(cfg, benchmark):
    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        cfg.data.metadata_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
        augmentation_crop=False,
        rotated=False
    )


    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.num_neighbors,
        cfg.backbone.subsample_ratio,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, cfg.backbone.num_neighbors
