from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
from lib.config import cfg
from lib.datasets.cotracker.tapvid_datasets import CoTrackerData


# collator function for sanity check
def collate_fn_test(batch):
    """
    Collate function for video tracks data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    valid = torch.stack([b.valid for b in batch], dim=0)
    query_points = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]

    return CoTrackerData(
        video,
        segmentation,
        trajectory,
        visibility,
        valid=valid,
        seq_name=seq_name,
        query_points=query_points,
    )


def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    segmentation = torch.stack([b.segmentation for b, _ in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b, _ in batch], dim=0)
    visibility = torch.stack([b.visibility for b, _ in batch], dim=0)
    valid = torch.stack([b.valid for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    return (
        CoTrackerData(video, segmentation, trajectory, visibility, valid, seq_name),
        gotit,
    )

_collators = {
    'train': collate_fn_train,
    'test': collate_fn_test
}

def make_collator(cfg, is_train):
    collator = cfg.train.collator if is_train else cfg.test.collator
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate
