import os
import io
import torch
import torch.utils.data as data
import pickle
import numpy as np
import mediapy as media
from PIL import Image
from lib.datasets.cotracker.data_util import (
    CoTrackerData, sample_queries_first, sample_queries_strided
)


class Dataset(data.Dataset):
    """
    Dataset Class for tap_vid davis
    """
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        # load parameters
        self.data_root = os.path.join(kwargs['data_root'], "tapvid_davis.pkl")
        self.split = kwargs['split']
        self.seq_len = kwargs['seq_len']
        self.traj_per_sample = kwargs['traj_per_sample']
        self.resize_to_256 = kwargs['resize_to_256']
        self.queried_type = kwargs['queried_type']

        with open(self.data_root, "rb") as f:
            self.points_dataset = pickle.load(f)
            self.video_names = list(self.points_dataset.keys())
        print("found %d unique videos in %s" % (len(self.points_dataset), self.data_root))

    def __getitem__(self, index):

        video_name = self.video_names[index]
        video = self.points_dataset[video_name]
        frames = video["video"]

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = self.points_dataset[video_name]["points"]
        if self.resize_to_256:
            frames = media.resize_video(frames, (256, 256))
            target_points *= np.array([256, 256])
        else:
            target_points *= np.array([frames.shape[2], frames.shape[1]])

        T, H, W, C = frames.shape
        N, T, D = target_points.shape

        target_occ = self.points_dataset[video_name]["occluded"]

        # random crop video
        if self.split == 'train':
            assert self.seq_len <= T
            if self.seq_len < T:
                start_ind = np.random.choice(T - self.seq_len, 1)[0]

                frames = frames[start_ind: start_ind + self.seq_len]
                target_points = target_points[:, start_ind: start_ind + self.seq_len]
                target_occ = target_occ[:, start_ind: start_ind + self.seq_len]

        if self.queried_type == 'first':
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = (
            torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()
        )  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        segs = torch.ones(self.seq_len if self.split == 'train' else T, 1, H, W).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(
            1, 0
        )  # T, N

        if self.split == 'train':
            gotit = True
            visibile_pts_first_frame_inds = (visibles[0]).nonzero(as_tuple=False)[:, 0]
            point_inds = torch.randperm(len(visibile_pts_first_frame_inds))[: self.traj_per_sample]
            if len(point_inds) < self.traj_per_sample:
                gotit = False

            visible_inds_sampled = visibile_pts_first_frame_inds[point_inds]

            trajs = trajs[:, visible_inds_sampled].float()
            visibles = visibles[:, visible_inds_sampled]
            valids = torch.ones((self.seq_len, self.traj_per_sample))

            sample = CoTrackerData(
                video=rgbs,
                segmentation=segs,
                trajectory=trajs,
                visibility=visibles,
                valid=valids,
                seq_name=str(video_name)
            )

            return sample

        else:
            query_points = torch.from_numpy(converted["query_points"])[0]  # T, N
            return CoTrackerData(
                rgbs,
                segs,
                trajs,
                visibles,
                seq_name=str(video_name),
                query_points=query_points,
            )

    def __len__(self):
        return len(self.points_dataset)



