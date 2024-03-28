import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Mapping
import numpy as np


@dataclass(eq=False)
class CoTrackerData:
    """
    Data Struct for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    segmentation: torch.Tensor  # B, S, 1, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, S, N
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj


def resize_sample(rgbs, trajs_g, segs, interp_shape):
    S, C, H, W = rgbs.shape
    S, N, D = trajs_g.shape

    assert D == 2

    rgbs = F.interpolate(rgbs, interp_shape, mode="bilinear")
    segs = F.interpolate(segs, interp_shape, mode="nearest")

    trajs_g[:, :, 0] *= interp_shape[1] / W
    trajs_g[:, :, 1] *= interp_shape[0] / H
    return rgbs, trajs_g, segs


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }

