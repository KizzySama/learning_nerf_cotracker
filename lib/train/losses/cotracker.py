import torch
import torch.nn as nn
from lib.networks.cotracker.network_utils import reduce_masked_mean
from lib.config import cfg
import torch
import torch.nn.functional as F


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net

    def forward(self, batch):
        rgbs = batch.video
        trajs_g = batch.trajectory
        vis_g = batch.visibility
        valids = batch.valid
        B, T, C, H, W = rgbs.shape
        assert C == 3
        B, T, N, D = trajs_g.shape
        device = rgbs.device

        __, first_positive_inds = torch.max(vis_g, dim=1)
        # We want to make sure that during training the model sees visible points
        # that it does not need to track just yet: they are visible but queried from a later frame
        N_rand = N // 4
        # inds of visible points in the 1st frame
        nonzero_inds = [torch.nonzero(vis_g[0, :, i]) for i in range(N)]
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds
            ],
            dim=1,
        )
        first_positive_inds = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[:, N_rand:]], dim=1
        )
        ind_array_ = torch.arange(T, device=device)
        ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
        assert torch.allclose(
            vis_g[ind_array_ == first_positive_inds[:, None, :]],
            torch.ones_like(vis_g),
        )
        assert torch.allclose(
            vis_g[ind_array_ == rand_vis_inds[:, None, :]], torch.ones_like(vis_g)
        )

        gather = torch.gather(
            trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, 2)
        )
        xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

        queries = torch.cat([first_positive_inds[:, :, None], xys], dim=2)

        output = self.net(rgbs, queries, is_train=True)
        predictions, __, visibility, train_data = output

        vis_predictions, coord_predictions, wind_inds, sort_inds = train_data

        trajs_g = trajs_g[:, :, sort_inds]
        vis_g = vis_g[:, :, sort_inds]
        valids = valids[:, :, sort_inds]

        vis_gts = []
        traj_gts = []
        valids_gts = []

        for i, wind_idx in enumerate(wind_inds):
            ind = i * (cfg.network.S // 2)

            vis_gts.append(vis_g[:, ind: ind + cfg.network.S, :wind_idx])
            traj_gts.append(trajs_g[:, ind: ind + cfg.network.S, :wind_idx])
            valids_gts.append(valids[:, ind: ind + cfg.network.S, :wind_idx])

        scalar_stats = {}
        loss = 0

        seq_loss = sequence_loss(coord_predictions, traj_gts, vis_gts, valids_gts, 0.8).mean()
        scalar_stats.update({'seq_loss': seq_loss.mean()})
        loss += seq_loss

        vis_loss = balanced_ce_loss(vis_predictions, vis_gts, valids_gts).mean() * 10.0
        scalar_stats.update({'vis_loss': vis_loss})
        loss += vis_loss

        scalar_stats.update({'loss': loss})

        image_stats = {}

        return output, loss, scalar_stats, image_stats


def balanced_ce_loss(pred, gt, valid=None):
    total_balanced_loss = 0.0
    for j in range(len(gt)):
        B, S, N = gt[j].shape
        # pred and gt are the same shape
        for (a, b) in zip(pred[j].size(), gt[j].size()):
            assert a == b  # some shape mismatch!
        # if valid is not None:
        for (a, b) in zip(pred[j].size(), valid[j].size()):
            assert a == b  # some shape mismatch!

        pos = (gt[j] > 0.95).float()
        neg = (gt[j] < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred[j]
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

        pos_loss = reduce_masked_mean(loss, pos * valid[j])
        neg_loss = reduce_masked_mean(loss, neg * valid[j])

        balanced_loss = pos_loss + neg_loss
        total_balanced_loss += balanced_loss / float(N)
    return total_balanced_loss


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """Loss function defined over sequence of flow predictions"""
    total_flow_loss = 0.0
    for j in range(len(flow_gt)):
        B, S, N, D = flow_gt[j].shape
        assert D == 2
        B, S1, N = vis[j].shape
        B, S2, N = valids[j].shape
        assert S == S1
        assert S == S2
        n_predictions = len(flow_preds[j])
        flow_loss = 0.0
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            flow_pred = flow_preds[j][i]
            i_loss = (flow_pred - flow_gt[j]).abs()  # B, S, N, 2
            i_loss = torch.mean(i_loss, dim=3)  # B, S, N
            flow_loss += i_weight * reduce_masked_mean(i_loss, valids[j])
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss / float(N)
    return total_flow_loss