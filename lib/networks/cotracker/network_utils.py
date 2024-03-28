import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp


# Functions
EPS = 1e-6


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
    # returns shape-1
    # axis can be a list of axes
    for (a, b) in zip(x.size(), mask.size()):
        assert a == b  # some shape mismatch!
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean

def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def bilinear_sample2d(im, x, y, return_inbounds=False):
    # x and y are each B, N
    # output is B, C, N
    if len(im.shape) == 5:
        B, N, C, H, W = list(im.shape)
    else:
        B, C, H, W = list(im.shape)
    N = list(x.shape)[1]

    x = x.float()
    y = y.float()
    H_f = torch.tensor(H, dtype=torch.float32)
    W_f = torch.tensor(W, dtype=torch.float32)

    # inbound_mask = (x>-0.5).float()*(y>-0.5).float()*(x<W_f+0.5).float()*(y<H_f+0.5).float()

    max_y = (H_f - 1).int()
    max_x = (W_f - 1).int()

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    x0_clip = torch.clamp(x0, 0, max_x)
    x1_clip = torch.clamp(x1, 0, max_x)
    y0_clip = torch.clamp(y0, 0, max_y)
    y1_clip = torch.clamp(y1, 0, max_y)
    dim2 = W
    dim1 = W * H

    base = torch.arange(0, B, dtype=torch.int64, device=x.device) * dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N])

    base_y0 = base + y0_clip * dim2
    base_y1 = base + y1_clip * dim2

    idx_y0_x0 = base_y0 + x0_clip
    idx_y0_x1 = base_y0 + x1_clip
    idx_y1_x0 = base_y1 + x0_clip
    idx_y1_x1 = base_y1 + x1_clip

    # use the indices to lookup pixels in the flat image
    # im is B x C x H x W
    # move C out to last dim
    if len(im.shape) == 5:
        im_flat = (im.permute(0, 3, 4, 1, 2)).reshape(B * H * W, N, C)
        i_y0_x0 = torch.diagonal(im_flat[idx_y0_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y0_x1 = torch.diagonal(im_flat[idx_y0_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x0 = torch.diagonal(im_flat[idx_y1_x0.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
        i_y1_x1 = torch.diagonal(im_flat[idx_y1_x1.long()], dim1=1, dim2=2).permute(
            0, 2, 1
        )
    else:
        im_flat = (im.permute(0, 2, 3, 1)).reshape(B * H * W, C)
        i_y0_x0 = im_flat[idx_y0_x0.long()]
        i_y0_x1 = im_flat[idx_y0_x1.long()]
        i_y1_x0 = im_flat[idx_y1_x0.long()]
        i_y1_x1 = im_flat[idx_y1_x1.long()]

    # Finally calculate interpolated values.
    x0_f = x0.float()
    x1_f = x1.float()
    y0_f = y0.float()
    y1_f = y1.float()

    w_y0_x0 = ((x1_f - x) * (y1_f - y)).unsqueeze(2)
    w_y0_x1 = ((x - x0_f) * (y1_f - y)).unsqueeze(2)
    w_y1_x0 = ((x1_f - x) * (y - y0_f)).unsqueeze(2)
    w_y1_x1 = ((x - x0_f) * (y - y0_f)).unsqueeze(2)

    output = (
        w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1
    )
    # output is B*N x C
    output = output.view(B, -1, C)
    output = output.permute(0, 2, 1)
    # output is B x C x N

    if return_inbounds:
        x_valid = (x > -0.5).byte() & (x < float(W_f - 0.5)).byte()
        y_valid = (y > -0.5).byte() & (y < float(H_f - 0.5)).byte()
        inbounds = (x_valid & y_valid).float()
        inbounds = inbounds.reshape(
            B, N
        )  # something seems wrong here for B>1; i'm getting an error here (or downstream if i put -1)
        return output, inbounds

    return output  # B, C, N


def meshgrid2d(B, Y, X, stack=False, norm=False, device="cpu"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x


def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)


# Blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """
    CNN Feature Extract Network
    """
    def __init__(
        self, input_dim=3, output_dim=128, stride=8, norm_fn="batch", dropout=0.0
    ):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = 64

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.shallow = False
        if self.shallow:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.conv2 = nn.Conv2d(128 + 96 + 64, output_dim, kernel_size=1)
        else:
            self.layer1 = self._make_layer(64, stride=1)
            self.layer2 = self._make_layer(96, stride=2)
            self.layer3 = self._make_layer(128, stride=2)
            self.layer4 = self._make_layer(128, stride=2)

            self.conv2 = nn.Conv2d(
                128 + 128 + 96 + 64,
                output_dim * 2,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        if self.shallow:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c], dim=1))
        else:
            a = self.layer1(x)
            b = self.layer2(a)
            c = self.layer3(b)
            d = self.layer4(c)
            a = F.interpolate(
                a,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            b = F.interpolate(
                b,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            c = F.interpolate(
                c,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            d = F.interpolate(
                d,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )
            x = self.conv2(torch.cat([a, b, c, d], dim=1))
            x = self.norm2(x)
            x = self.relu2(x)
            x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x


class AttnBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            _, _, _, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B * S * N, 1, H, W), coords_lvl)
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class UpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=12,
        time_depth=12,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor):
        x = self.input_transform(input_tensor)

        j = 0
        for i in range(len(self.time_blocks)):
            B, N, T, _ = x.shape
            x_time = rearrange(x, "b n t c -> (b n) t c", b=B, t=T, n=N)
            x_time = self.time_blocks[i](x_time)

            x = rearrange(x_time, "(b n) t c -> b n t c ", b=B, t=T, n=N)
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_blocks)) == 0
            ):
                x_space = rearrange(x, "b n t c -> (b t) n c ", b=B, t=T, n=N)
                x_space = self.space_blocks[j](x_space)
                x = rearrange(x_space, "(b t) n c -> b n t c  ", b=B, t=T, n=N)
                j += 1

        flow = self.flow_head(x)
        return flow
