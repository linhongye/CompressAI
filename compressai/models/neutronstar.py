"""NeutronStar2026 model — project-local image compression model.

Phase 7a: Replaced conv-based ``AttentionBlock`` (gated mechanism) with
``WindowAttentionBlock`` (window-based multi-head self-attention with
shifted windows and relative position bias) for true global/semi-global
feature modelling.

Previous phases:
- Phase 3: Native single-channel grayscale support via ``in_ch`` (default 1).
- Phase 6: Residual-aware loss training (TotalBppLoss).
"""

import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    WindowAttentionBlock,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
)
from compressai.registry import register_model

from .base import SimpleVAECompressionModel
from .utils import conv, deconv

__all__ = ["NeutronStar2026"]


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block (project-local copy).

    Sandwiches a 3x3 convolution between two 1x1 convolutions that reduce
    and then restore the channel count, following [He2016].

    [He2016]: "Deep Residual Learning for Image Recognition",
    He et al., CVPR 2016.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity


@register_model("neutron-star-2026")
class NeutronStar2026(SimpleVAECompressionModel):
    """NeutronStar2026: project-local image compression model.

    Phase 3 baseline — structurally identical to the ELIC 2022 official
    architecture [He2022] with native single-channel grayscale support via
    ``in_ch`` (default 1).  Copied here so that all subsequent project
    changes (residual training, etc.) are isolated from the upstream
    ``compressai`` codebase.

    Context model: unevenly grouped space-channel contextual adaptive
    coding (SCCTX) with a checkerboard spatial context per group.
    Backbone: modified attention-based transform from [Cheng2020].

    [He2022]: "ELIC: Efficient Learned Image Compression with Unevenly
    Grouped Space-Channel Contextual Adaptive Coding", He et al., CVPR 2022.

    [Cheng2020]: "Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules", Cheng et al., CVPR 2020.

    Args:
        N (int): Number of main network channels.
        M (int): Number of latent space channels.
        groups (list[int]): Channel counts per channel group (must sum to M).
        in_ch (int): Number of pixel-space channels (1 for grayscale, 3 for RGB).
    """

    def __init__(self, N: int = 192, M: int = 320, groups=None, in_ch: int = 1, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M
        self.in_ch = in_ch

        self.g_a = nn.Sequential(
            conv(in_ch, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            WindowAttentionBlock(N, shift=False),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            WindowAttentionBlock(M, shift=True),
        )

        self.g_s = nn.Sequential(
            WindowAttentionBlock(M, shift=True),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            WindowAttentionBlock(N, shift=False),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, in_ch, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )

        # Channel context networks — g_ch^(k) in [He2022].
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), 224, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(224, 128, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(128, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # Spatial context networks — g_sp^(k) in [He2022].
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # Parameter aggregation networks — "Param Aggregation" in [He2022].
        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                interp="linear",
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # Space-channel context model (SCCTX) — one codec per channel group.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate from a state dict.

        Automatically infers ``N`` and ``in_ch`` from the weight shapes so
        that both grayscale (in_ch=1) and RGB (in_ch=3) checkpoints load
        correctly without any manual argument passing.
        """
        N = state_dict["g_a.0.weight"].size(0)
        in_ch = state_dict["g_a.0.weight"].size(1)
        net = cls(N, in_ch=in_ch)
        net.load_state_dict(state_dict)
        return net
