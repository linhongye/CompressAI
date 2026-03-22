# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


@register_criterion("ResidualAwareRDLoss")
class ResidualAwareRDLoss(nn.Module):
    """Rate-distortion loss using L1 distortion as a proxy for residual compressibility.

    Under a Laplacian residual model the optimal estimator minimises L1 (not MSE).
    The scaling factor is 255 (not 255**2) because L1 is first-order in pixel values.
    """

    def __init__(self, lmbda=0.01, return_type="all"):
        super().__init__()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["l1_loss"] = F.l1_loss(output["x_hat"], target)
        out["residual_l1"] = out["l1_loss"]
        out["mse_loss"] = F.mse_loss(output["x_hat"], target)

        distortion = 255 * out["l1_loss"]
        out["loss"] = self.lmbda * distortion + out["bpp_loss"]

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]


@register_criterion("TotalBppLoss")
class TotalBppLoss(nn.Module):
    """Directly minimize total BPP = neural BPP + estimated residual entropy.

    Uses the Laplacian entropy formula H = log2(2*e*b) where b = MAE (in uint8
    scale) as the residual BPP estimate.  Both terms are in bpp units so no
    Lagrangian lambda is needed.  Includes STE (Straight-Through Estimator)
    quantization to account for the uint8 rounding that happens at inference.
    """

    _EPSILON = 1e-6

    def __init__(self, return_type="all"):
        super().__init__()
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        x_hat = output["x_hat"]
        x_hat_q = (x_hat * 255).round() / 255
        x_hat_q = x_hat + (x_hat_q - x_hat).detach()

        residual_abs = (target - x_hat_q).abs() * 255
        mae = residual_abs.mean()
        out["bpp_residual_est"] = torch.log2(2 * math.e * (mae + self._EPSILON))

        out["loss"] = out["bpp_loss"] + out["bpp_residual_est"]

        out["l1_loss"] = mae / 255
        out["mse_loss"] = F.mse_loss(x_hat, target)

        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
