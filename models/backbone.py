from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet101_Weights
from torchvision.models.resnet import Bottleneck, conv1x1


class FrozenBatchNorm2d(nn.Module):
    """Batch normalization with fixed running stats and trainable affine terms."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(0, dtype=torch.long),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(
            inputs,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


class TruncatedResNet101(nn.Module):
    """The ResNet101 stem through layer2, exposing three feature taps."""

    tap_channels = {"f0": 64, "f1": 256, "f2": 512}

    def __init__(
        self,
        weights: ResNet101_Weights | None = ResNet101_Weights.IMAGENET1K_V2,
        output_stride: int = 8,
        *,
        progress: bool = True,
        weights_path: Optional[Path] = None,
        trainable_stages: Sequence[str] = ("layer2",),
    ) -> None:
        super().__init__()
        if output_stride not in (4, 8):
            raise ValueError(f"output_stride must be 4 or 8, got {output_stride}")

        self.output_stride = output_stride
        self._norm_layer: Callable[[int], nn.Module] = FrozenBatchNorm2d
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, planes=64, blocks=3)
        self.layer2 = self._make_layer(
            Bottleneck,
            planes=128,
            blocks=4,
            stride=2,
            dilate=output_stride == 4,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

        self._load_pretrained_weights(weights, weights_path, progress)
        self._set_trainable_stages(trainable_stages)

    def _load_pretrained_weights(
        self,
        weights: ResNet101_Weights | None,
        weights_path: Optional[Path],
        progress: bool,
    ) -> None:
        if weights_path is not None:
            loaded = torch.load(str(weights_path), map_location="cpu")
            if isinstance(loaded, Mapping) and "state_dict" in loaded:
                loaded = loaded["state_dict"]
            if not isinstance(loaded, Mapping):
                raise ValueError("weights_path must contain a state-dict mapping")
            pretrained = {
                (str(key)[7:] if str(key).startswith("module.") else str(key)): value
                for key, value in loaded.items()
            }
        else:
            verified_weights = ResNet101_Weights.verify(weights)
            if verified_weights is None:
                return
            pretrained = verified_weights.get_state_dict(progress=progress, check_hash=True)
        own_keys = self.state_dict().keys()
        missing = [key for key in own_keys if key not in pretrained]
        if missing:
            raise ValueError("pretrained weights are missing keys: {}".format(missing[:3]))
        self.load_state_dict({key: pretrained[key] for key in own_keys})

    def _set_trainable_stages(self, trainable_stages: Sequence[str]) -> None:
        allowed = ("conv1", "bn1", "layer1", "layer2")
        requested = tuple(str(stage) for stage in trainable_stages)
        unknown = sorted(set(requested).difference(allowed))
        if unknown:
            raise ValueError("unknown trainable stages: {}".format(unknown))
        self.trainable_stages = requested
        trainable = set(requested)
        for stage_name in allowed:
            getattr(self, stage_name).requires_grad_(stage_name in trainable)

    def _make_layer(
        self,
        block: type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.conv1(inputs)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        f0 = features
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        return {"f0": f0, "f1": f1, "f2": f2}
