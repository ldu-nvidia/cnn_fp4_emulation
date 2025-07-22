import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "brevitas" / "src"))
import torch
import torch.nn as nn
from typing import List, Optional
import brevitas

from brevitas.nn import QuantConv2d, QuantConvTranspose2d, QuantIdentity
from brevitas.quant.fixed_point import (
    Int8ActPerTensorFixedPoint,
)
from brevitas.quant.fixed_point import Int4WeightPerTensorFixedPointDecoupled as Int4WeightPerTensorFixedPoint
from brevitas.inject.enum import QuantType


# Brevitas ships only weight FP4 quantizers; define a thin activation wrapper

class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    """4-bit activation quantizer based on the Int8 fixed-point implementation."""

    bit_width = 4


class ConvBlock(nn.Module):
    """2×(Conv→GN→ReLU) with optional 4-bit quantisation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
        quant: bool = True,
    ) -> None:
        super().__init__()

        Conv = QuantConv2d if quant else nn.Conv2d
        w_qkwargs = dict(
            weight_bit_width=4,
            weight_quant=Int4WeightPerTensorFixedPoint,
            quant_type=QuantType.INT,
        ) if quant else {}
        a_q = QuantIdentity(
            act_quant=Int4ActPerTensorFixedPoint,
            act_bit_width=4,
            quant_type=QuantType.INT,
        ) if quant else nn.Identity()

        self.block = nn.Sequential(
            Conv(in_ch, out_ch, kernel_size=3, padding=1, **w_qkwargs),  # type: ignore[arg-type]
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=False),
            Conv(out_ch, out_ch, kernel_size=3, padding=1, **w_qkwargs),  # type: ignore[arg-type]
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=False),
            a_q,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class QuantUNetFP4(nn.Module):
    """UNet identical to baseline but with optional FP4 quantisation per layer.

    Parameters
    ----------
    task : str
        "semantic" | "instance"
    quantized_layers : Optional[List[str]]
        Names (e.g. "enc1", "dec3", "final") to quantise.  If `None`, all
        conv/transpose-conv layers are quantised.
    """

    def __init__(
        self,
        task: str,
        in_channels: int = 3,
        out_channels: int = 80,
        num_instances: int = 10,
        quantized_layers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.scale = 0.25
        self.quantized_layers = quantized_layers

        def ch(v: int) -> int:
            return int(self.scale * v)

        def q(layer_name: str) -> bool:
            return quantized_layers is None or layer_name in quantized_layers

        # Encoder ----------------------------------------------------------------
        self.enc1 = ConvBlock(in_channels, ch(64), quant=q("enc1"))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(ch(64), ch(128), quant=q("enc2"))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(ch(128), ch(256), quant=q("enc3"))
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(ch(256), ch(512), quant=q("enc4"))
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch(512), ch(1024), quant=q("bottleneck"))

        # Decoder -----------------------------------------------------------------
        UpConv = lambda qflag, in_c, out_c: (
            QuantConvTranspose2d(
                in_c,
                out_c,
                kernel_size=2,
                stride=2,
                weight_bit_width=4,
                weight_quant=Int4WeightPerTensorFixedPoint,
                quant_type=QuantType.INT,
            )
            if qflag
            else nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        )

        self.upconv4 = UpConv(q("upconv4"), ch(1024), ch(512))
        self.dec4 = ConvBlock(ch(1024), ch(512), quant=q("dec4"))

        self.upconv3 = UpConv(q("upconv3"), ch(512), ch(256))
        self.dec3 = ConvBlock(ch(512), ch(256), quant=q("dec3"))

        self.upconv2 = UpConv(q("upconv2"), ch(256), ch(128))
        self.dec2 = ConvBlock(ch(256), ch(128), quant=q("dec2"))

        self.upconv1 = UpConv(q("upconv1"), ch(128), ch(64))
        self.dec1 = ConvBlock(ch(128), ch(64), quant=q("dec1"))

        # Output head -------------------------------------------------------------
        FinalConv = (
            QuantConv2d
            if q("final")
            else nn.Conv2d
        )
        w_kwargs = (
            dict(
                weight_bit_width=4,
                weight_quant=Int4WeightPerTensorFixedPoint,
                quant_type=QuantType.INT,
            )
            if q("final")
            else {}
        )

        if task == "semantic":
            assert out_channels != -1
            self.out_conv = FinalConv(ch(64), out_channels, kernel_size=1, **w_kwargs)
        elif task == "instance":
            assert num_instances != -1
            self.out_conv = FinalConv(ch(64), num_instances, kernel_size=1, **w_kwargs)
        else:
            raise ValueError("Unsupported task type")

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)
