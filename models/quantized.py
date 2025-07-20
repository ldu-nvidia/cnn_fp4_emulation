import torch
import torch.nn as nn
from qtorch import BlockFloatingPoint
from qtorch.quant import Quantizer

# --- Quantization Hooks ---
def quantize_weights_hook(module, input):
    with torch.no_grad():
        if hasattr(module, "weight") and module.weight is not None and hasattr(module, "weight_quantizer"):
            module.prequant_weight_values = module.weight.detach().cpu().view(-1)
            module.weight.copy_(module.weight_quantizer(module.weight))

def quantize_activations_hook(module, input, output):
    if hasattr(module, "act_quantizer"):
        module.prequant_activation_values = output.detach().cpu().view(-1)
        quantized_output = module.act_quantizer(output)
        return quantized_output
    return output

# --- Gradient hook ----
def get_gradient_hook(quantizer):
    """Return a hook that quantizes gradients during backward pass."""

    def _hook(grad):
        return quantizer(grad)

    return _hook

# --- UNet Blocks ---
class ConvBlock(nn.Module):
    """Same conv–GroupNorm–ReLU stack used in baseline UNetFP16."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)

# --- Quantized UNet Model ---
class QuantizedUNetDeep(nn.Module):
    """Quantized version of UNetFP16 with identical topology (scaled channels)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, exp_bits: int = 2, man_bits: int = 1):
        super().__init__()

        self.exp_bits = exp_bits
        self.man_bits = man_bits
        wl = self.exp_bits + self.man_bits + 1  # sign + exponent + mantissa bits

        # Channel scaling identical to baseline (0.25)
        self.scale = 0.25

        self.act_quantizer = Quantizer(
            forward_number=BlockFloatingPoint(wl=wl),
            backward_number=BlockFloatingPoint(wl=wl),
            forward_rounding="stochastic"
        )
        self.weight_quantizer = Quantizer(
            forward_number=BlockFloatingPoint(wl=wl),
            backward_number=BlockFloatingPoint(wl=wl),
            forward_rounding="stochastic"
        )
        self.grad_quantizer = Quantizer(
            forward_number=BlockFloatingPoint(wl=wl),
            backward_number=BlockFloatingPoint(wl=wl),
            forward_rounding="stochastic"
        )

        s = self.scale

        def ch(x: int) -> int:
            return int(s * x)

        # Encoder -------------------------------------------------------
        self.enc1 = ConvBlock(in_ch,  ch(64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(ch(64),  ch(128))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(ch(128), ch(256))
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(ch(256), ch(512))
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck ----------------------------------------------------
        self.bottleneck = ConvBlock(ch(512), ch(1024))

        # Decoder -------------------------------------------------------
        self.up4 = nn.ConvTranspose2d(ch(1024), ch(512), 2, stride=2)
        self.dec4 = ConvBlock(ch(1024),  ch(512))

        self.up3 = nn.ConvTranspose2d(ch(512),  ch(256), 2, stride=2)
        self.dec3 = ConvBlock(ch(512),  ch(256))

        self.up2 = nn.ConvTranspose2d(ch(256),  ch(128), 2, stride=2)
        self.dec2 = ConvBlock(ch(256),  ch(128))

        self.up1 = nn.ConvTranspose2d(ch(128),  ch(64), 2, stride=2)
        self.dec1 = ConvBlock(ch(128),  ch(64))

        self.final = nn.Conv2d(ch(64), out_ch, 1)

        # Register quantization hooks
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                module.weight_quantizer = self.weight_quantizer
                module.act_quantizer = self.act_quantizer
                module.register_forward_pre_hook(quantize_weights_hook)
                module.register_forward_hook(quantize_activations_hook)

        # Register gradient hooks on all learnable parameters
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(get_gradient_hook(self.grad_quantizer))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        x_b = self.bottleneck(self.pool4(x4))

        x = self.up4(x_b)
        x = self.dec4(torch.cat([x, x4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.final(x)