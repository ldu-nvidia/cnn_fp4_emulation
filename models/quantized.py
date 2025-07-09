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

# --- UNet Blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)

# --- Quantized UNet Model ---
class QuantizedUNetDeep(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, exp_bits=2, man_bits=1):
        super().__init__()
        self.exp_bits = exp_bits
        self.man_bits = man_bits

        wl = self.exp_bits + self.man_bits + 1  # sign + exponent + mantissa bits

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

        # Encoder
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

        # Register quantization hooks
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                module.weight_quantizer = self.weight_quantizer
                module.act_quantizer = self.act_quantizer
                module.register_forward_pre_hook(quantize_weights_hook)
                module.register_forward_hook(quantize_activations_hook)

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