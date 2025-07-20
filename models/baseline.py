import torch
import torch.nn as nn

# Helper: FP16-safe Conv Block with GroupNorm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        # groups x groups is the number of pixels used to calculate statistics
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)

# UNet model with GroupNorm and AMP-safe layers
class UNetFP16(nn.Module):
    def __init__(self, task, in_channels=3, out_channels=80, num_instances=10):
        super().__init__()
        self.scale = 0.25

        # convenience: apply scaling factor exactly as in quantized model
        def ch(x: int) -> int:  # local helper
            return int(self.scale * x)

        # Encoder --------------------------------------------------------
        self.enc1 = ConvBlock(in_channels, ch(64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(ch(64), ch(128))
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(ch(128), ch(256))
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(ch(256), ch(512))
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch(512), ch(1024))

        self.upconv4 = nn.ConvTranspose2d(ch(1024), ch(512), kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch(1024), ch(512))

        self.upconv3 = nn.ConvTranspose2d(ch(512), ch(256), kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch(512), ch(256))

        self.upconv2 = nn.ConvTranspose2d(ch(256), ch(128), kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch(256), ch(128))

        self.upconv1 = nn.ConvTranspose2d(ch(128), ch(64), kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch(128), ch(64))

        if task == "semantic":
            assert out_channels != -1, "out_channels must be specified for semantic task"
            self.out_conv = nn.Conv2d(ch(64), out_channels, kernel_size=1)
        elif task == "instance":
            assert num_instances != -1, "num_instances must be specified for instance task"
            self.out_conv = nn.Conv2d(ch(64), num_instances, kernel_size=1)
        else:
            raise ValueError("Unsupported task type")

    def forward(self, x):
        # Do NOT cast to FP16 manually — use autocast externally during training
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
