import torch
import torch.nn as nn
from qtorch import BlockFloatingPoint
from qtorch.quant import Quantizer
from torch.utils.data import DataLoader
import torch.optim as optim
from data import prepare_data
import wandb

# Create quantizers simulating FP4 behavior with shared exponent
weight_quantizer = Quantizer(
    forward_number=BlockFloatingPoint(wl=4),
    backward_number=BlockFloatingPoint(wl=4),
    forward_rounding="stochastic"
)
act_quantizer = Quantizer(
    forward_number=BlockFloatingPoint(wl=4),
    backward_number=BlockFloatingPoint(wl=4),
    forward_rounding="stochastic"
)

# UNet Encoder Block
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

# Quantized UNet with deeper architecture & skip connections
class QuantizedUNetDeep(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, exp_bits=2, man_bits=1):
        super().__init__()
        self.exp_bits = exp_bits
        self.man_bits = man_bits

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
        
    def forward(self, x):
        # Encoder
        x1 = act_quantizer(self.enc1(x))
        x2 = act_quantizer(self.enc2(self.pool1(x1)))
        x3 = act_quantizer(self.enc3(self.pool2(x2)))
        x4 = act_quantizer(self.enc4(self.pool3(x3)))
        
        # Bottleneck
        x_b = act_quantizer(self.bottleneck(self.pool4(x4)))
        
        # Decoder with skip connections
        x = self.up4(x_b)
        x = act_quantizer(self.dec4(torch.cat([x, x4], dim=1)))
        
        x = self.up3(x)
        x = act_quantizer(self.dec3(torch.cat([x, x3], dim=1)))
        
        x = self.up2(x)
        x = act_quantizer(self.dec2(torch.cat([x, x2], dim=1)))
        
        x = self.up1(x)
        x = act_quantizer(self.dec1(torch.cat([x, x1], dim=1)))
        
        x = act_quantizer(self.final(x))
        return x

# Standalone train function instead of broken class
def train():
    wandb.init(project="fp4-unet-coco", name="unet-fp4-training")

    # 🟢 Prepare dataset
    train_ds = prepare_data()  # Make sure this function returns a torch Dataset object
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    # 🟢 Setup model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantizedUNetDeep(in_ch=3, out_ch=80, exp_bits=2, man_bits=1).to(device)  # COCO has 80 classes
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()  # Simplified; real detection uses more complex loss.

    # 🟢 Training loop
    model.train()
    for epoch in range(5):  # Adjust epochs as needed
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            # Dummy masks: in real detection, generate target masks from bbox annotations.
            dummy_masks = torch.randint(0, 2, (imgs.size(0), 80, imgs.shape[2], imgs.shape[3]), device=device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, dummy_masks)
            loss.backward()
            optimizer.step()
            wandb.log({"epoch": epoch, "train_loss": loss.item()})
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
