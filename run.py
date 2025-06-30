import torch
import torch.nn as nn
from qtorch import BlockFloatingPoint
from qtorch.quant import Quantizer
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data import prepare_data
import wandb
from visualize import log_2d_histogram

# --- Quantization and Logging Hooks ---

def quantize_weights_hook(module, input):
    with torch.no_grad():
        if hasattr(module, "weight") and module.weight is not None and hasattr(module, "weight_quantizer"):
            # Save pre-quantized weights for visualization
            module.prequant_weight_values = module.weight.detach().cpu().view(-1)
            module.weight.copy_(module.weight_quantizer(module.weight))

def quantize_activations_hook(module, input, output):
    if hasattr(module, "act_quantizer"):
        # Save pre-quantized activations for visualization
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

# --- Quantized UNet with dynamic precision ---
class QuantizedUNetDeep(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, exp_bits=2, man_bits=1):
        super().__init__()
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        
        wl = self.exp_bits + self.man_bits + 1  # sign + exp + mantissa bit

        # quantizers applied to activation, weight and gradient as hooks
        # weight is forward prehook, applied before running forward method
        # activation is forward hook, applied after running forward method
        # grad is register hook, apply to a tensor rather than a module/layer

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

# --- Training Script ---
def train(val_every=1):
    wandb.init(project="fp4-unet-coco", name="unet-fp4-training")

    train_loader, val_loader, test_loader = prepare_data()  # Your data loader should provide these

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantizedUNetDeep(in_ch=3, out_ch=80, exp_bits=2, man_bits=1).to(device)

    # 🟢 Register quantization hooks for weights & activations
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            module.weight_quantizer = model.weight_quantizer
            module.act_quantizer = model.act_quantizer
            module.register_forward_pre_hook(quantize_weights_hook)
            module.register_forward_hook(quantize_activations_hook)

    # 🟢 Register gradient quantization hooks
    def quantize_gradients_hook(grad):
        return model.grad_quantizer(grad)

    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(quantize_gradients_hook)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        model.train()

        # Enable activation/weight logging for first batch of each epoch
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                module.log_activations_this_step = True

        running_loss = 0.0
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            print(imgs)
            print(imgs[0])
            imgs = imgs.to(device)
            dummy_masks = torch.randint(0, 2, (imgs.size(0), 80, imgs.shape[2], imgs.shape[3]), device=device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, dummy_masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Collect and log 2D histogram after first batch
            if batch_idx == 0:
                # Collect weights and activations separately
                weight_layer_indices, weight_values = [], []
                activation_layer_indices, activation_values = [], []
                for idx, module in enumerate(model.modules()):
                    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                        if hasattr(module, "prequant_weight_values"):
                            v = module.prequant_weight_values.numpy()
                            weight_layer_indices.extend([idx] * len(v))
                            weight_values.extend(v)
                        if hasattr(module, "prequant_activation_values"):
                            v = module.prequant_activation_values.numpy()
                            activation_layer_indices.extend([idx] * len(v))
                            activation_values.extend(v)

                if len(weight_values) > 0:
                    log_2d_histogram(
                        wandb=wandb,
                        wandb_key="Weights_2D_Histogram",
                        layer_indices=weight_layer_indices,
                        values=weight_values,
                        title="2D Histogram of Weights",
                        ylabel="Pre-Quantized Weight Value",
                        epoch=epoch,
                    )

                if len(activation_values) > 0:
                    log_2d_histogram(
                        wandb=wandb,
                        wandb_key="Activations_2D_Histogram",
                        layer_indices=activation_layer_indices,
                        values=activation_values,
                        title="2D Histogram of Activations",
                        ylabel="Pre-Quantized Activation Value",
                        epoch=epoch,
                    )

                # Disable logging after first batch
                for module in model.modules():
                    if hasattr(module, "log_activations_this_step"):
                        module.log_activations_this_step = False

        avg_train_loss = running_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
        print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.4f}")

        if (epoch + 1) % val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    dummy_masks = torch.randint(0, 2, (imgs.size(0), 80, imgs.shape[2], imgs.shape[3]), device=device, dtype=torch.float)
                    outputs = model(imgs)
                    loss = criterion(outputs, dummy_masks)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            wandb.log({"epoch": epoch, "val_loss": avg_val_loss})
            print(f"Epoch {epoch} Avg Validation Loss: {avg_val_loss:.4f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            dummy_masks = torch.randint(0, 2, (imgs.size(0), 80, imgs.shape[2], imgs.shape[3]), device=device, dtype=torch.float)
            outputs = model(imgs)
            loss = criterion(outputs, dummy_masks)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    wandb.log({"final_test_loss": avg_test_loss})
    print(f"Final Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    train()
