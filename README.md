# CNN FP4 Emulation 🧪🚀

Welcome to **cnn_fp4_emulation** – a playground for exploring **4-bit floating-point (FP4) quantisation** on convolutional neural networks while keeping the whole training loop fully differentiable.  Two flavours of UNet live here:

| Model | Precision | Quantisation Path |
|-------|-----------|-------------------|
| `UNetFP16` | FP32 → FP16 autocast | No quantisation – a high-precision baseline ✅ |
| `UNetNVFP4` | FP32 → FP4 (E2M1) emulated | Kitchen 🔪 autograd fake-quant on every Conv/ConvT layer (GroupNorm & output layer stay full-precision) |

The repository is set up to **train both models sequentially** and log a *ton* of telemetry to Weights & Biases:

🟢 Raw FP weights  
🟣 Int-encoded FP4 weights  
🔵 De-quantised FP4 weights  
🟡 Gradients

Everything is saved hierarchically under `plots/heatmaps/<model>/…` so runs never overwrite each other.

---
## Quick Start ⚡
```bash
cd cnn_fp4_emulation
python -m venv env && source env/bin/activate
pip install -r requirements.txt  # make sure torch & wandb are present

# Train both models on GPU 4 with 0.25 channel scaling
python main.py \
  --models fp16 nvfp4 \
  --model_scale_factor 0.25 \
  --logf 50  # log every 50 steps to keep W&B tidy
```

Check out the resulting artefacts:
```
plots/heatmaps/
├── fp16/
│   ├── weights/
│   │   └── prequantize/semantic_weights.{png,html,json}
│   └── gradients/semantic_gradients.{png,html,json}
└── nvfp4/
    ├── weights/
    │   ├── prequantize/
    │   ├── quantized/
    │   └── dequantized/
    └── gradients/
```

---
## What We Learned 📚
1. **Quantisation granularity matters.**  Per-tile power-of-two scaling drastically increases the number of FP4 codes actually used.
2. **GroupNorm can stay full precision** without degrading the quantised model.
3. **Logging the whole pipeline** (raw → int → de-q) reveals hidden bottlenecks that aren’t obvious from accuracy alone.

---
## Roadmap ✨
- [ ] Plug in **mixed-precision gradient scaling** for the quantised path.  
- [ ] Add **activation quantisation histograms** alongside weights.  
- [ ] Experiment with **learnable scaling factors** instead of power-of-two.  
- [ ] Integrate **CUTLASS CuTe kernels 🐱** for a speed boost.  
- [ ] Extend to **object detection** tasks (the trainer already supports it!).

PRs & issues welcome – let’s push FP4 to its limits! 🤖💾 