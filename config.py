import argparse
from typing import Optional, List
from dataclasses import dataclass
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # keeps numbering stable
os.environ["CUDA_VISIBLE_DEVICES"] = "4"        # hide others *before* torch import

import torch

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Data parameters
    coco_root: str = "coco2017/train2017"
    ann_file: str = "coco2017/annotations/instances_train2017.json"
    
    # Training parameters
    batch_size: int = 32
    lr: float = 5e-5
    epochs: int = 2
    task: str = "semantic" 

    wandb_name: str = "1st full run" 
    project_name: str = "semantic_segmentation_full_vs_quantized"

    # Model parameters
    models: List[str] = ("nvfp4", "fp16")  # list of model keys to train sequentially

    # Logging parameters
    logf: int = 10
    enable_logging: bool = True
    log_weights: bool = True
    log_grads: bool = True
    visualize_val: bool = True
    seeds: int = 715

    # Architecture parameter
    model_scale_factor: float = 0.25  # scales base channel counts
    # Debug parameters
    debug: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.task not in ['semantic', 'instance', 'detection']:
            raise ValueError(f"Invalid task: {self.task}. Must be one of: semantic, instance, detection")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {self.batch_size}")
        
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got: {self.lr}")

        if self.model_scale_factor <= 0 or self.model_scale_factor > 1:
            raise ValueError("model_scale_factor must be in (0,1].")

        # Validate model choices here to fail fast.
        valid_models = ["fp16", "nvfp4"]
        for m in self.models:
            if m not in valid_models:
                raise ValueError(f"Invalid model: {m}. Must be one of: {', '.join(valid_models)}")

def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig object."""
    parser = argparse.ArgumentParser(description="Training configuration for CNN FP4 emulation")
    
    # Data parameters
    parser.add_argument('--coco_root', type=str, 
                       help='Path to COCO images folder', 
                       default=TrainingConfig.coco_root)
    parser.add_argument('--ann_file', type=str, 
                       help='Path to COCO annotation file', 
                       default=TrainingConfig.ann_file)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.batch_size)
    parser.add_argument('--lr', type=float, default=TrainingConfig.lr)
    parser.add_argument('--epochs', type=int, default=TrainingConfig.epochs)
    parser.add_argument('--task', type=str, 
                       choices=['semantic', 'instance', 'detection'], 
                       default=TrainingConfig.task)

    # Model parameters (accept multiple)
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['nvfp4', 'fp16'],
                        default=list(TrainingConfig.models),
                        help='List of UNet backbones to train sequentially')
    
    # Logging parameters
    parser.add_argument('--logf', type=int, default=TrainingConfig.logf)
    parser.add_argument('--enable_logging', action='store_true', default=TrainingConfig.enable_logging)
    parser.add_argument('--log_weights', action='store_true', default=TrainingConfig.log_weights)
    parser.add_argument('--log_grads', action='store_true', default=TrainingConfig.log_grads)
    parser.add_argument('--visualize_val', action='store_true', default=TrainingConfig.visualize_val)

    # Model architecture scale
    parser.add_argument('--model_scale_factor', type=float, default=TrainingConfig.model_scale_factor,
                        help='Multiply base channel counts by this factor (0<sf<=1).')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true', default=TrainingConfig.debug,
                       help="debug mode, terminate early to make sure everything works")
    
    args = parser.parse_args()
    
    return TrainingConfig(
        coco_root=args.coco_root,
        ann_file=args.ann_file,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        task=args.task,
        logf=args.logf,
        enable_logging=args.enable_logging,
        log_weights=args.log_weights,
        log_grads=args.log_grads,
        visualize_val=args.visualize_val,
        debug=args.debug,
        models=args.models,
        model_scale_factor=args.model_scale_factor
    )

# For backward compatibility
def configs():
    """Legacy function that returns parsed configuration."""
    return parse_args()
