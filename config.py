import argparse
from typing import Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Data parameters
    coco_root: str = "coco2017/train2017"
    ann_file: str = "coco2017/annotations/instances_train2017.json"
    
    # Training parameters
    batch_size: int = 16
    lr: float = 5e-5
    epochs: int = 5
    task: str = "semantic" 

    wandb_name: str = "5_epochs_full_run_2" 
    project_name: str = "semantic_segmentation_full_precision"

    # Logging parameters
    logf: int = 10
    enable_logging: bool = True
    log_weights: bool = True
    log_grads: bool = True
    visualize_val: bool = True
    seeds: int = 892
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
    
    # Logging parameters
    parser.add_argument('--logf', type=int, default=TrainingConfig.logf)
    parser.add_argument('--enable_logging', action='store_true', default=TrainingConfig.enable_logging)
    parser.add_argument('--log_weights', action='store_true', default=TrainingConfig.log_weights)
    parser.add_argument('--log_grads', action='store_true', default=TrainingConfig.log_grads)
    parser.add_argument('--visualize_val', action='store_true', default=TrainingConfig.visualize_val)

    
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
        debug=args.debug
    )

# For backward compatibility
def configs():
    """Legacy function that returns parsed configuration."""
    return parse_args()
