import os
import subprocess
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def prepare_data():
    # Where to store the COCO data locally
    COCO_DIR = "./coco2017"
    IMG_DIR = os.path.join(COCO_DIR, "images/train2017")
    ANN_DIR = os.path.join(COCO_DIR, "annotations")

    os.makedirs(COCO_DIR, exist_ok=True)

    # Download COCO train2017 images if not already present
    if not os.path.exists(IMG_DIR):
        print("Downloading COCO train2017 images...")
        subprocess.run([
            "wget",
            "http://images.cocodataset.org/zips/train2017.zip",
            "-P",
            COCO_DIR
        ])
        print("Extracting train2017.zip...")
        subprocess.run([
            "unzip",
            os.path.join(COCO_DIR, "train2017.zip"),
            "-d",
            os.path.join(COCO_DIR, "images")
        ])

    # Download COCO 2017 annotations if not already present
    if not os.path.exists(ANN_DIR):
        print("Downloading COCO annotations...")
        subprocess.run([
            "wget",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "-P",
            COCO_DIR
        ])
        print("Extracting annotations_trainval2017.zip...")
        subprocess.run([
            "unzip",
            os.path.join(COCO_DIR, "annotations_trainval2017.zip"),
            "-d",
            COCO_DIR
        ])



    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    dataset = CocoDetection(
        root=IMG_DIR,
        annFile=os.path.join(COCO_DIR, "annotations/instances_train2017.json"),
        transform=transform
    )

    # Split ratios
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader, test_loader

