import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from pycocotools import mask as coco_mask
from PIL import Image

class CocoSegmentationDataset(Dataset):
    def __init__(self, img_root, ann_file, category_id_to_class_idx, target_size=(256, 256)):
        self.base = CocoDetection(root=img_root, annFile=ann_file)
        self.category_id_to_class_idx = category_id_to_class_idx
        self.target_size = target_size
        self.transform_img = T.Compose([
            T.Resize(self.target_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, annotations = self.base[idx]
        orig_w, orig_h = image.size            # original dims

        # attach original size to each annotation
        for ann in annotations:
            ann["orig_height"] = orig_h
            ann["orig_width"]  = orig_w

        image = self.transform_img(image)      # resized image-tensor
        return image, annotations
