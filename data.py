import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from pycocotools import mask as coco_mask
from PIL import Image
import os
from typing import List, Optional, Tuple, Dict

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


class MVTecSegmentationDataset(Dataset):
    """Supervised segmentation dataset for MVTec AD-style directory layout.

    Assumes a root directory structure like:
        root/
          category_a/
            train/good/*.png
            test/good/*.png
            test/<defect_type>/*.png
            ground_truth/<defect_type>/*_mask.png

    Behaviour:
      - subset in {"train", "test", "all"}
      - task_mode in {"binary", "per-defect"}
          • binary: background=0, defect(any)=1 (num_classes=2)
          • per-defect: background=0, each (category/defect_type)=1..K

    Returns:
      image: FloatTensor [3,H,W] in [0,1]
      target: LongTensor [H,W] with class ids per pixel
    """

    def __init__(
        self,
        root_dir: str,
        subset: str = "train",
        categories: Optional[List[str]] = None,
        task_mode: str = "binary",
        target_size: Tuple[int, int] = (256, 256),
        image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ):
        assert os.path.isdir(root_dir), f"Root dir not found: {root_dir}"
        assert subset in {"train", "test", "all"}
        assert task_mode in {"binary", "per-defect"}
        self.root_dir = root_dir
        self.subset = subset
        self.task_mode = task_mode
        self.target_size = target_size
        self.image_exts = image_exts

        # Determine categories
        all_categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d not in {"license.txt", "readme.txt"}]
        if categories is None:
            self.categories = sorted([d for d in all_categories if os.path.isdir(os.path.join(root_dir, d))])
        else:
            for c in categories:
                if not os.path.isdir(os.path.join(root_dir, c)):
                    raise FileNotFoundError(f"Category not found under root: {c}")
            self.categories = sorted(categories)

        # Index samples
        self.samples: List[Tuple[str, Optional[str], str, Optional[str]]] = []
        # each entry: (image_path, mask_path_or_None, category, defect_key_or_None)

        for cat in self.categories:
            cat_dir = os.path.join(root_dir, cat)

            def is_image(fname: str) -> bool:
                return os.path.splitext(fname)[1].lower() in self.image_exts

            def collect_from_subdir(img_dir: str, defect_type: Optional[str]):
                if not os.path.isdir(img_dir):
                    return
                for fname in sorted(os.listdir(img_dir)):
                    if not is_image(fname):
                        continue
                    img_path = os.path.join(img_dir, fname)
                    mask_path = None
                    defect_key = None
                    if defect_type and defect_type != "good":
                        # map image file e.g. 000.png -> ground_truth/<defect_type>/000_mask.png
                        base, _ = os.path.splitext(fname)
                        mask_fname = f"{base}_mask.png"
                        mask_dir = os.path.join(cat_dir, "ground_truth", defect_type)
                        mask_path = os.path.join(mask_dir, mask_fname)
                        if not os.path.isfile(mask_path):
                            # Some datasets may use .bmp masks or missing masks for good images
                            alt = os.path.join(mask_dir, f"{base}_mask.bmp")
                            mask_path = alt if os.path.isfile(alt) else None
                        defect_key = f"{cat}/{defect_type}"
                    self.samples.append((img_path, mask_path, cat, defect_key))

            if self.subset in {"train", "all"}:
                collect_from_subdir(os.path.join(cat_dir, "train", "good"), defect_type=None)
            if self.subset in {"test", "all"}:
                test_dir = os.path.join(cat_dir, "test")
                if os.path.isdir(test_dir):
                    for defect_type in sorted(os.listdir(test_dir)):
                        collect_from_subdir(os.path.join(test_dir, defect_type), defect_type=defect_type)

        # Build class mapping for per-defect
        if self.task_mode == "per-defect":
            defect_keys = sorted({dk for (_, m, _, dk) in self.samples if dk is not None})
            self.class_to_index: Dict[str, int] = {"background": 0}
            for i, dk in enumerate(defect_keys, start=1):
                self.class_to_index[dk] = i
            self.num_classes = 1 + len(defect_keys)
        else:
            self.class_to_index = {"background": 0, "defect": 1}
            self.num_classes = 2

        # Transforms
        self.transform_img = T.Compose([
            T.Resize(self.target_size, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        # Note: masks use nearest-neighbour to preserve labels
        self.transform_mask = T.Resize(self.target_size, interpolation=InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, mask_path: Optional[str], defect_key: Optional[str], size_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = size_hw
        if self.task_mode == "per-defect":
            target = torch.zeros((H, W), dtype=torch.long)
            if mask_path is not None and defect_key is not None and os.path.isfile(mask_path):
                m = Image.open(mask_path).convert("L")
                m = self.transform_mask(m)  # [H,W]
                m = torch.from_numpy(np.array(m)).long()
                m_bin = (m > 0).long()
                target[m_bin.bool()] = self.class_to_index[defect_key]
            return target
        else:  # binary
            if mask_path is None or not os.path.isfile(mask_path):
                return torch.zeros((H, W), dtype=torch.long)
            m = Image.open(mask_path).convert("L")
            m = self.transform_mask(m)
            m = torch.from_numpy(np.array(m)).long()
            m_bin = (m > 0).long()
            return m_bin  # 0 background, 1 defect

    def __getitem__(self, idx: int):
        img_path, mask_path, _, defect_key = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image_t = self.transform_img(image)
        H, W = self.target_size
        target = self._load_mask(mask_path, defect_key, (H, W))
        return image_t, target
