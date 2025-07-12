import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Configuration ---
img_root = "coco2017/images/train2017"
ann_file = "coco2017/annotations/instances_train2017.json"
output_sem_path = "semantic_vis.png"
output_inst_path = "instance_vis.png"

# --- Load COCO annotations ---
coco = COCO(ann_file)
img_ids = coco.getImgIds()
img_id = img_ids[100]  # just pick the first image for demo
img_info = coco.loadImgs(img_id)[0]

# --- Load image ---
img_path = os.path.join(img_root, img_info['file_name'])
image = Image.open(img_path).convert("RGB")
width, height = image.size
image_np = np.array(image)

# --- Load annotations ---
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)

# --- Semantic mask ---
sem_mask = np.zeros((height, width), dtype=np.uint8)
# --- Instance mask (color-coded by instance id) ---
inst_mask = np.zeros((height, width), dtype=np.int32)

for idx, ann in enumerate(anns):
    cat_id = ann["category_id"]
    inst_id = ann["id"]
    if ann.get("iscrowd", 0):
        rle = ann["segmentation"]
    else:
        rle = coco_mask.frPyObjects(ann["segmentation"], height, width)

    binary_mask = coco_mask.decode(rle)
    if binary_mask.ndim == 3:
        binary_mask = np.any(binary_mask, axis=2)

    sem_mask[binary_mask > 0] = cat_id
    inst_mask[binary_mask > 0] = inst_id  # unique per instance

# --- Save semantic segmentation visualization ---
norm_sem_mask = sem_mask.astype(float)
norm_sem_mask /= max(1, norm_sem_mask.max())  # normalize to [0,1]
sem_color = (cm.nipy_spectral(norm_sem_mask)[:, :, :3] * 255).astype(np.uint8)
sem_image = Image.fromarray(sem_color)
sem_image.save(output_sem_path)

# --- Save instance segmentation visualization ---
rand_colors = np.random.randint(0, 255, (inst_mask.max() + 1, 3), dtype=np.uint8)
inst_rgb = rand_colors[inst_mask]
overlay = (0.5 * image_np + 0.5 * inst_rgb).astype(np.uint8)
inst_image = Image.fromarray(overlay)
inst_image.save(output_inst_path)

print(f"✅ Saved semantic mask to {output_sem_path}")
print(f"✅ Saved instance overlay to {output_inst_path}")
