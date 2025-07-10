import argparse

def configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', type=str, help='Path to COCO images folder', default="coco2017/images/train2017")
    parser.add_argument('--ann_file', type=str, help='Path to COCO annotation file', default="coco2017/annotations/instances_train2017.json")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--task', type=str, choices=['segmentation', 'instance', 'detection'], default='segmentation')
    parser.add_argument('--logf', type=int, default=50)
    parser.add_argument('--enable_logging', type=bool, default=True)
    parser.add_argument('--log_weights', type=bool, default=True)
    parser.add_argument('--log_grads', type=bool, default=False)
    args = parser.parse_args()
    return args
