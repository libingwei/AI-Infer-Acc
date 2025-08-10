#!/usr/bin/env python3
"""
Prepare ImageNet validation set for evaluation.
- Reorganize images into class subfolders (ILSVRC2012_img_val -> val/<class_name>/*)
- Generate labels CSV for trt_compare: filename,labelIndex

Note: This script expects you to have the following files locally:
- ILSVRC2012_img_val.tar (validation images)
- ILSVRC2012_devkit_t12.tar.gz (devkit with mapping)

You need sufficient disk space. This script is a helper; it does not download from the internet.
"""
import os
import tarfile
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL_TAR = ROOT / 'datasets' / 'ILSVRC2012_img_val.tar'
DEVKIT_TAR = ROOT / 'datasets' / 'ILSVRC2012_devkit_t12.tar.gz'
OUT_DIR = ROOT / 'imagenet_val'
CSV_PATH = ROOT / 'imagenet_val_labels.csv'

# Mapping helpers per devkit
# Will produce mapping from image filename -> label index (0..999)

def extract_devkit(devkit_tar, tmp_dir):
    with tarfile.open(devkit_tar, 'r:gz') as tar:
        tar.extractall(tmp_dir)
    # Locate ground truth labels and wnid mapping
    # files: data/ILSVRC2012_validation_ground_truth.txt (1..1000 labels)
    #        data/map_clsloc.txt (class index, wnid, words)
    gt_file = Path(tmp_dir) / 'ILSVRC2012_devkit_t12' / 'data' / 'ILSVRC2012_validation_ground_truth.txt'
    map_file = Path(tmp_dir) / 'ILSVRC2012_devkit_t12' / 'data' / 'map_clsloc.txt'
    with open(gt_file, 'r') as f:
        gt = [int(x.strip()) for x in f]
    wnids = []
    with open(map_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                wnid = parts[1]
                wnids.append((idx, wnid))
    wnids.sort(key=lambda x: x[0])
    # Build idx->wnid
    idx_to_wnid = {idx: wnid for idx, wnid in wnids}
    return gt, idx_to_wnid


def extract_val_images(val_tar, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(val_tar, 'r:') as tar:
        tar.extractall(out_dir)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / '_devkit'
    tmp_dir.mkdir(exist_ok=True)
    print('Extracting devkit...')
    gt, idx_to_wnid = extract_devkit(DEVKIT_TAR, tmp_dir)

    print('Extracting validation images...')
    val_dir = OUT_DIR / 'val'
    extract_val_images(VAL_TAR, val_dir)

    # The validation images are named ILSVRC2012_val_00000001.JPEG ... 50000
    # Ground truth labels are 1..1000 (1-based). We'll map to 0..999
    # And also write CSV filename,labelIndex
    print('Generating labels CSV...')
    with open(CSV_PATH, 'w', newline='') as cf:
        writer = csv.writer(cf)
        # Assume alphabetical order corresponds to ground truth file ordering
        files = sorted([p.name for p in val_dir.glob('*.JPEG')])
        if len(files) != len(gt):
            print('Warning: files count and ground truth count differ: ', len(files), len(gt))
        for i, fname in enumerate(files):
            idx1 = gt[i]  # 1..1000
            idx0 = idx1 - 1
            writer.writerow([fname, idx0])
    print('Wrote labels CSV to', CSV_PATH)

    print('Done. You can run:')
    print('  ./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt', val_dir, '200 --labels', CSV_PATH)

if __name__ == '__main__':
    main()
