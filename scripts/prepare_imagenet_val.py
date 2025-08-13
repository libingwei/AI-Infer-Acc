#!/usr/bin/env python3
"""
Prepare ImageNet validation set for evaluation.
- Generate labels CSV for trt_compare: filename,labelIndex
- Support extracting only the first N images to save space (default 1000)
- Optionally reorganize images into class subfolders (val/<wnid>/...) when --reorg is provided

Expected local files (placed under datasets/):
- ILSVRC2012_img_val.tar (validation images)
- ILSVRC2012_devkit_t12.tar.gz (devkit with mapping: ground-truth and meta.mat)

This script does not download from the internet.
"""
import os
import argparse
import shutil
import tarfile
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pkgutil

ROOT = Path(__file__).resolve().parent.parent
VAL_TAR = ROOT / 'datasets' / 'ILSVRC2012_img_val.tar'
DEVKIT_TAR = ROOT / 'datasets' / 'ILSVRC2012_devkit_t12.tar.gz'
OUT_DIR = ROOT / 'imagenet_val'
CSV_PATH = OUT_DIR / 'imagenet_val_labels.csv'
CLASSES_TSV = OUT_DIR / 'imagenet_classes.tsv'

# Mapping helpers per devkit
# Will produce mapping from image filename -> label index (0..999)

def extract_devkit(devkit_tar: Path, tmp_dir: Path) -> Tuple[List[int], Dict[int, str], Dict[int, str]]:
    """Extract devkit and return (ground_truth_labels, idx_to_wnid, idx_to_name).
    - ground_truth_labels: list of 50k integers in 1..1000
    - idx_to_wnid: mapping ILSVRC2012_ID (1..1000) -> wnid (e.g., 'n01440764')
    - idx_to_name: mapping ILSVRC2012_ID (1..1000) -> human readable name (from 'words')
    """
    with tarfile.open(devkit_tar, 'r:gz') as tar:
        tar.extractall(tmp_dir)
    devkit_dir = Path(tmp_dir) / 'ILSVRC2012_devkit_t12' / 'data'
    gt_file = devkit_dir / 'ILSVRC2012_validation_ground_truth.txt'
    with open(gt_file, 'r') as f:
        gt = [int(x.strip()) for x in f]

    # Prefer parsing meta.mat to build idx->wnid; map_clsloc.txt may not exist in this devkit
    idx_to_wnid: Dict[int, str] = {}
    idx_to_name: Dict[int, str] = {}
    meta_mat = devkit_dir / 'meta.mat'
    if meta_mat.exists():
        try:
            from scipy.io import loadmat  # type: ignore
            import numpy as np  # type: ignore
            meta = loadmat(meta_mat.as_posix(), squeeze_me=False, struct_as_record=False)
            synsets = meta.get('synsets', None)
            if synsets is None:
                # Retry with squeeze
                meta = loadmat(meta_mat.as_posix(), squeeze_me=True, struct_as_record=False)
                synsets = meta.get('synsets', None)
            if synsets is not None:
                arr = synsets
                # synsets is often shape (1,1000) or (1000,1); flatten
                arr = np.array(arr).reshape(-1)
                cnt = 0
                for s in arr:
                    try:
                        # Common patterns:
                        # 1) object with attributes (when struct_as_record=False, squeeze False)
                        if hasattr(s, 'ILSVRC2012_ID') and hasattr(s, 'WNID'):
                            idx = int(np.array(s.ILSVRC2012_ID).flatten()[0])
                            wn = str(np.array(s.WNID).flatten()[0])
                            words = None
                            if hasattr(s, 'words'):
                                try:
                                    words = str(np.array(s.words).flatten()[0])
                                except Exception:
                                    words = None
                        # 2) dict-like access with dtype.names
                        elif hasattr(s, 'dtype') and getattr(s.dtype, 'names', None):
                            names = s.dtype.names
                            if 'ILSVRC2012_ID' in names and 'WNID' in names:
                                idx = int(np.array(s['ILSVRC2012_ID']).flatten()[0])
                                wn = str(np.array(s['WNID']).flatten()[0])
                                words = None
                                if 'words' in names:
                                    try:
                                        words = str(np.array(s['words']).flatten()[0])
                                    except Exception:
                                        words = None
                            else:
                                # Unknown layout, skip entry
                                continue
                        else:
                            # Unknown layout, skip entry
                            continue
                        # Normalize python str
                        wn = wn.strip()
                        if 1 <= idx <= 1000:
                            idx_to_wnid[idx] = wn
                            if words:
                                idx_to_name[idx] = words
                            cnt += 1
                    except Exception:
                        continue
                print(f'Parsed meta.mat synsets -> idx_to_wnid entries: {cnt}')
            else:
                print('Warning: meta.mat loaded but synsets missing')
        except Exception as e:
            print('Warning: failed to parse meta.mat; skipping wnid mapping. Err:', e)
    else:
        print('Warning: meta.mat not found; skipping wnid mapping.')

    return gt, idx_to_wnid, idx_to_name


def extract_val_images(val_tar: Path, out_dir: Path, limit: int = 0):
    """Extract validation JPEGs into out_dir.
    If limit > 0, only extract the first N images in lexicographical order
    (which matches numeric order ILSVRC2012_val_XXXXXXXX.JPEG).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(val_tar, 'r:') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.JPEG') or m.name.endswith('.jpeg')]
        # Keep only basenames (top-level files in tar)
        members.sort(key=lambda m: os.path.basename(m.name))
        if limit and limit > 0:
            members = members[:limit]
        for m in members:
            tar.extract(m, out_dir)


def _load_torchvision_wnid_to_index() -> Dict[str, int] | None:
    """Load ImageNet class mapping (wnid -> class index 0..999).
    Preference order:
    1) assets/imagenet_class_index.json (repo-local, offline)
    2) torchvision.datasets/imagenet_class_index.json (package path)
    3) torchvision.datasets via pkgutil data
    Returns None if unavailable.
    """
    # 1) Prefer repo-local offline asset
    asset_json = ROOT / 'assets' / 'imagenet_class_index.json'
    if asset_json.exists():
        try:
            with open(asset_json, 'r') as f:
                idx_map = json.load(f)
            wnid_to_idx = {v[0]: int(k) for k, v in idx_map.items()}
            print(f"Loaded imagenet_class_index.json from assets ({asset_json}) (classes={len(wnid_to_idx)})")
            return wnid_to_idx
        except Exception as e:
            print('Warning: failed to load assets/imagenet_class_index.json:', e)
    # Try to read from installed torchvision package path first
    try:
        import torchvision  # type: ignore
        tv_path = Path(torchvision.__file__).parent / 'datasets' / 'imagenet_class_index.json'
        if tv_path.exists():
            with open(tv_path, 'r') as f:
                idx_map = json.load(f)
            wnid_to_idx = {v[0]: int(k) for k, v in idx_map.items()}
            print(f"Loaded torchvision imagenet_class_index.json from {tv_path} (classes={len(wnid_to_idx)})")
            return wnid_to_idx
    except Exception as e:
        print('Info: torchvision package path read failed, will try pkgutil. Err:', e)

    # Fallback to pkgutil data loader
    try:
        data = pkgutil.get_data('torchvision.datasets', 'imagenet_class_index.json')
        if not data:
            print('Warning: pkgutil returned no data for imagenet_class_index.json')
            return None
        idx_map = json.loads(data)
        wnid_to_idx = {v[0]: int(k) for k, v in idx_map.items()}
        print(f"Loaded torchvision imagenet_class_index.json via pkgutil (classes={len(wnid_to_idx)})")
        return wnid_to_idx
    except Exception as e:
        print('Warning: failed to load torchvision imagenet_class_index.json via pkgutil:', e)
        return None

def _load_torchvision_index_entries() -> List[Tuple[int, str, str]] | None:
    """Return list of (index, wnid, name) entries.
    Preference order:
    1) assets/imagenet_class_index.json
    2) torchvision package path
    3) pkgutil data
    Returns None if unavailable.
    """
    asset_json = ROOT / 'assets' / 'imagenet_class_index.json'
    if asset_json.exists():
        try:
            with open(asset_json, 'r') as f:
                idx_map = json.load(f)
            entries = [(int(k), v[0], v[1]) for k, v in idx_map.items()]
            entries.sort(key=lambda x: x[0])
            return entries
        except Exception as e:
            print('Warning: failed to parse assets/imagenet_class_index.json:', e)
    try:
        import torchvision  # type: ignore
        tv_path = Path(torchvision.__file__).parent / 'datasets' / 'imagenet_class_index.json'
        if tv_path.exists():
            with open(tv_path, 'r') as f:
                idx_map = json.load(f)
            entries = [(int(k), v[0], v[1]) for k, v in idx_map.items()]
            entries.sort(key=lambda x: x[0])
            return entries
    except Exception:
        pass
    try:
        data = pkgutil.get_data('torchvision.datasets', 'imagenet_class_index.json')
        if not data:
            return None
        idx_map = json.loads(data)
        entries = [(int(k), v[0], v[1]) for k, v in idx_map.items()]
        entries.sort(key=lambda x: x[0])
        return entries
    except Exception as e:
        print('Warning: failed to load torchvision imagenet_class_index.json entries:', e)
        return None


def main():
    parser = argparse.ArgumentParser(description='Prepare ImageNet val set (subset + labels CSV and optional calib/eval split)')
    parser.add_argument('--limit', type=int, default=1000, help='Total images to extract into val/ (0 for all). Default: 1000')
    parser.add_argument('--calib-count', type=int, default=0, help='Number of images for calibration subset (placed under imagenet_val/calib). 0 to disable split.')
    parser.add_argument('--eval-count', type=int, default=0, help='Number of images for evaluation subset (placed under imagenet_val/eval). 0 to disable split.')
    parser.add_argument('--split-mode', type=str, default='head-tail', choices=['head-tail', 'interleave'], help='How to split calib/eval from sorted list. Default: head-tail')
    parser.add_argument('--keep-devkit', dest='keep_devkit', action='store_true', help='Do not delete extracted devkit temp directory')
    parser.add_argument('--reorg', action='store_true', help='Reorganize images into wnid subfolders (val/<wnid>/...). Ignored when calib/eval split is enabled.')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / '_devkit'
    tmp_dir.mkdir(exist_ok=True)
    print('Extracting devkit...')
    gt, idx_to_wnid, idx_to_name = extract_devkit(DEVKIT_TAR, tmp_dir)

    # If split requested, ensure we extract enough images (calib+eval)
    requested_total = args.limit
    if args.calib_count > 0 or args.eval_count > 0:
        need = max(args.calib_count + args.eval_count, 0)
        if requested_total and requested_total < need:
            print(f'Info: limit={requested_total} < calib+eval={need}, increasing limit to {need} for split.')
            requested_total = need
        elif requested_total == 0:
            requested_total = need

    print(f'Extracting validation images (limit={requested_total})...')
    val_dir = OUT_DIR / 'val'
    extract_val_images(VAL_TAR, val_dir, limit=requested_total)

    # The validation images are named ILSVRC2012_val_00000001.JPEG ... 50000
    # Ground truth labels are 1..1000 (1-based). We'll map to 0..999
    # And also write CSV filename,labelIndex
    # Enumerate files (sorted) to match ground-truth ordering
    files = sorted({p.name for p in list(val_dir.glob('*.JPEG')) + list(val_dir.glob('*.jpeg'))})
    if len(files) != len(gt):
        print('Warning: files count and ground truth count differ: ', len(files), len(gt))

    print('Preparing filename→label mapping...')
    wnid_to_idx = _load_torchvision_wnid_to_index()
    # Offline fallback: if torchvision JSON missing, build wnid->index by sorting wnids
    if not wnid_to_idx and idx_to_wnid:
        sorted_wnids = sorted([wn for k, wn in idx_to_wnid.items() if 1 <= k <= 1000])
        wnid_to_idx = {wn: i for i, wn in enumerate(sorted_wnids)}
        print('Built wnid->index mapping by sorted WNIDs (fallback).')
    # Build full mapping: filename -> class index
    fname_to_label = {}
    for i, fname in enumerate(files):
        idx1 = gt[i]
        label_to_write = None
        if idx_to_wnid and wnid_to_idx:
            wnid = idx_to_wnid.get(idx1)
            if wnid and wnid in wnid_to_idx:
                label_to_write = wnid_to_idx[wnid]
        if label_to_write is None:
            label_to_write = idx1 - 1
        fname_to_label[fname] = label_to_write

    # If split requested, create subsets and only write labels for eval subset
    do_split = (args.calib_count > 0 or args.eval_count > 0)
    calib_dir = OUT_DIR / 'calib'
    eval_dir = OUT_DIR / 'eval'
    if do_split:
        calib_list: List[str] = []
        eval_list: List[str] = []
        if args.split_mode == 'head-tail':
            calib_list = files[: max(args.calib_count, 0)]
            eval_list = files[max(args.calib_count, 0) : max(args.calib_count, 0) + max(args.eval_count, 0)]
        else:  # interleave
            # Take alternating samples into calib and eval until counts satisfied
            c_need, e_need = max(args.calib_count, 0), max(args.eval_count, 0)
            for f in files:
                if c_need > 0:
                    calib_list.append(f); c_need -= 1
                    if e_need == 0 and c_need == 0:
                        continue
                if e_need > 0:
                    eval_list.append(f); e_need -= 1
                if c_need == 0 and e_need == 0:
                    break

        calib_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        # Copy files
        for f in calib_list:
            src = val_dir / f
            dst = calib_dir / f
            if src.exists():
                if not dst.exists():
                    shutil.copy2(src, dst)
        for f in eval_list:
            src = val_dir / f
            dst = eval_dir / f
            if src.exists():
                if not dst.exists():
                    shutil.copy2(src, dst)

        # Write manifests
        (OUT_DIR / 'calib_list.txt').write_text('\n'.join(calib_list) + ('\n' if calib_list else ''))
        (OUT_DIR / 'eval_list.txt').write_text('\n'.join(eval_list) + ('\n' if eval_list else ''))

        # Write labels CSV for eval subset only (used by trt_compare)
        with open(CSV_PATH, 'w', newline='') as cf:
            writer = csv.writer(cf)
            for fname in eval_list:
                label = fname_to_label.get(fname)
                if label is not None:
                    writer.writerow([fname, label])
        print('Wrote eval labels CSV to', CSV_PATH)
    else:
        # Backward compatible: write labels for entire extracted subset under val/
        with open(CSV_PATH, 'w', newline='') as cf:
            writer = csv.writer(cf)
            for fname in files:
                writer.writerow([fname, fname_to_label[fname]])
        print('Wrote labels CSV to', CSV_PATH)

    # Also write a classes TSV for C++ tools to load readable names easily
    entries = _load_torchvision_index_entries()
    if entries:
        with open(CLASSES_TSV, 'w', newline='') as f:
            for idx, wnid, name in entries:
                f.write(f"{idx}\t{wnid}\t{name}\n")
        print('Wrote class name mapping to', CLASSES_TSV)
    elif wnid_to_idx and idx_to_name:
        # Fallback: compose entries from meta words and sorted wnid mapping
        # Build wnid -> name via idx
        wnid_to_name = {}
        for idx1, wn in idx_to_wnid.items():
            name = idx_to_name.get(idx1, '')
            wnid_to_name[wn] = name
        # Invert mapping to index order
        inv = sorted(((i, wn) for wn, i in wnid_to_idx.items()), key=lambda x: x[0])
        with open(CLASSES_TSV, 'w', newline='') as f:
            for i, wn in inv:
                name = wnid_to_name.get(wn, wn)
                f.write(f"{i}\t{wn}\t{name}\n")
        print('Wrote class name mapping (fallback, meta words) to', CLASSES_TSV)
    print('Wrote labels CSV to', CSV_PATH)

    # Optional: reorganize into wnid subfolders if mapping is available and enabled
    # When performing calib/eval split, skip reorg to keep subsets稳定
    if (args.calib_count > 0 or args.eval_count > 0):
        print('Split enabled: skip folder reorg to keep calib/eval subsets intact.')
    elif idx_to_wnid and args.reorg:
        print('Reorganizing images into wnid subfolders...')
        moved = 0
        for i, fname in enumerate(files):
            idx1 = gt[i]
            wnid = idx_to_wnid.get(idx1)
            if not wnid:
                continue
            src = val_dir / fname
            dst_dir = val_dir / wnid
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / fname
            try:
                os.replace(src, dst)
                moved += 1
            except FileNotFoundError:
                continue
        print(f'Reorganized {moved} images under {val_dir}')
    else:
        if not idx_to_wnid:
            print('Skipping folder reorg (wnid mapping not available).')
        else:
            print('Skipping folder reorg (not enabled; use --reorg).')

    # Cleanup devkit temp directory unless requested to keep it
    if not args.keep_devkit and tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            print('Removed temporary devkit directory:', tmp_dir)
        except Exception as e:
            print('Warning: failed to remove temp devkit dir:', e)

    print('Done. Next steps:')
    if do_split:
        print('  # INT8 标定建议使用（无需标签）')
        print('  export CALIB_DATA_DIR', '=', calib_dir)
        print('  # 评估与一致性对比（使用 eval 子集 + labels）:')
        print('  ./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt', eval_dir, '1000 --labels', CSV_PATH, '--class-names', CLASSES_TSV)
    else:
        print('  ./bin/trt_compare models/resnet18.trt models/resnet18_int8.trt', val_dir, '200 --labels', CSV_PATH)

if __name__ == '__main__':
    main()
