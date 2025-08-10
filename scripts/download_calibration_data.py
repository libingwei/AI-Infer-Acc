import os
import requests
import zipfile
import shutil
from glob import glob
from tqdm import tqdm

def download_and_unzip(url, save_path, extract_path) -> bool:
    """
    Downloads a file from a URL, shows a progress bar, and unzips it.
    Returns True if successful, False otherwise.
    """
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading calibration data from {url}...")

    try:
        # Streaming download with progress bar
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MiB blocks to speed up

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                if not data:
                    continue
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download.")
            return False

        print(f"Download complete. Saved to {save_path}")

        # Unzipping the file
        print(f"Unzipping {save_path} to {extract_path}...")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Unzipping complete.")

        return True

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
        return False
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file is not a valid zip file.")
        return False
    finally:
        # Clean up the downloaded zip file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                print(f"Cleaned up {save_path}.")
            except OSError:
                pass


def images_in_root(path: str) -> bool:
    """Return True if there are images (*.jpg/*.png) directly under path."""
    jpgs = glob(os.path.join(path, "*.jpg"))
    pngs = glob(os.path.join(path, "*.png"))
    return len(jpgs) + len(pngs) > 0


def prepare_flat_subset(extract_path: str, max_images: int = 500) -> None:
    """
    After extraction, ensure calibration_data contains images directly under its root.
    If images are inside a subfolder (e.g., val2017/), copy up to max_images into root.
    """
    if images_in_root(extract_path):
        # Already flat, nothing to do
        count = len(glob(os.path.join(extract_path, "*.jpg"))) + len(glob(os.path.join(extract_path, "*.png")))
        print(f"Found {count} images directly under {extract_path}. No flattening needed.")
        return

    # Find a likely subfolder containing images, e.g., val2017
    candidate_dirs = []
    for name in os.listdir(extract_path):
        full = os.path.join(extract_path, name)
        if os.path.isdir(full):
            candidate_dirs.append(full)

    src_dir = None
    for d in candidate_dirs:
        if glob(os.path.join(d, "*.jpg")) or glob(os.path.join(d, "*.png")):
            src_dir = d
            break

    if src_dir is None:
        print(f"Warning: No images found under {extract_path} after extraction.")
        return

    # Collect images and copy a subset to the root
    imgs = sorted(glob(os.path.join(src_dir, "*.jpg")) + glob(os.path.join(src_dir, "*.png")))
    if not imgs:
        print(f"Warning: No images found under {src_dir}.")
        return

    os.makedirs(extract_path, exist_ok=True)
    limit = min(max_images, len(imgs)) if max_images and max_images > 0 else len(imgs)
    print(f"Flattening: copying {limit} images from {src_dir} to {extract_path} ...")
    copied = 0
    for p in imgs[:limit]:
        dst = os.path.join(extract_path, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
            copied += 1
    print(f"Copied {copied} images to {extract_path}.")

if __name__ == "__main__":
    # Prefer the official COCO 2017 validation set (5k images, ~780MB)
    # We'll flatten and optionally limit the number of images for calibration.
    DATA_URLS = [
        "https://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]

    # Project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ZIP_SAVE_PATH = os.path.join(PROJECT_ROOT, "scripts", "calibration_data.zip")
    EXTRACT_PATH = os.path.join(PROJECT_ROOT, "calibration_data")

    # Allow users to limit number of images via env var (default 500)
    try:
        MAX_IMAGES = int(os.environ.get("CALIB_MAX_IMAGES", "500"))
    except ValueError:
        MAX_IMAGES = 500

    # If images already present at root, skip.
    if os.path.exists(EXTRACT_PATH) and images_in_root(EXTRACT_PATH):
        print(f"Calibration images already present in {EXTRACT_PATH}. Skipping download.")
    else:
        os.makedirs(EXTRACT_PATH, exist_ok=True)

        # Try multiple URLs until one works
        success = False
        for url in DATA_URLS:
            if download_and_unzip(url, ZIP_SAVE_PATH, EXTRACT_PATH):
                success = True
                break
        if not success:
            print("Failed to download COCO val2017 from all mirrors. Please check your network or provide your own images in 'calibration_data/'.")
            raise SystemExit(1)

        # Ensure images are directly under calibration_data and limited in count
        prepare_flat_subset(EXTRACT_PATH, MAX_IMAGES)
