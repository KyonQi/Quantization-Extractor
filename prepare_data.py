import os
# from datasets import load_dataset
import requests
import tarfile
from tqdm import tqdm

def download_calibration_data(target_root: str) -> str:
    """ Download ImageNet validation dataset for calibration """
    print(f"Downloading ImageNet validation dataset to {target_root}...")

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = "imagenette2-320.tgz"
    filepath = os.path.join(target_root, filename)
    val_dir = os.path.join(target_root, "imagenette2-320", "val")
    final_link_dir = os.path.join(target_root, "imagenet_val_1k")

    if not os.path.exists(target_root):
        os.makedirs(target_root)
    
    # 1. Download and extract dataset
    if not os.path.exists(filepath):
        print(f"Downloading from {url}...")
        
        response = requests.get(url=url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print(f"File {filepath} already exists, skipping download.")

    # Extract
    if not os.path.exists(val_dir):
        print(f"Extracting {filepath}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=target_root)
        print(f"Extracted to {val_dir}.")
    return val_dir
    
if __name__ == "__main__":
    target_root = "./data"
    val_dir = download_calibration_data(target_root=target_root)
    print(f"Validation data is ready at {val_dir}.")